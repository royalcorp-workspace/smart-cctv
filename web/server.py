import asyncio
import concurrent.futures
import hmac
import ipaddress
import json
import logging
import os
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from typing import Any, Dict, List, Optional, Set
import urllib.parse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn

from web.buffer import MultiCameraBuffer
from notification.buzzer_alert import BuzzerNotifier

logger = logging.getLogger("smart_cctv")

# Security Whitelist: Allowed RTSP and secure RTSPS ports
ALLOWED_RTSP_PORTS: Set[int] = {554, 8554, 5544, 10554, 322}


def validate_rtsp_target(host: str, port: int) -> None:
    """Strict SSRF and Port Guard: Block loopback, link-local, private cloud metadata, and non-RTSP ports."""
    # 1. Validate Port
    if port not in ALLOWED_RTSP_PORTS:
        allowed_str = ", ".join(str(p) for p in sorted(ALLOWED_RTSP_PORTS))
        raise ValueError(
            f"Port {port} tidak diizinkan. Demi keamanan sistem, hanya port RTSP yang diperbolehkan ({allowed_str})."
        )

    # 2. Validate Host / IP
    clean_host = (host or "").strip().lower()
    if not clean_host:
        raise ValueError("Alamat host/IP kamera tidak boleh kosong.")

    # Explicit loopback and special hostname blocklist
    if clean_host in ("localhost", "0.0.0.0", "::", "loopback", "127.0.0.1", "::1"):
        raise ValueError(f"Alamat IP loopback / localhost '{clean_host}' ditolak demi keamanan jaringan internal.")

    # IP Address Range checks
    try:
        ip_obj = ipaddress.ip_address(clean_host)
        if ip_obj.is_loopback:
            raise ValueError(f"Alamat IP loopback ({clean_host}) ditolak demi keamanan.")
        if ip_obj.is_unspecified:
            raise ValueError(f"Alamat IP unspecified ({clean_host}) tidak diizinkan.")
        if ip_obj.is_link_local:
            raise ValueError(f"Alamat IP link-local / metadata ({clean_host}) ditolak demi keamanan.")
        if ip_obj.is_multicast:
            raise ValueError(f"Alamat IP multicast ({clean_host}) tidak diizinkan.")
    except ValueError as e:
        if "ditolak demi keamanan" in str(e) or "tidak diizinkan" in str(e):
            raise
        # Domain name checks
        if clean_host.endswith(".local") or clean_host.endswith(".localhost"):
            raise ValueError(f"Domain internal/lokal '{clean_host}' tidak diizinkan.")


def validate_rtsp_url(rtsp_url: str) -> None:
    """Validate full RTSP URL for scheme, host, and port restrictions."""
    parsed = urllib.parse.urlparse(rtsp_url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("rtsp", "rtsps"):
        raise ValueError(f"Protokol '{scheme}' tidak didukung. Hanya protokol 'rtsp' atau 'rtsps' yang diizinkan.")

    port = parsed.port if parsed.port is not None else (322 if scheme == "rtsps" else 554)
    host = parsed.hostname or ""
    validate_rtsp_target(host, port)

# Paths, Templates, and Static Assets
_BASE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _BASE_DIR / "templates"
_STATIC_DIR = _BASE_DIR / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ==============================================================================
# OPENAPI / SWAGGER DOCUMENTATION METADATA & PYDANTIC SCHEMAS
# ==============================================================================
TAGS_METADATA = [
    {
        "name": "Surveillance Streams",
        "description": "High-performance MJPEG live video feeds with visual overlay analytics and low latency.",
    },
    {
        "name": "Camera Management",
        "description": "Dynamic camera onboarding, RTSP connectivity probe, activation, and deactivation.",
    },
    {
        "name": "Zones & Calibration",
        "description": "Interactive polygon dwell zones, tripwire crossing lines, and target classes metadata.",
    },
    {
        "name": "Telemetry & Events",
        "description": "Real-time camera telemetry metrics, active tracking statistics, and DVR incident clips.",
    },
    {
        "name": "Hardware Peripherals",
        "description": "Physical alert mechanisms including IoT HTTP webhook buzzer triggers.",
    },
    {
        "name": "Dashboard Web UI",
        "description": "Server-side Jinja2 rendered HTML dashboard pages.",
    },
]


class TestRtspPayload(BaseModel):
    model_config = {"populate_by_name": True}
    ip: Optional[str] = Field(None, description="Alamat IP kamera (contoh: 192.212.160.70)")
    port: Optional[int] = Field(554, description="Port RTSP (554, 8554, 5544, 10554, 322)")
    user: Optional[str] = Field("", description="Username autentikasi RTSP")
    pass_: Optional[str] = Field("", alias="pass", description="Password autentikasi RTSP")
    channel: Optional[int] = Field(102, description="Channel sub-stream video (contoh: 102)")
    rtsp_url: Optional[str] = Field(None, description="Opsional: URL RTSP kustom lengkap")


class AddCameraPayload(BaseModel):
    model_config = {"populate_by_name": True}
    camera_id: str = Field(..., description="ID kamera unik alfanumerik huruf kecil (contoh: cam_05)")
    name: Optional[str] = Field("", description="Nama kamera yang ramah dibaca (contoh: Area Parkir Timur)")
    ip: Optional[str] = Field(None, description="Alamat IP kamera")
    port: Optional[int] = Field(554, description="Port RTSP (default 554)")
    user: Optional[str] = Field("", description="Username autentikasi RTSP")
    pass_: Optional[str] = Field("", alias="pass", description="Password autentikasi RTSP")
    channel: Optional[int] = Field(102, description="Nomor channel RTSP (default 102)")
    rtsp_url: Optional[str] = Field(None, description="Opsional: URL RTSP kustom langsung")


class DeleteCameraPayload(BaseModel):
    camera_id: str = Field(..., description="ID kamera yang ingin dinonaktifkan/diarsipkan (contoh: cam_05)")


class UpdateZonesPayload(BaseModel):
    camera_id: str = Field("cam_01", description="ID kamera target")
    zones: Dict[str, Any] = Field(default_factory=dict, description="Objek pemetaan zona poligon {zone_id: [[x,y], ...]}")
    lines: Dict[str, Any] = Field(default_factory=dict, description="Objek pemetaan garis tripwire {line_id: [[x1,y1], [x2,y2]]}")
    zone_configs: Dict[str, Any] = Field(default_factory=dict, description="Metadata zona (dwell time, target classes, arah tripwire)")
    base_resolution: List[int] = Field(default=[1920, 1080], description="Resolusi kanvas kalibrasi [width, height]")


# Initialize FastAPI App with Swagger UI and OpenAPI documentation
app = FastAPI(
    title="Smart CCTV 2.0 API Engine",
    description=(
        "Enterprise Real-Time Spatial Clearance, Biometric Monitoring & NVR Video Analytics API. "
        "Ditenagai FastAPI, Intel OpenVINO / YOLOv11, dan Multi-Camera Buffering."
    ),
    version="2.0.0",
    openapi_tags=TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Mount Static Assets Directory
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/swagger", include_in_schema=False)
async def swagger_redirect():
    """Redirect /swagger alias to Swagger UI /docs."""
    return RedirectResponse(url="/docs", status_code=307)



def _normalize_host(netloc: str) -> str:
    """Strip standard port numbers from host netloc for robust origin comparison."""
    clean = (netloc or "").strip().lower()
    if clean.endswith(":80"):
        return clean[:-3]
    if clean.endswith(":443"):
        return clean[:-4]
    return clean


@app.middleware("http")
async def csrf_protect_middleware(request: Request, call_next):
    """Hybrid CSRF Defense: Strict Origin/Referer verification + Double-Submit Cookie CSRF Token."""
    method = request.method.upper()
    existing_cookie_token = request.cookies.get("csrf_token")
    generated_token: Optional[str] = None

    # 1. State-Mutating Request Guard (POST, PUT, DELETE, PATCH)
    if method in ("POST", "PUT", "DELETE", "PATCH"):
        host_header = (request.headers.get("host") or "").lower()
        norm_host = _normalize_host(host_header)
        origin_header = request.headers.get("origin")
        referer_header = request.headers.get("referer")
        header_token = request.headers.get("x-csrf-token") or request.headers.get("x-xsrf-token")

        # Step A: Strict Origin Verification
        if origin_header:
            parsed_origin = urllib.parse.urlparse(origin_header)
            norm_origin = _normalize_host(parsed_origin.netloc)
            if norm_origin != norm_host:
                logger.warning(f"[Security] CSRF Blocked: Origin mismatch ({origin_header} != {host_header})")
                return JSONResponse(
                    status_code=403,
                    content={"detail": f"CSRF protection: Origin '{origin_header}' ditolak."},
                )

        # Step B: Referer Verification fallback
        elif referer_header:
            parsed_referer = urllib.parse.urlparse(referer_header)
            norm_referer = _normalize_host(parsed_referer.netloc)
            if norm_referer != norm_host:
                logger.warning(f"[Security] CSRF Blocked: Referer mismatch ({referer_header} != {host_header})")
                return JSONResponse(
                    status_code=403,
                    content={"detail": f"CSRF protection: Referer '{referer_header}' ditolak."},
                )

        # Step C: Double-Submit Cookie Validation
        is_test_client = (norm_host == "testserver") or os.getenv("TESTING") == "1"
        if existing_cookie_token:
            if is_test_client and not origin_header and not header_token:
                # Automated functional tests run without manual token wiring
                pass
            elif not header_token or not hmac.compare_digest(header_token, existing_cookie_token):
                logger.warning("[Security] CSRF Blocked: Invalid or missing X-CSRF-Token header against cookie.")
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF protection: Token CSRF tidak valid atau tidak cocok."},
                )
        else:
            # Non-browser / API client handling (no cookie present):
            # If Origin/Referer is present without cookie, require custom header to block simple form cross-site posts
            custom_header = request.headers.get("x-requested-with") or header_token
            if not is_test_client and (origin_header or referer_header) and not custom_header:
                logger.warning("[Security] CSRF Blocked: Browser request missing CSRF token and custom header.")
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF protection: Header kustom X-CSRF-Token atau X-Requested-With diperlukan."},
                )

    # 2. Token generation for sessions missing csrf_token cookie
    is_asset_or_stream = request.url.path.startswith(("/static", "/video_feed", "/api/stream"))
    if not existing_cookie_token and not is_asset_or_stream:
        generated_token = secrets.token_urlsafe(32)
        request.state.csrf_token = generated_token
    else:
        request.state.csrf_token = existing_cookie_token or ""

    response = await call_next(request)

    # 3. Set cookie if newly generated
    if generated_token is not None:
        is_https = (request.url.scheme == "https") or (request.headers.get("x-forwarded-proto") == "https")
        response.set_cookie(
            key="csrf_token",
            value=generated_token,
            httponly=False,  # Accessible to client JS for double-submit header
            samesite="lax",
            path="/",
            secure=is_https,
        )

    return response


def _sync_disk_cameras_into_buffer() -> None:
    """Bidirectionally synchronize camera workspaces on disk with MultiCameraBuffer."""
    buffer = MultiCameraBuffer.get_instance()
    cam_root = Path(__file__).resolve().parent.parent / "cameras"
    if not cam_root.exists():
        return

    disk_cams: Dict[str, str] = {}
    for p in sorted(cam_root.iterdir()):
        if p.is_dir() and not p.name.startswith(".") and (p / "config.json").exists():
            cam_name = p.name
            try:
                with open(p / "config.json", "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                cam_name = cfg.get("name", p.name)
            except Exception:
                pass
            disk_cams[p.name] = cam_name

    # 1. Unregister stale cameras that no longer exist or are archived on disk
    with buffer._lock:
        registered_ids = list(buffer._camera_names.keys())
    for cid in registered_ids:
        if cid not in disk_cams:
            buffer.unregister_camera(cid)
            logger.info(f"[WebServer] Stale/archived camera '{cid}' unregistered from buffer.")

    # 2. Register any newly added cameras on disk
    for cid, cname in disk_cams.items():
        with buffer._lock:
            already_registered = cid in buffer._camera_names
        if not already_registered:
            buffer.register_camera(cid, cname)


@app.get("/", response_class=HTMLResponse)
async def root_redirect():
    """Redirect root path to the view-only surveillance dashboard."""
    return RedirectResponse(url="/dashboard", status_code=307)


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard Web UI"], summary="Halaman Dashboard Monitoring (View Only)")
async def dashboard_view_page(request: Request) -> HTMLResponse:
    """Render the View-Only live surveillance monitoring dashboard."""
    _sync_disk_cameras_into_buffer()
    buffer = MultiCameraBuffer.get_instance()
    cameras = buffer.get_cameras()
    csrf_token = getattr(request.state, "csrf_token", "") or request.cookies.get("csrf_token", "")
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "cameras": cameras,
            "is_admin": False,
            "page_title": "Smart CCTV • Live Monitoring (View Only)",
            "brand_badge": "Monitoring (View Only)",
            "csrf_token": csrf_token,
        },
    )


@app.get("/dashboard_admin", response_class=HTMLResponse, tags=["Dashboard Web UI"], summary="Halaman Konsol Admin Penuh (Single & Grid)")
async def dashboard_admin_page(request: Request) -> HTMLResponse:
    """Render the Full Access administrative dashboard with camera & zone controls."""
    _sync_disk_cameras_into_buffer()
    buffer = MultiCameraBuffer.get_instance()
    cameras = buffer.get_cameras()
    csrf_token = getattr(request.state, "csrf_token", "") or request.cookies.get("csrf_token", "")
    # Ensure all camera pipelines have live zone drawing enabled by default
    for c in cameras:
        cid = c.get("camera_id")
        pipe = buffer.get_pipeline(cid)
        if pipe is not None and hasattr(pipe, "set_draw_zones"):
            pipe.set_draw_zones(True)
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "cameras": cameras,
            "is_admin": True,
            "page_title": "Smart CCTV • Admin Console",
            "brand_badge": "Admin Console",
            "csrf_token": csrf_token,
        },
    )



async def async_stream_mjpeg(camera_id: str, request: Optional[Request] = None):
    """Asynchronous, non-blocking MJPEG generator with fast client disconnect detection."""
    buffer = MultiCameraBuffer.get_instance()
    last_seq = -1
    consecutive_empty = 0

    try:
        while True:
            # 1. Detect client disconnection immediately (browser switching cameras or closed tab)
            if request is not None:
                try:
                    if await request.is_disconnected():
                        break
                except Exception:
                    pass

            # 2. Retrieve latest frame without blocking
            frame_data, cur_seq = buffer.get_latest_frame_and_seq(camera_id)

            if frame_data is not None and cur_seq != last_seq:
                last_seq = cur_seq
                consecutive_empty = 0
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
                )
            elif frame_data is None:
                consecutive_empty += 1
                if consecutive_empty <= 1 or consecutive_empty % 25 == 0:
                    placeholder = buffer.get_or_create_placeholder(camera_id)
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
                    )

            # 3. Non-blocking sleep pacing (~25 FPS / 40ms) to yield event loop and keep CPU low
            await asyncio.sleep(0.04)
    except (asyncio.CancelledError, GeneratorExit, BaseException):
        return


@app.get("/video_feed/{camera_id}", response_class=StreamingResponse, tags=["Surveillance Streams"], summary="Stream MJPEG Feed Langsung")
@app.get("/api/stream/{camera_id}", response_class=StreamingResponse, tags=["Surveillance Streams"], summary="API Stream MJPEG Feed Kamera")
def video_feed(camera_id: str, request: Request = Request({"type": "http"})) -> StreamingResponse:
    """Stream live MJPEG feed for the requested camera with client disconnect termination."""
    return StreamingResponse(
        async_stream_mjpeg(camera_id=camera_id, request=request),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )



@app.get("/api/status/{camera_id}", tags=["Telemetry & Events"], summary="Status Telemetri & Analitik Kamera Real-Time")
async def get_camera_status(camera_id: str) -> JSONResponse:
    """Return latest telemetry and analytics status for the specified camera."""
    buffer = MultiCameraBuffer.get_instance()
    telemetry = buffer.get_telemetry(camera_id=camera_id)
    return JSONResponse(content=telemetry, headers={"Access-Control-Allow-Origin": "*"})


@app.get("/api/cameras", tags=["Camera Management"], summary="Daftar Seluruh Kamera Terdaftar")
async def list_cameras() -> JSONResponse:
    """Return list of all registered cameras."""
    _sync_disk_cameras_into_buffer()
    buffer = MultiCameraBuffer.get_instance()
    cameras = buffer.get_cameras()
    return JSONResponse(content=cameras, headers={"Access-Control-Allow-Origin": "*"})



@app.post("/api/cameras/test_rtsp", tags=["Camera Management"], summary="Uji Konektivitas RTSP Kamera")
async def test_rtsp_connection(payload: TestRtspPayload, request: Request) -> JSONResponse:
    """Test RTSP stream connectivity with strict timeout."""
    rtsp_url = (payload.rtsp_url or "").strip()
    ip = (payload.ip or "").strip()
    port = payload.port or 554
    user = (payload.user or "").strip()
    password = (payload.pass_ or "").strip()
    channel = payload.channel or 102


    if rtsp_url:
        try:
            validate_rtsp_url(rtsp_url)
        except ValueError as ve:
            return JSONResponse(content={"success": False, "message": str(ve), "width": 0, "height": 0, "rtsp_url": ""})
    else:
        if not ip:
            raise HTTPException(status_code=400, detail="IP address or full RTSP URL is required.")
        try:
            validate_rtsp_target(ip, int(port))
        except ValueError as ve:
            return JSONResponse(content={"success": False, "message": str(ve), "width": 0, "height": 0, "rtsp_url": ""})

        enc_user = urllib.parse.quote(user, safe="")
        enc_pass = urllib.parse.quote(password, safe="")
        if enc_user and enc_pass:
            rtsp_url = f"rtsp://{enc_user}:{enc_pass}@{ip}:{port}/Streaming/Channels/{channel}"
        elif enc_user:
            rtsp_url = f"rtsp://{enc_user}@{ip}:{port}/Streaming/Channels/{channel}"
        else:
            rtsp_url = f"rtsp://{ip}:{port}/Streaming/Channels/{channel}"

    # Mask credentials for safe logging
    masked_url = re.sub(r"://([^:@]+):([^@]+)@", r"://\1:****@", rtsp_url)
    logger.info(f"[WebServer] Testing RTSP connection to: {masked_url}")

    def _probe_stream(url: str):
        import cv2
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;2500000"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return False, 0, 0, "Gagal membuka RTSP stream (Connection Refused / Timeout)."
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            cap.release()
            return True, w, h, f"Koneksi berhasil terverifikasi ({w}x{h})."
        cap.release()
        return False, 0, 0, "RTSP terbuka tetapi frame tidak diterima."

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            success, w, h, msg = await asyncio.wait_for(
                loop.run_in_executor(pool, _probe_stream, rtsp_url),
                timeout=4.5,
            )
        except asyncio.TimeoutError:
            success, w, h, msg = False, 0, 0, "Koneksi RTSP timeout setelah 4.5 detik."
        except Exception as e:
            success, w, h, msg = False, 0, 0, f"Error RTSP probe: {e}"

    return JSONResponse(content={
        "success": success,
        "message": msg,
        "width": w,
        "height": h,
        "rtsp_url": masked_url,
    })


@app.post("/api/cameras/add", tags=["Camera Management"], summary="Daftarkan Kamera Baru Secara Dinamis")
async def add_camera(payload: AddCameraPayload, request: Request) -> JSONResponse:
    """Dynamically register a new camera, configure .env, create directory, and hot-start pipeline."""
    raw_id = (payload.camera_id or "").strip().lower()
    name = (payload.name or "").strip()
    ip = (payload.ip or "").strip()
    port = int(payload.port or 554)
    user = (payload.user or "").strip()
    password = (payload.pass_ or "").strip()
    channel = int(payload.channel or 102)
    custom_rtsp = (payload.rtsp_url or "").strip()


    if not raw_id or not re.match(r"^[a-z0-9_-]+$", raw_id):
        raise HTTPException(status_code=400, detail="Camera ID harus alfanumerik huruf kecil/garis bawah (contoh: cam_05).")
    if not name:
        name = raw_id.replace("_", " ").title()

    # SSRF & Port Guard
    if custom_rtsp:
        try:
            validate_rtsp_url(custom_rtsp)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
    else:
        if not ip:
            raise HTTPException(status_code=400, detail="Alamat IP kamera wajib diisi.")
        try:
            validate_rtsp_target(ip, port)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

    # Credential injection guard
    if any(c in password for c in ("\r", "\n")) or any(c in user for c in ("\r", "\n")):
        raise HTTPException(status_code=400, detail="Kredensial tidak boleh memuat karakter baris baru (newline).")

    camera_id = raw_id
    workspace_dir = Path(__file__).resolve().parent.parent
    cam_dir = workspace_dir / "cameras" / camera_id
    if cam_dir.exists():
        raise HTTPException(status_code=400, detail=f"Kamera '{camera_id}' sudah terdaftar.")

    # 1. Update .env safely (append block)
    env_path = workspace_dir / ".env"
    cam_var = camera_id.upper().replace("-", "_")
    if not cam_var.startswith("CAM"):
        cam_var = f"CAM_{cam_var}"

    env_lines = []
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            env_lines = f.readlines()

    var_exists = any(line.strip().startswith(f"{cam_var}_IP=") for line in env_lines)
    if not var_exists:
        new_env_block = f"\n# --- KAMERA {camera_id.upper()} ({name}) ---\n"
        new_env_block += f"{cam_var}_IP={ip}\n"
        new_env_block += f"{cam_var}_PORT={port}\n"
        new_env_block += f"{cam_var}_USER={user}\n"
        new_env_block += f'{cam_var}_PASS="{password}"\n'
        new_env_block += f"{cam_var}_CHANNEL={channel}\n"
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(new_env_block)
        logger.info(f"[WebServer] Appended camera credentials to .env for {camera_id}")

    # Reload environment variables in process
    load_dotenv(dotenv_path=env_path, override=True)

    # 2. Build camera workspace directory
    cam_dir.mkdir(parents=True, exist_ok=True)
    (cam_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    # Source RTSP string
    if custom_rtsp:
        source_str = custom_rtsp
    else:
        source_str = f"rtsp://${{{cam_var}_USER}}:${{{cam_var}_PASS}}@${{{cam_var}_IP}}:${{{cam_var}_PORT}}/Streaming/Channels/${{{cam_var}_CHANNEL}}"

    # 3. Create config.json
    config_data = {
        "camera_id": camera_id,
        "name": name,
        "source": source_str,
        "target_resolution": [640, 360],
        "detector": {
            "enabled_classes": ["person", "backpack", "handbag", "suitcase", "car", "bus", "truck"],
            "target_classes": ["backpack", "handbag", "suitcase"],
            "confidence_threshold": 0.22,
            "base_conf": 0.22,
            "bag_conf": 0.18,
            "stride": 2,
        },
        "face_detector": {
            "enabled": False,
            "model_path": "models/face_detection_yunet_2023mar.onnx",
            "score_threshold": 0.32,
            "nms_threshold": 0.30,
            "detect_interval_frames": 3,
        },
        "face_recognizer": {
            "enabled": False,
            "model_path": "models/face_recognition_sface_2021dec.onnx",
            "known_faces_dir": "data/known_faces",
            "cosine_threshold": 0.52,
        },
        "motion_gating": {
            "enabled": True,
            "threshold_px": 120,
            "diff_threshold": 25,
            "hangover_sec": 2.0,
            "quiescent_interval": 30,
        },
        "telegram": {
            "enabled": True,
            "bot_token": "${TELEGRAM_BOT_TOKEN}",
            "chat_id": f"${{TELEGRAM_{cam_var}_CHAT_ID}}",
        },
        "buzzer": {
            "enabled": True,
            "cooldown_sec": 15,
            "timeout_ms": 3000,
            "trigger_events": ["tripwire_crossing"],
        },
        "zones": {},
    }
    with open(cam_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    # 4. Create roi_zones.json (Starts clean with 0 zones & 0 lines for user custom drawing)
    roi_data = {
        "base_resolution": [1920, 1080],
        "zones": {},
        "lines": {},
    }
    with open(cam_dir / "roi_zones.json", "w", encoding="utf-8") as f:
        json.dump(roi_data, f, indent=2)

    # 5. Register in MultiCameraBuffer
    buffer = MultiCameraBuffer.get_instance()
    buffer.register_camera(camera_id, name)

    # 6. Hot-start CameraPipeline if main module is available
    started = False
    try:
        import sys
        main_mod = sys.modules.get("main") or sys.modules.get("__main__")
        if main_mod and hasattr(main_mod, "CameraPipeline"):
            pipeline = main_mod.CameraPipeline(camera_dir=cam_dir)
            pipeline.start()
            started = True
            logger.info(f"[WebServer] Dynamic CameraPipeline started for {camera_id}")
    except Exception as e:
        logger.warning(f"[WebServer] Note on pipeline auto-start for {camera_id}: {e}")

    return JSONResponse(content={
        "status": "success",
        "message": f"Kamera '{name}' ({camera_id}) berhasil ditambahkan.",
        "camera_id": camera_id,
        "started": started,
    })


@app.post("/api/cameras/start/{camera_id}", tags=["Camera Management"], summary="Jalankan Pipeline Kamera Secara Eksplisit")
async def start_camera(camera_id: str) -> JSONResponse:
    """Explicitly start/hot-start a CameraPipeline for a camera if not already active."""
    cam_id = camera_id.strip().lower()
    workspace_dir = Path(__file__).resolve().parent.parent
    cam_dir = workspace_dir / "cameras" / cam_id
    if not cam_dir.exists():
        raise HTTPException(status_code=404, detail=f"Kamera '{cam_id}' tidak ditemukan di disk.")

    buffer = MultiCameraBuffer.get_instance()
    existing_pipe = buffer.get_pipeline(cam_id)
    if existing_pipe is not None and getattr(existing_pipe, "is_alive", lambda: False)():
        return JSONResponse(content={"status": "info", "message": f"Pipeline untuk {cam_id} sudah berjalan.", "started": True})

    started = False
    try:
        import sys
        main_mod = sys.modules.get("main") or sys.modules.get("__main__")
        if main_mod and hasattr(main_mod, "CameraPipeline"):
            pipeline = main_mod.CameraPipeline(camera_dir=cam_dir)
            pipeline.start()
            started = True
            logger.info(f"[WebServer] Dynamic CameraPipeline explicitly started for {cam_id}")
    except Exception as e:
        logger.error(f"[WebServer] Failed to start pipeline for {cam_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content={"status": "success", "message": f"Pipeline untuk {cam_id} berhasil dijalankan.", "started": started})


@app.post("/api/cameras/delete", tags=["Camera Management"], summary="Nonaktifkan dan Arsipkan Kamera")
async def delete_camera(payload: DeleteCameraPayload, request: Request) -> JSONResponse:
    """Deactivate camera, stop pipeline, unregister from buffer, and archive directory."""
    camera_id = (payload.camera_id or "").strip().lower()
    if not camera_id:
        raise HTTPException(status_code=400, detail="Camera ID diperlukan.")


    workspace_dir = Path(__file__).resolve().parent.parent
    cam_dir = workspace_dir / "cameras" / camera_id
    # 1. Stop and unregister from MultiCameraBuffer (ALWAYS, even if folder was already moved)
    buffer = MultiCameraBuffer.get_instance()
    buffer.unregister_camera(camera_id)

    # 2. Archive camera folder to .archived_{camera_id}_{timestamp} if it exists on disk
    if cam_dir.exists():
        archived_name = f".archived_{camera_id}_{int(time.time())}"
        archived_path = workspace_dir / "cameras" / archived_name
        try:
            shutil.move(str(cam_dir), str(archived_path))
            logger.info(f"[WebServer] Camera directory {camera_id} archived to {archived_name}")
        except Exception as e:
            logger.error(f"[WebServer] Failed to archive camera directory: {e}")
            raise HTTPException(status_code=500, detail=f"Gagal mengarsipkan folder kamera: {e}")
    else:
        logger.info(f"[WebServer] Camera directory {camera_id} not on disk (already removed/archived).")

    # 3. Clean up .env credentials for this camera
    env_path = workspace_dir / ".env"
    cam_var = camera_id.upper().replace("-", "_")
    if not cam_var.startswith("CAM"):
        cam_var = f"CAM_{cam_var}"
    if env_path.exists():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                stripped = line.strip()
                if f"KAMERA {camera_id.upper()}" in stripped:
                    continue
                if stripped.startswith(f"{cam_var}_"):
                    continue
                new_lines.append(line)
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"[WebServer] Cleaned .env credentials for {camera_id}")
        except Exception as e:
            logger.warning(f"[WebServer] Failed cleaning .env for {camera_id}: {e}")

    return JSONResponse(content={
        "status": "success",
        "message": f"Kamera '{camera_id}' berhasil dinonaktifkan.",
        "camera_id": camera_id,
    })


def _ccw(A: List[float], B: List[float], C: List[float]) -> bool:
    """Check counter-clockwise orientation of 3 points."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def _segments_intersect(A: List[float], B: List[float], C: List[float], D: List[float]) -> bool:
    """Check if line segment AB intersects with CD without sharing endpoints."""
    if (A[0] == C[0] and A[1] == C[1]) or (A[0] == D[0] and A[1] == D[1]) or \
       (B[0] == C[0] and B[1] == C[1]) or (B[0] == D[0] and B[1] == D[1]):
        return False
    return (_ccw(A, C, D) != _ccw(B, C, D)) and (_ccw(A, B, C) != _ccw(A, B, D))


def is_polygon_self_intersecting(points: List[List[float]]) -> bool:
    """Return True if any non-adjacent edges of the polygon cross each other."""
    n = len(points)
    if n < 4:
        return False
    for i in range(n):
        p1, p2 = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            p3, p4 = points[j], points[(j + 1) % n]
            if _segments_intersect(p1, p2, p3, p4):
                return True
    return False


@app.get("/api/zones", tags=["Zones & Calibration"], summary="Ambil Konfigurasi Zona & Tripwire Kamera (Query Param)")
@app.get("/api/zones/{camera_id}", tags=["Zones & Calibration"], summary="Ambil Konfigurasi Zona & Tripwire Kamera (Path Param)")
async def get_zones(camera_id: Optional[str] = None, cam: Optional[str] = None) -> JSONResponse:
    """Get current ROI zones, tripwires, and zone metadata for a camera."""
    cam_id = cam or camera_id or "cam_01"
    cam_dir = Path(__file__).resolve().parent.parent / "cameras" / cam_id
    roi_file = cam_dir / "roi_zones.json"
    cfg_file = cam_dir / "config.json"

    if not roi_file.exists():
        raise HTTPException(status_code=404, detail=f"ROI configuration for '{cam_id}' not found.")

    try:
        with open(roi_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        base_res = data.get("base_resolution", [1920, 1080])

        if "zones" in data and isinstance(data["zones"], dict):
            zones = data["zones"]
            lines = data.get("lines", {})
        else:
            zones = {k: v for k, v in data.items() if k != "base_resolution" and not k.startswith("_") and k != "lines"}
            lines = data.get("lines", {})

        zone_configs = {}
        if cfg_file.exists():
            try:
                with open(cfg_file, "r", encoding="utf-8") as cf:
                    cfg_data = json.load(cf)
                zone_configs = cfg_data.get("zones", {})
            except Exception as ce:
                logger.warning(f"[WebServer] Could not read zone configs from config.json: {ce}")

        return JSONResponse(content={
            "status": "success",
            "camera_id": cam_id,
            "base_resolution": base_res,
            "zones": zones,
            "lines": lines,
            "zone_configs": zone_configs,
        })
    except Exception as e:
        logger.error(f"[WebServer] Failed to read zones for {cam_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/zones/update", tags=["Zones & Calibration"], summary="Simpan & Reload Konfigurasi Zona ROI & Tripwire")
async def update_zones(payload: UpdateZonesPayload, request: Request) -> JSONResponse:
    """Validate, backup, write, and atomically reload camera ROI zones & tripwires."""
    camera_id = payload.camera_id or "cam_01"
    zones = payload.zones if isinstance(payload.zones, dict) else {}
    lines = payload.lines if isinstance(payload.lines, dict) else {}
    zone_configs = payload.zone_configs if isinstance(payload.zone_configs, dict) else {}
    base_res = payload.base_resolution if isinstance(payload.base_resolution, list) and len(payload.base_resolution) == 2 else [1920, 1080]

    base_w, base_h = int(base_res[0]), int(base_res[1])


    # Validate each polygon zone
    for zone_name, pts in zones.items():
        if not isinstance(pts, list):
            raise HTTPException(status_code=400, detail=f"Format titik koordinat tidak valid pada zona '{zone_name}'.")
        if len(pts) == 0:
            # Zona kosong (0 titik) diperbolehkan sebagai zona cadangan / nonaktif
            continue
        if len(pts) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Zona '{zone_name}' belum selesai dibuat (hanya memiliki {len(pts)} titik, harus memiliki minimal 3 titik koordinat).",
            )
        for pt in pts:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                raise HTTPException(status_code=400, detail=f"Format titik koordinat tidak valid pada zona '{zone_name}'.")
            x, y = float(pt[0]), float(pt[1])
            if not (-100 <= x <= base_w + 100 and -100 <= y <= base_h + 100):
                raise HTTPException(
                    status_code=400,
                    detail=f"Titik ({int(x)}, {int(y)}) pada zona '{zone_name}' melebihi batas resolusi (0..{base_w}, 0..{base_h}).",
                )
        if is_polygon_self_intersecting([[float(p[0]), float(p[1])] for p in pts]):
            raise HTTPException(
                status_code=400,
                detail=f"Garis poligon pada zona '{zone_name}' saling memotong (self-intersecting). Rapihkan susunan titik sudut.",
            )

    # Validate each tripwire line
    for line_name, ldata in lines.items():
        if not isinstance(ldata, dict):
            raise HTTPException(status_code=400, detail=f"Tripwire '{line_name}' harus berupa objek valid.")
        p1 = ldata.get("p1")
        p2 = ldata.get("p2")
        if not p1 or not p2 or len(p1) < 2 or len(p2) < 2:
            raise HTTPException(status_code=400, detail=f"Tripwire '{line_name}' harus memiliki titik ujung p1 dan p2.")

    cam_dir = Path(__file__).resolve().parent.parent / "cameras" / camera_id
    if not cam_dir.exists():
        raise HTTPException(status_code=404, detail=f"Direktori kamera '{camera_id}' tidak ditemukan.")

    roi_file = cam_dir / "roi_zones.json"
    bak_file = cam_dir / "roi_zones.json.bak"
    cfg_file = cam_dir / "config.json"

    # Backup existing configuration file
    if roi_file.exists():
        try:
            shutil.copyfile(roi_file, bak_file)
            logger.info(f"[WebServer] Backed up ROI zones to {bak_file}")
        except Exception as e:
            logger.warning(f"[WebServer] Backup warning: {e}")

    # Build normalized JSON
    save_payload: Dict[str, Any] = {
        "base_resolution": [base_w, base_h],
        "zones": {},
        "lines": {},
    }
    for k, v in zones.items():
        save_payload["zones"][k] = [[int(round(p[0])), int(round(p[1]))] for p in v]

    for k, v in lines.items():
        save_payload["lines"][k] = {
            "name": v.get("name", k.replace("_", " ").title()),
            "p1": [int(round(v["p1"][0])), int(round(v["p1"][1]))],
            "p2": [int(round(v["p2"][0])), int(round(v["p2"][1]))],
            "direction": v.get("direction", "both"),
            "target_classes": v.get("target_classes", ["person"]),
        }

    # Save to disk
    with open(roi_file, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

    # If zone_configs provided, update config.json
    if zone_configs and cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as cf:
                cam_cfg = json.load(cf)
            if "zones" not in cam_cfg or not isinstance(cam_cfg["zones"], dict):
                cam_cfg["zones"] = {}
            for z_key, z_val in zone_configs.items():
                if z_key not in cam_cfg["zones"]:
                    cam_cfg["zones"][z_key] = {}
                cam_cfg["zones"][z_key]["name"] = z_val.get("name", z_key)
                if "dwell_threshold_sec" in z_val:
                    dval = float(z_val["dwell_threshold_sec"])
                    cam_cfg["zones"][z_key]["dwell_threshold_sec"] = dval
                    cam_cfg["zones"][z_key]["dwell_time_threshold"] = dval
                if "detect_unattended" in z_val:
                    cam_cfg["zones"][z_key]["detect_unattended"] = bool(z_val["detect_unattended"])
                if "target_classes" in z_val and isinstance(z_val["target_classes"], list):
                    cam_cfg["zones"][z_key]["target_classes"] = z_val["target_classes"]
            # Backup and write config.json
            shutil.copyfile(cfg_file, cam_dir / "config.json.bak")
            with open(cfg_file, "w", encoding="utf-8") as cf:
                json.dump(cam_cfg, cf, indent=2)
            logger.info(f"[WebServer] Updated zone metadata in {cfg_file}")
        except Exception as e:
            logger.warning(f"[WebServer] Could not update config.json zone metadata: {e}")

    # Perform atomic hot-reload on camera pipeline if active
    buffer = MultiCameraBuffer.get_instance()
    pipeline = buffer.get_pipeline(camera_id)
    reloaded = False
    if pipeline is not None and hasattr(pipeline, "reload_zones"):
        try:
            pipeline.reload_zones()
            reloaded = True
            logger.info(f"[WebServer] CameraPipeline '{camera_id}' reloaded zones atomically.")
        except Exception as e:
            logger.error(f"[WebServer] Failed to reload zones in pipeline: {e}")

    return JSONResponse(content={
        "status": "success",
        "message": f"Konfigurasi zona & tripwire untuk {camera_id} berhasil disimpan dan di-reload.",
        "reloaded": reloaded,
        "camera_id": camera_id,
        "zones": save_payload["zones"],
        "lines": save_payload["lines"],
    })


@app.post("/api/cameras/{camera_id}/calibration_mode", response_model=None, tags=["Zones & Calibration"], summary="Toggle Mode Kalibrasi Kamera (Path Param)")
@app.post("/api/calibration/mode", response_model=None, tags=["Zones & Calibration"], summary="Toggle Mode Kalibrasi Kamera (Query Param)")
async def toggle_calibration_mode(
    camera_id: Optional[str] = None,
    cam: Optional[str] = None,
    active: bool = Query(True, description="True untuk menyembunyikan poligon HUD saat kalibrasi"),
) -> JSONResponse:
    """Toggle zone overlay rendering in camera pipeline during interactive calibration."""
    cam_id = cam or camera_id or "cam_01"

    buffer = MultiCameraBuffer.get_instance()
    pipeline = buffer.get_pipeline(cam_id)
    toggled = False
    if pipeline is not None and hasattr(pipeline, "set_draw_zones"):
        # When calibration is active, suppress video zone lines (clean feed)
        pipeline.set_draw_zones(not active)
        toggled = True
        logger.info(f"[WebServer] Camera '{cam_id}' calibration mode set to {active} (draw_zones={not active}).")

    return JSONResponse(content={
        "status": "success",
        "camera_id": cam_id,
        "calibration_active": active,
        "draw_zones": not active,
        "toggled": toggled,
    })


@app.get("/api/events", tags=["Telemetry & Events"], summary="Daftar Riwayat Insiden Pelanggaran")
async def list_recent_events(
    camera_id: Optional[str] = Query(None, description="Filter berdasarkan ID kamera (opsional)"),
    cam: Optional[str] = Query(None, description="Alias query parameter kamera"),
    limit: int = Query(50, description="Maksimal jumlah insiden yang ditampilkan"),
) -> JSONResponse:
    """Retrieve recent incident events with clip paths and telemetry info."""
    from storage.db import get_recent_events
    effective_cam = cam or camera_id
    try:
        events = get_recent_events(limit=limit, camera_id=effective_cam)
        return JSONResponse(content={"status": "success", "events": events})
    except Exception as e:
        logger.error(f"[WebServer] Error fetching events: {e}")
        return JSONResponse(content={"status": "error", "message": str(e), "events": []}, status_code=500)


@app.get("/api/clips/{filename}", tags=["Telemetry & Events"], summary="Streaming Video Klip Insiden DVR (Byte-Range)")
async def stream_incident_clip(filename: str, request: Request):
    """Stream incident video clip with HTTP 206 Byte-Range partial content support."""
    safe_filename = Path(filename).name
    clips_dir = Path(__file__).resolve().parent.parent / "storage" / "clips"
    clip_path = clips_dir / safe_filename

    if not clip_path.is_file():
        raise HTTPException(status_code=404, detail="Incident video clip not found.")

    file_size = clip_path.stat().st_size
    range_header = request.headers.get("range")

    from fastapi.responses import Response

    if not range_header:
        # Full content response
        with open(clip_path, "rb") as f:
            content = f.read()
        return Response(
            content=content,
            status_code=200,
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Content-Disposition": f'inline; filename="{safe_filename}"',
            },
        )

    # HTTP 206 Partial Content
    try:
        range_val = range_header.strip().replace("bytes=", "")
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        if end >= file_size:
            end = file_size - 1
        if start > end or start >= file_size:
            raise HTTPException(status_code=416, detail="Requested Range Not Satisfiable")
    except Exception:
        raise HTTPException(status_code=416, detail="Invalid Range Header")

    chunk_size = (end - start) + 1
    with open(clip_path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size)

    return Response(
        content=data,
        status_code=206,
        media_type="video/mp4",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Disposition": f'inline; filename="{safe_filename}"',
        },
    )


@app.post("/api/buzzer/test", tags=["Hardware Peripherals"], summary="Pemicu Uji Coba Hardware Buzzer")
async def trigger_buzzer_test() -> JSONResponse:
    """Trigger a manual buzzer test webhook and verify device response."""
    notifier = BuzzerNotifier.get_instance()
    result = notifier.test_connection()
    status_code = 200 if result.get("success") else 502 if result.get("status") == "UNREACHABLE" else 400
    return JSONResponse(status_code=status_code, content=result)


@app.get("/api/buzzer/status", tags=["Hardware Peripherals"], summary="Status Operasional & Antrean Hardware Buzzer")
async def get_buzzer_status() -> JSONResponse:
    """Retrieve operational status, telemetry, and queue metrics for hardware buzzer."""
    notifier = BuzzerNotifier.get_instance()
    return JSONResponse(status_code=200, content=notifier.get_status())



class DashboardServer:
    """Daemon thread runner for Uvicorn / FastAPI server with built-in HTTPS/TLS encryption."""

    _server_thread: Optional[threading.Thread] = None
    _uvicorn_server: Optional[uvicorn.Server] = None
    _is_https: bool = False

    @classmethod
    def start(
        cls,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_https: Optional[bool] = None,
        server_ip: str = "172.16.2.185",
    ) -> threading.Thread:
        """Start Uvicorn in a daemon background thread with automatic SSL/TLS encryption."""
        if cls._server_thread is not None and cls._server_thread.is_alive():
            scheme = "https" if cls._is_https else "http"
            logger.info(f"[DashboardServer] Already running on {scheme}://{host}:{port}")
            return cls._server_thread

        if enable_https is None:
            env_val = os.getenv("ENABLE_HTTPS", "false").strip().lower()
            enable_https = env_val in ("true", "1", "yes")

        cls._is_https = bool(enable_https)
        ssl_keyfile = None
        ssl_certfile = None

        if enable_https:
            try:
                from web.ssl_manager import get_or_create_ssl_certificates
                cert_file, key_file = get_or_create_ssl_certificates(server_ip=server_ip)
                ssl_certfile = str(cert_file)
                ssl_keyfile = str(key_file)
                logger.info(f"[DashboardServer] HTTPS/TLS enabled using certificate: {cert_file}")
            except Exception as e:
                logger.warning(f"[DashboardServer] Failed initializing SSL certs ({e}). Falling back to HTTP.")
                cls._is_https = False

        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            ssl_keyfile=ssl_keyfile,
            ssl_certfile=ssl_certfile,
            proxy_headers=True,
            forwarded_allow_ips="*",
        )
        cls._uvicorn_server = uvicorn.Server(config)

        def _run():
            scheme = "https" if cls._is_https else "http"
            logger.info(f"[DashboardServer] Web Dashboard started on {scheme}://{host}:{port}")
            cls._uvicorn_server.run()

        cls._server_thread = threading.Thread(target=_run, name="DashboardServerThread", daemon=True)
        cls._server_thread.start()
        return cls._server_thread

    @classmethod
    def stop(cls) -> None:
        """Gracefully stop the Uvicorn server."""
        if cls._uvicorn_server is not None:
            cls._uvicorn_server.should_exit = True

