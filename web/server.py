import asyncio
import concurrent.futures
import json
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.parse

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from web.buffer import MultiCameraBuffer
from notification.buzzer_alert import BuzzerNotifier

logger = logging.getLogger("smart_cctv")

# Paths, Templates, and Static Assets
_BASE_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _BASE_DIR / "templates"
_STATIC_DIR = _BASE_DIR / "static"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# Initialize FastAPI App
app = FastAPI(
    title="Smart CCTV 2.0 Web Dashboard",
    description="Real-Time Spatial Clearance & Biometric Monitoring Dashboard",
    version="2.0.0",
)

# Mount Static Assets Directory
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


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


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_view_page(request: Request) -> HTMLResponse:
    """Render the View-Only live surveillance monitoring dashboard."""
    _sync_disk_cameras_into_buffer()
    buffer = MultiCameraBuffer.get_instance()
    cameras = buffer.get_cameras()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "cameras": cameras,
            "is_admin": False,
            "page_title": "Smart CCTV • Live Monitoring (View Only)",
            "brand_badge": "Monitoring (View Only)",
        },
    )


@app.get("/dashboard_admin", response_class=HTMLResponse)
async def dashboard_admin_page(request: Request) -> HTMLResponse:
    """Render the Full Access administrative dashboard with camera & zone controls."""
    _sync_disk_cameras_into_buffer()
    buffer = MultiCameraBuffer.get_instance()
    cameras = buffer.get_cameras()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "cameras": cameras,
            "is_admin": True,
            "page_title": "Smart CCTV • Admin Console",
            "brand_badge": "Admin Console",
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


@app.get("/video_feed/{camera_id}", response_class=StreamingResponse)
@app.get("/api/stream/{camera_id}", response_class=StreamingResponse)
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
        },
    )


@app.get("/api/status/{camera_id}")
async def get_camera_status(camera_id: str) -> JSONResponse:
    """Return latest telemetry and analytics status for the specified camera."""
    buffer = MultiCameraBuffer.get_instance()
    telemetry = buffer.get_telemetry(camera_id=camera_id)
    return JSONResponse(content=telemetry)


@app.get("/api/cameras")
async def list_cameras() -> JSONResponse:
    """Return list of all registered cameras."""
    _sync_disk_cameras_into_buffer()
    buffer = MultiCameraBuffer.get_instance()
    cameras = buffer.get_cameras()
    return JSONResponse(content=cameras)


@app.post("/api/cameras/test_rtsp")
async def test_rtsp_connection(request: Request) -> JSONResponse:
    """Test RTSP stream connectivity with strict timeout."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    rtsp_url = (data.get("rtsp_url") or "").strip()
    ip = (data.get("ip") or "").strip()
    port = data.get("port", 554)
    user = (data.get("user") or "").strip()
    password = (data.get("pass") or "").strip()
    channel = data.get("channel", 102)

    if not rtsp_url:
        if not ip:
            raise HTTPException(status_code=400, detail="IP address or full RTSP URL is required.")
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


@app.post("/api/cameras/add")
async def add_camera(request: Request) -> JSONResponse:
    """Dynamically register a new camera, configure .env, create directory, and hot-start pipeline."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    raw_id = (data.get("camera_id") or "").strip().lower()
    name = (data.get("name") or "").strip()
    ip = (data.get("ip") or "").strip()
    port = int(data.get("port") or 554)
    user = (data.get("user") or "").strip()
    password = (data.get("pass") or "").strip()
    channel = int(data.get("channel") or 102)
    custom_rtsp = (data.get("rtsp_url") or "").strip()

    if not raw_id or not re.match(r"^[a-z0-9_-]+$", raw_id):
        raise HTTPException(status_code=400, detail="Camera ID harus alfanumerik huruf kecil/garis bawah (contoh: cam_05).")
    if not name:
        name = raw_id.replace("_", " ").title()

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


@app.post("/api/cameras/start/{camera_id}")
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


@app.post("/api/cameras/delete")
async def delete_camera(request: Request) -> JSONResponse:
    """Deactivate camera, stop pipeline, unregister from buffer, and archive directory."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    camera_id = (data.get("camera_id") or "").strip().lower()
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


@app.get("/api/zones")
@app.get("/api/zones/{camera_id}")
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


@app.post("/api/zones/update")
async def update_zones(request: Request) -> JSONResponse:
    """Validate, backup, write, and atomically reload camera ROI zones & tripwires."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    camera_id = data.get("camera_id", "cam_01")
    zones = data.get("zones", {})
    lines = data.get("lines", {})
    zone_configs = data.get("zone_configs", {})
    base_res = data.get("base_resolution", [1920, 1080])

    if not isinstance(zones, dict):
        zones = {}
    if not isinstance(lines, dict):
        lines = {}

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


@app.post("/api/cameras/{camera_id}/calibration_mode", response_model=None)
@app.post("/api/calibration/mode", response_model=None)
async def toggle_calibration_mode(
    camera_id: Optional[str] = None,
    cam: Optional[str] = None,
    active: bool = Query(True),
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


@app.get("/api/events")
async def list_recent_events(
    camera_id: Optional[str] = None,
    cam: Optional[str] = None,
    limit: int = 50,
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


@app.get("/api/clips/{filename}")
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


@app.post("/api/buzzer/test")
async def trigger_buzzer_test() -> JSONResponse:
    """Trigger a manual buzzer test webhook and verify device response."""
    notifier = BuzzerNotifier.get_instance()
    result = notifier.test_connection()
    status_code = 200 if result.get("success") else 502 if result.get("status") == "UNREACHABLE" else 400
    return JSONResponse(status_code=status_code, content=result)


@app.get("/api/buzzer/status")
async def get_buzzer_status() -> JSONResponse:
    """Retrieve operational status, telemetry, and queue metrics for hardware buzzer."""
    notifier = BuzzerNotifier.get_instance()
    return JSONResponse(status_code=200, content=notifier.get_status())


class DashboardServer:
    """Daemon thread runner for Uvicorn / FastAPI server."""

    _server_thread: Optional[threading.Thread] = None
    _uvicorn_server: Optional[uvicorn.Server] = None

    @classmethod
    def start(cls, host: str = "127.0.0.1", port: int = 8000) -> threading.Thread:
        """Start Uvicorn in a daemon background thread."""
        if cls._server_thread is not None and cls._server_thread.is_alive():
            logger.info(f"[DashboardServer] Already running on http://{host}:{port}")
            return cls._server_thread

        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        cls._uvicorn_server = uvicorn.Server(config)

        def _run():
            logger.info(f"[DashboardServer] Web Dashboard started on http://{host}:{port}")
            cls._uvicorn_server.run()

        cls._server_thread = threading.Thread(target=_run, name="DashboardServerThread", daemon=True)
        cls._server_thread.start()
        return cls._server_thread

    @classmethod
    def stop(cls) -> None:
        """Gracefully stop the Uvicorn server."""
        if cls._uvicorn_server is not None:
            cls._uvicorn_server.should_exit = True
