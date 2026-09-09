"""FastAPI Web Server for Smart CCTV Multi-Camera Streaming and Telemetry."""

import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from web.buffer import MultiCameraBuffer

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


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request) -> HTMLResponse:
    """Render the single-page dark mode surveillance dashboard."""
    buffer = MultiCameraBuffer.get_instance()
    cameras = buffer.get_cameras()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"cameras": cameras},
    )


@app.get("/video_feed/{camera_id}")
def video_feed(camera_id: str) -> StreamingResponse:
    """Stream live MJPEG feed for the requested camera."""
    buffer = MultiCameraBuffer.get_instance()
    return StreamingResponse(
        buffer.stream_generator(camera_id=camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
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
    buffer = MultiCameraBuffer.get_instance()
    cameras = buffer.get_cameras()
    return JSONResponse(content=cameras)


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
async def get_zones(camera_id: Optional[str] = None) -> JSONResponse:
    """Get current ROI zones configuration for a camera."""
    cam_id = camera_id or "cam_01"
    cam_dir = Path(__file__).resolve().parent.parent / "cameras" / cam_id
    roi_file = cam_dir / "roi_zones.json"

    if not roi_file.exists():
        raise HTTPException(status_code=404, detail=f"ROI configuration for '{cam_id}' not found.")

    try:
        import json
        with open(roi_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        base_res = data.get("base_resolution", [1920, 1080])
        zones = {k: v for k, v in data.items() if k != "base_resolution" and not k.startswith("_")}
        return JSONResponse(content={
            "status": "success",
            "camera_id": cam_id,
            "base_resolution": base_res,
            "zones": zones,
        })
    except Exception as e:
        logger.error(f"[WebServer] Failed to read zones for {cam_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/zones/update")
async def update_zones(request: Request) -> JSONResponse:
    """Validate, backup, write, and atomically reload camera ROI zones."""
    import json
    import shutil

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.")

    camera_id = data.get("camera_id", "cam_01")
    zones = data.get("zones", {})
    base_res = data.get("base_resolution", [1920, 1080])

    if not isinstance(zones, dict) or not zones:
        raise HTTPException(status_code=400, detail="Zones object must contain at least one zone.")

    base_w, base_h = int(base_res[0]), int(base_res[1])

    # Validate each polygon
    for zone_name, pts in zones.items():
        if not isinstance(pts, list) or len(pts) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Zone '{zone_name}' must have at least 3 vertices (found {len(pts) if isinstance(pts, list) else 0}).",
            )
        for pt in pts:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                raise HTTPException(status_code=400, detail=f"Invalid point format in zone '{zone_name}'.")
            x, y = float(pt[0]), float(pt[1])
            if not (-100 <= x <= base_w + 100 and -100 <= y <= base_h + 100):
                raise HTTPException(
                    status_code=400,
                    detail=f"Point ({int(x)}, {int(y)}) in zone '{zone_name}' exceeds coordinate bounds (0..{base_w}, 0..{base_h}).",
                )
        if is_polygon_self_intersecting([[float(p[0]), float(p[1])] for p in pts]):
            raise HTTPException(
                status_code=400,
                detail=f"Zone '{zone_name}' has self-intersecting edges. Please untangle polygon vertices.",
            )

    cam_dir = Path(__file__).resolve().parent.parent / "cameras" / camera_id
    if not cam_dir.exists():
        raise HTTPException(status_code=404, detail=f"Camera directory '{camera_id}' not found.")

    roi_file = cam_dir / "roi_zones.json"
    bak_file = cam_dir / "roi_zones.json.bak"

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
    }
    for k, v in zones.items():
        save_payload[k] = [[int(round(p[0])), int(round(p[1]))] for p in v]

    # Save to disk
    with open(roi_file, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, indent=2)

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
        "message": f"Zones for {camera_id} successfully saved and reloaded.",
        "reloaded": reloaded,
        "camera_id": camera_id,
        "zones": save_payload,
    })


@app.get("/api/events")
async def list_recent_events(camera_id: Optional[str] = None, limit: int = 50) -> JSONResponse:
    """Retrieve recent incident events with clip paths and telemetry info."""
    from storage.db import get_recent_events
    try:
        events = get_recent_events(limit=limit, camera_id=camera_id)
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
