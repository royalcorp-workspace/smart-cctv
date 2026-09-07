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
