"""Unit and Integration Tests for FastAPI Web Dashboard and MultiCameraBuffer."""

import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.buffer import MultiCameraBuffer
from web.server import app, video_feed
from starlette.testclient import TestClient


def test_buffer_operations():
    print("[1] Testing MultiCameraBuffer core functionality...")
    buf = MultiCameraBuffer.get_instance()
    buf.register_camera("cam_01", "Koridor Utama")
    buf.register_camera("cam_02", "Area Transit")

    cameras = buf.get_cameras()
    assert len(cameras) >= 2, f"Expected at least 2 cameras, got {len(cameras)}"
    print(f" - Registered cameras: {[c['id'] for c in cameras]}")

    # Create dummy 1080p frame
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dummy_frame[:] = (50, 100, 150)

    telemetry = {
        "fps": 11.2,
        "online": True,
        "rtsp_status": "Connected",
        "violations": 1,
        "active_tracks": 2,
        "identified_faces": [{"name": "Rian", "confidence": 0.88, "zone": "zone_1_koridor"}],
    }

    buf.update_frame("cam_01", dummy_frame, telemetry=telemetry)
    latest_jpeg = buf.get_latest_frame("cam_01")
    assert latest_jpeg is not None, "Failed to get latest JPEG frame"
    assert len(latest_jpeg) > 1000, f"JPEG bytes too small: {len(latest_jpeg)}"
    print(f" - Frame encoded to JPEG: {len(latest_jpeg)} bytes")

    telem = buf.get_telemetry("cam_01")
    assert telem["fps"] == 11.2
    assert telem["violations"] == 1
    assert len(telem["identified_faces"]) == 1
    print(f" - Telemetry verified: FPS={telem['fps']}, Violations={telem['violations']}")
    print(" - [OK] MultiCameraBuffer tests passed!")


def test_fastapi_endpoints():
    print("\n[2] Testing FastAPI endpoints with TestClient...")
    client = TestClient(app)

    # Test GET / (Redirect to /dashboard)
    resp_root = client.get("/", follow_redirects=False)
    assert resp_root.status_code == 307, f"GET / returned {resp_root.status_code}"
    assert resp_root.headers.get("location") == "/dashboard"
    print(" - [OK] GET / successfully redirects (307) to /dashboard")

    # Test GET /dashboard (View-Only)
    resp_view = client.get("/dashboard")
    assert resp_view.status_code == 200, f"GET /dashboard returned {resp_view.status_code}"
    assert "SMART CCTV 2.0" in resp_view.text
    assert "Monitoring (View Only)" in resp_view.text
    assert "cameraSelect" in resp_view.text
    assert "streamFeed" in resp_view.text
    assert "Riwayat Insiden & Putar Ulang DVR" not in resp_view.text
    assert "window.IS_ADMIN = false;" in resp_view.text
    # Grid View components
    assert "dashboardMain" in resp_view.text
    assert "gridContainer" in resp_view.text
    assert "cameraGridMatrix" in resp_view.text
    assert "btnModeGrid" in resp_view.text
    assert "btnModeSingle" in resp_view.text
    assert "gridSizeSelect" in resp_view.text
    assert "grid_manager.js" in resp_view.text
    assert "toggleFullscreen()" in resp_view.text
    # Edit controls must NOT be present
    assert "btnAddCamera" not in resp_view.text
    assert "btnDeleteCamera" not in resp_view.text
    assert "btnToggleEditor" not in resp_view.text
    assert "zoneEditorCanvas" not in resp_view.text
    assert "zoneToolbar" not in resp_view.text
    assert "modal_add_camera" not in resp_view.text
    assert "camera_manager.js" not in resp_view.text
    assert "calibration_canvas.js" not in resp_view.text
    print(" - [OK] GET /dashboard rendered View-Only mode with Grid View matrix & controls")

    # Test GET /dashboard_admin (Full Access)
    resp_admin = client.get("/dashboard_admin")
    assert resp_admin.status_code == 200, f"GET /dashboard_admin returned {resp_admin.status_code}"
    assert "SMART CCTV 2.0" in resp_admin.text
    assert "Admin Console" in resp_admin.text
    assert "window.IS_ADMIN = true;" in resp_admin.text
    assert "gridContainer" in resp_admin.text
    # Edit controls MUST be present
    assert "btnAddCamera" in resp_admin.text
    assert "btnDeleteCamera" in resp_admin.text
    assert "btnToggleEditor" in resp_admin.text
    assert "zoneEditorCanvas" in resp_admin.text
    assert "zoneToolbar" in resp_admin.text
    assert "camera_manager.js" in resp_admin.text
    assert "calibration_canvas.js" in resp_admin.text
    print(" - [OK] GET /dashboard_admin rendered Admin mode with full edit controls")

    # Test GET /static/css/dashboard.css
    resp_css = client.get("/static/css/dashboard.css")
    assert resp_css.status_code == 200, f"GET /static/css/dashboard.css returned {resp_css.status_code}"
    assert "@import" in resp_css.text and "variables.css" in resp_css.text
    assert "grid.css" in resp_css.text
    print(" - [OK] GET /static/css/dashboard.css returned HTTP 200 OK with grid.css import")

    # Test GET /static/css/modules/grid.css
    resp_grid_css = client.get("/static/css/modules/grid.css")
    assert resp_grid_css.status_code == 200
    assert ".grid-matrix-2x2" in resp_grid_css.text
    assert ".grid-matrix-6x6" in resp_grid_css.text
    print(" - [OK] GET /static/css/modules/grid.css returned valid 2x2..6x6 matrix classes")

    # Test GET /static/js/modules/grid_manager.js
    resp_grid_js = client.get("/static/js/modules/grid_manager.js")
    assert resp_grid_js.status_code == 200
    assert "toggleCameraFullscreen" in resp_grid_js.text
    assert "switchViewMode" in resp_grid_js.text
    print(" - [OK] GET /static/js/modules/grid_manager.js returned valid grid orchestrator")

    # Test GET /static/js/dashboard.js
    resp_js = client.get("/static/js/dashboard.js")
    assert resp_js.status_code == 200, f"GET /static/js/dashboard.js returned {resp_js.status_code}"
    assert "onCameraChange" in resp_js.text
    print(" - [OK] GET /static/js/dashboard.js returned HTTP 200 OK")

    # Test GET /api/cameras
    resp_cams = client.get("/api/cameras")
    assert resp_cams.status_code == 200
    cam_data = resp_cams.json()
    assert isinstance(cam_data, list) and len(cam_data) >= 1
    print(f" - [OK] GET /api/cameras returned: {[c['id'] for c in cam_data]}")

    # Test GET /api/status/cam_01
    resp_stat = client.get("/api/status/cam_01")
    assert resp_stat.status_code == 200
    stat_data = resp_stat.json()
    assert stat_data["camera_id"] == "cam_01"
    assert "fps" in stat_data
    assert "violations" in stat_data
    print(f" - [OK] GET /api/status/cam_01 returned valid telemetry payload")

    # Test video_feed streaming generator
    import asyncio
    resp_feed = video_feed("cam_01")
    assert "multipart/x-mixed-replace" in resp_feed.media_type
    first_chunk = asyncio.run(anext(resp_feed.body_iterator))
    assert b"--frame" in first_chunk
    assert b"Content-Type: image/jpeg" in first_chunk
    print(f" - [OK] video_feed('cam_01') yields valid multipart/x-mixed-replace chunk ({len(first_chunk)} bytes)")


def main():
    print("=" * 65)
    print("      SMART CCTV 2.0 - WEB DASHBOARD VERIFICATION SUITE       ")
    print("=" * 65)
    test_buffer_operations()
    test_fastapi_endpoints()
    print("\n" + "=" * 65)
    print("      ALL WEB DASHBOARD SUITES PASSED SUCCESSFULLY!          ")
    print("=" * 65)


if __name__ == "__main__":
    main()
