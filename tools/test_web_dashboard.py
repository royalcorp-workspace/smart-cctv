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

    # Test GET /
    resp = client.get("/")
    assert resp.status_code == 200, f"GET / returned {resp.status_code}"
    assert "SMART CCTV 2.0" in resp.text
    assert "cameraSelect" in resp.text
    assert "streamFeed" in resp.text
    assert "/static/css/dashboard.css" in resp.text
    assert "/static/js/dashboard.js" in resp.text
    print(" - [OK] GET / rendered dashboard HTML with static asset links")

    # Test GET /static/css/dashboard.css
    resp_css = client.get("/static/css/dashboard.css")
    assert resp_css.status_code == 200, f"GET /static/css/dashboard.css returned {resp_css.status_code}"
    assert "--bg-base" in resp_css.text
    print(" - [OK] GET /static/css/dashboard.css returned HTTP 200 OK")

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
