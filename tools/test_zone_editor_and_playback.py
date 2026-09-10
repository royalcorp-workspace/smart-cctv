"""Comprehensive test suite for Interactive Web Zone Editor and DVR Ring Buffer Playback."""

import json
import sys
from pathlib import Path
import shutil
import tempfile
import time
import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from engine.zone_filter import ZoneFilter
from main import RingBufferRecorder
from web.server import app, is_polygon_self_intersecting
from web.buffer import MultiCameraBuffer


def test_coordinate_conversion_precision():
    """Test CSS to 1080p native matrix scale mapping accuracy."""
    # Simulation of canvas bounding rect: 960x540 viewport offset by (50, 100)
    rect_left = 50.0
    rect_top = 100.0
    rect_width = 960.0
    rect_height = 540.0

    def css_to_native(client_x: float, client_y: float):
        scale_x = 1920.0 / rect_width
        scale_y = 1080.0 / rect_height
        nx = round((client_x - rect_left) * scale_x)
        ny = round((client_y - rect_top) * scale_y)
        return max(0, min(1920, nx)), max(0, min(1080, ny))

    # Top-Left corner
    assert css_to_native(50.0, 100.0) == (0, 0)
    # Bottom-Right corner
    assert css_to_native(1010.0, 640.0) == (1920, 1080)
    # Center
    assert css_to_native(530.0, 370.0) == (960, 540)
    # Quarter point
    assert css_to_native(290.0, 235.0) == (480, 270)


def test_geometry_validation():
    """Test client and server-side polygon validation (min vertices, self-intersection)."""
    # 1. Valid Rectangle
    valid_box = [[100, 100], [500, 100], [500, 400], [100, 400]]
    assert not is_polygon_self_intersecting(valid_box)

    # 2. Valid Triangle
    valid_triangle = [[200, 200], [400, 200], [300, 400]]
    assert not is_polygon_self_intersecting(valid_triangle)

    # 3. Bow-Tie (Self-intersecting hourglass)
    # (100,100) -> (500,400) crosses (500,100) -> (100,400)
    bowtie = [[100, 100], [500, 400], [500, 100], [100, 400]]
    assert is_polygon_self_intersecting(bowtie)

    # 4. Self-intersecting complex pentagon
    complex_crossing = [[100, 100], [300, 100], [150, 400], [300, 300], [50, 200]]
    assert is_polygon_self_intersecting(complex_crossing)


def test_ring_buffer_ram_optimization_and_h264_export():
    """Test in-memory JPEG compression and browser-compatible H.264 MP4 export."""
    recorder = RingBufferRecorder(maxlen=30, target_size=(1280, 720), fps=15.0)

    # Generate synthetic 1080p frames with distinct visual content
    for i in range(20):
        frame = np.full((1080, 1920, 3), fill_value=(i * 10, 120, 200), dtype=np.uint8)
        cv2.putText(
            frame,
            f"Frame {i:03d}",
            (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            3.0,
            (255, 255, 255),
            5,
        )
        recorder.push_frame(frame)

    assert len(recorder) == 20
    # Verify JPEG compression RAM savings: each item in deque is bytes, not raw numpy array
    for item in recorder._buffer:
        assert isinstance(item, bytes)
        assert len(item) < 150_000  # JPEG 720p is ~20-60 KB, vs ~2.7 MB raw

    # Export clip
    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_path = Path(tmp_dir) / "test_incident.mp4"
        success = recorder.export_clip(clip_path)
        assert success is True
        assert clip_path.exists()
        assert clip_path.stat().st_size > 5_000  # Non-trivial size

        # Verify readability with cv2.VideoCapture
        cap = cv2.VideoCapture(str(clip_path))
        assert cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert w == 1280
        assert h == 720

        frame_count = 0
        while True:
            ret, f = cap.read()
            if not ret or f is None:
                break
            frame_count += 1
        cap.release()

        assert frame_count == 20


def test_api_zones_and_atomic_reload():
    """Test GET /api/zones and POST /api/zones/update with validation and backup."""
    client = TestClient(app)

    # 1. GET current zones
    resp = client.get("/api/zones/cam_01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "zones" in data
    assert "base_resolution" in data

    # 2. POST invalid: points < 3
    bad_payload_short = {
        "camera_id": "cam_01",
        "zones": {
            "zone_test": [[100, 100], [200, 200]]
        }
    }
    resp_bad = client.post("/api/zones/update", json=bad_payload_short)
    assert resp_bad.status_code == 400
    assert "at least 3 vertices" in resp_bad.json()["detail"]

    # 3. POST invalid: self-intersecting bowtie
    bad_payload_bowtie = {
        "camera_id": "cam_01",
        "zones": {
            "zone_test": [[100, 100], [500, 400], [500, 100], [100, 400]]
        }
    }
    resp_bowtie = client.post("/api/zones/update", json=bad_payload_bowtie)
    assert resp_bowtie.status_code == 400
    assert "self-intersecting" in resp_bowtie.json()["detail"]

    # 4. POST valid update & check backup creation
    cam_dir = Path("cameras/cam_01")
    roi_file = cam_dir / "roi_zones.json"
    bak_file = cam_dir / "roi_zones.json.bak"

    original_content = roi_file.read_text(encoding="utf-8")

    try:
        valid_update = {
            "camera_id": "cam_01",
            "base_resolution": [1920, 1080],
            "zones": {
                "zone_1_koridor": [[150, 600], [500, 500], [100, 150], [20, 150]],
                "zone_2_transit": [[300, 1000], [1200, 650], [1900, 950], [1900, 1050]],
            }
        }
        resp_ok = client.post("/api/zones/update", json=valid_update)
        assert resp_ok.status_code == 200
        assert resp_ok.json()["status"] == "success"
        assert bak_file.exists()

        # Verify updated content written
        with open(roi_file, "r", encoding="utf-8") as f:
            updated_data = json.load(f)
        assert updated_data["zone_1_koridor"][0] == [150, 600]

    finally:
        # Restore original roi_zones.json to avoid side effects
        with open(roi_file, "w", encoding="utf-8") as f:
            f.write(original_content)


def test_api_clips_http_206_partial_content():
    """Test GET /api/clips/{filename} with Byte-Range partial content streaming."""
    clips_dir = Path("storage/clips")
    clips_dir.mkdir(parents=True, exist_ok=True)
    test_clip = clips_dir / "test_partial_stream.mp4"

    # Write dummy mp4 data (5000 bytes)
    dummy_bytes = b"0" * 5000
    test_clip.write_bytes(dummy_bytes)

    client = TestClient(app)

    try:
        # Full content (no Range header)
        resp_full = client.get(f"/api/clips/{test_clip.name}")
        assert resp_full.status_code == 200
        assert resp_full.headers["Accept-Ranges"] == "bytes"
        assert int(resp_full.headers["Content-Length"]) == 5000

        # Partial Content: Range: bytes=0-499 (first 500 bytes)
        resp_range = client.get(f"/api/clips/{test_clip.name}", headers={"Range": "bytes=0-499"})
        assert resp_range.status_code == 206
        assert resp_range.headers["Content-Range"] == "bytes 0-499/5000"
        assert len(resp_range.content) == 500

        # Partial Content: Range: bytes=1000-1999 (1000 bytes)
        resp_range2 = client.get(f"/api/clips/{test_clip.name}", headers={"Range": "bytes=1000-1999"})
        assert resp_range2.status_code == 206
        assert resp_range2.headers["Content-Range"] == "bytes 1000-1999/5000"
        assert len(resp_range2.content) == 1000

        # Non-existent clip -> 404
        resp_404 = client.get("/api/clips/non_existent_clip_123.mp4")
        assert resp_404.status_code == 404

    finally:
        if test_clip.exists():
            test_clip.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
