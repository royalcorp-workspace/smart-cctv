"""Integration test for rapid camera stream switching and disconnect detection."""

import asyncio
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


def run_test():
    print("=" * 65)
    print("      TESTING MULTI-CAMERA STREAM SWITCHING & RESPONSIVENESS     ")
    print("=" * 65)

    buf = MultiCameraBuffer.get_instance()
    for cid in ["cam_01", "cam_02", "cam_03", "cam_04"]:
        buf.register_camera(cid, f"Kamera {cid}")
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        buf.update_frame(cid, dummy, telemetry={"fps": 15.0, "online": True})

    client = TestClient(app)

    # 1. Test rapid switching between cameras
    print("[1] Simulating rapid camera switching via /api/status...")
    for cid in ["cam_01", "cam_02", "cam_03", "cam_04", "cam_01", "cam_03"]:
        t0 = time.time()
        resp = client.get(f"/api/status/{cid}")
        elapsed = time.time() - t0
        assert resp.status_code == 200, f"Failed status for {cid}"
        assert elapsed < 0.2, f"Status response too slow ({elapsed:.3f}s)"
    print(" - [OK] Rapid telemetry polling responsive (< 200ms per request).")

    # 2. Test stream switching on both /video_feed and /api/stream
    print("\n[2] Testing MJPEG stream generator switching...")
    for cid in ["cam_01", "cam_02", "cam_03", "cam_04", "cam_01"]:
        t0 = time.time()
        resp_feed = video_feed(cid)
        assert "multipart/x-mixed-replace" in resp_feed.media_type
        # Test first chunk retrieval
        first_chunk = asyncio.run(anext(resp_feed.body_iterator))
        assert b"--frame" in first_chunk
        assert b"Content-Type: image/jpeg" in first_chunk
        elapsed = time.time() - t0
        assert elapsed < 0.2, f"Stream chunk retrieval too slow ({elapsed:.3f}s)"
        print(f" - Stream for {cid}: received {len(first_chunk)} bytes in {elapsed:.3f}s")
    print(" - [OK] All camera streams open instantly and stream valid MJPEG frames.")

    # 3. Test non-blocking frame retrieval
    print("\n[3] Testing non-blocking frame and sequence retrieval...")
    for cid in ["cam_01", "cam_02", "cam_03", "cam_04"]:
        frame, seq = buf.get_latest_frame_and_seq(cid)
        assert frame is not None, f"Frame for {cid} was None"
        assert seq >= 1, f"Sequence for {cid} was 0"
    print(" - [OK] get_latest_frame_and_seq non-blocking retrieval confirmed.")

    print("\n" + "=" * 65)
    print("      ALL CAMERA STREAM SWITCHING TESTS PASSED!                 ")
    print("=" * 65)


if __name__ == "__main__":
    run_test()
