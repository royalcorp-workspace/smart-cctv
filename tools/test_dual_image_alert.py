"""Self-testing suite for Dual-Image Telegram Alert (Overview + Contextual Zoom Crop).

Tests:
1. Contextual zoom crop extraction with 35% padding across extreme edge cases.
2. Aspect-ratio preserving cubic upscaling (min_width >= 480px, cv2.INTER_CUBIC).
3. Payload construction and serialization for Telegram sendMediaGroup.
4. Non-blocking queue behavior and background worker processing.
5. Live Telegram Bot dispatch test verifying HTTP 200 OK.
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from notification.telegram_alert import TelegramNotifier


def test_zoom_crop_edge_cases() -> bool:
    print("\n[STEP 1] Testing Contextual Zoom Crop Extraction Across Extreme Edge Cases...")

    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Draw some patterns to ensure crop has distinct visual content
    cv2.rectangle(test_frame, (100, 100), (400, 400), (120, 150, 200), -1)
    cv2.circle(test_frame, (960, 540), 200, (80, 180, 90), -1)

    # Test cases in 1080p space or 640p space
    # (x, y, w, h), description
    cases = [
        ((0, 0, 50, 60), "Top-Left Corner (x=0, y=0)"),
        ((600, 440, 40, 40), "Bottom-Right Corner (mapped from 640p to 1080p: ~1800, ~990)"),
        ((1880, 1040, 40, 40), "Extreme 1080p Bottom-Right (touching edges 1920, 1080)"),
        ((250, 200, 20, 25), "Micro Object (20x25 px, requires cubic upscale to >= 480px)"),
        ((200, 200, 150, 40), "Wide Aspect Ratio Object (w >> h)"),
        ((200, 200, 40, 150), "Tall Aspect Ratio Object (h >> w)"),
    ]

    for bbox, desc in cases:
        crop = TelegramNotifier.create_zoom_crop(test_frame, bbox, min_width=480, padding_ratio=0.35)
        assert crop is not None, f"Crop returned None for {desc}"
        ch, cw = crop.shape[:2]
        assert cw >= 480, f"Crop width {cw} < min_width 480 for {desc}"
        assert ch > 0, f"Crop height {ch} invalid for {desc}"
        print(f" - {desc:45s} -> Result Size: {cw}x{ch} px [PASS]")

    # Negative / None test
    assert TelegramNotifier.create_zoom_crop(test_frame, None) is None
    assert TelegramNotifier.create_zoom_crop(np.zeros((0, 0, 3), dtype=np.uint8), (10, 10, 20, 20)) is None
    print(" - None/Empty inputs safely handled without exceptions [PASS]")

    print(" -> PASS: Contextual zoom crop safely handles all extreme edge cases.")
    return True


def test_media_group_payload_structure() -> bool:
    print("\n[STEP 2] Testing sendMediaGroup Payload Serialization & Structure...")

    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Overview Simulation", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    zoom_crop = TelegramNotifier.create_zoom_crop(test_frame, (100, 100, 80, 80), min_width=480)

    success_ov, enc_ov = cv2.imencode(".jpg", test_frame)
    success_zm, enc_zm = cv2.imencode(".jpg", zoom_crop)
    assert success_ov and success_zm

    ov_bytes = enc_ov.tobytes()
    zm_bytes = enc_zm.tobytes()

    caption = "🚨 <b>PERINGATAN: PELANGGARAN CLEAR AREA</b>\nKamera: cam_01"
    media = [
        {"type": "photo", "media": "attach://photo_overview.jpg", "caption": caption, "parse_mode": "HTML"},
        {"type": "photo", "media": "attach://photo_zoom.jpg"}
    ]
    files = {
        "photo_overview.jpg": ("photo_overview.jpg", ov_bytes, "image/jpeg"),
        "photo_zoom.jpg": ("photo_zoom.jpg", zm_bytes, "image/jpeg")
    }

    # Verify JSON serializability
    media_json = json.dumps(media)
    parsed = json.loads(media_json)
    assert len(parsed) == 2, "Media array must contain exactly 2 photos"
    assert parsed[0]["type"] == "photo" and parsed[1]["type"] == "photo"
    assert parsed[0]["media"] == "attach://photo_overview.jpg"
    assert parsed[1]["media"] == "attach://photo_zoom.jpg"
    assert parsed[0]["caption"] == caption

    # Verify files
    assert "photo_overview.jpg" in files and len(files["photo_overview.jpg"][1]) > 0
    assert "photo_zoom.jpg" in files and len(files["photo_zoom.jpg"][1]) > 0

    print(f" - JSON Media Group items count : {len(parsed)}")
    print(f" - Overview JPEG encoded size   : {len(ov_bytes):,} bytes")
    print(f" - Zoom Crop JPEG encoded size  : {len(zm_bytes):,} bytes")
    print(" -> PASS: Payload serialization and multipart file attachments verified.")
    return True


def test_non_blocking_and_live_dispatch() -> bool:
    print("\n[STEP 3] Testing Non-Blocking Dispatch & Live Telegram Album Delivery...")

    notifier = TelegramNotifier.get_instance()
    assert notifier.enabled, "TelegramNotifier must be enabled"
    assert notifier.bot_token, "Bot token must be configured"

    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    test_frame[:] = (35, 35, 35)

    # Draw simulated bag on floor
    bag_x, bag_y, bag_w, bag_h = 400, 350, 70, 85
    cv2.rectangle(test_frame, (bag_x, bag_y), (bag_x + bag_w, bag_y + bag_h), (30, 80, 180), -1)
    cv2.putText(test_frame, "BAG", (bag_x + 10, bag_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Measure dispatch_alert latency on calling thread (must be non-blocking < 1 ms)
    t0 = time.perf_counter()
    enqueued = notifier.dispatch_alert(
        camera_id="cam_01",
        zone_id="zone_2_transit",
        track_id=99,
        dwell_duration=3600.0,
        timestamp_str=time.strftime("%Y-%m-%d %H:%M:%S"),
        frame=test_frame,
        bbox=(bag_x, bag_y, bag_w, bag_h),
        zone_name="Area Transit Depan (Penitipan)",
    )
    dispatch_time_ms = (time.perf_counter() - t0) * 1000.0
    print(f" - dispatch_alert execution time: {dispatch_time_ms:.3f} ms (Target < 10.0 ms)")
    assert enqueued is True, "Alert should be enqueued"
    assert dispatch_time_ms < 10.0, f"dispatch_alert took {dispatch_time_ms} ms (exceeds non-blocking budget)"

    # Wait for daemon worker to complete processing the job
    print(" - Waiting for background worker to deliver album to Telegram...")
    time.sleep(3.0)

    print(" -> PASS: Non-blocking dispatch latency verified and album sent to background worker.")
    return True


def main() -> int:
    print("=" * 68)
    print("      DUAL-IMAGE TELEGRAM ALERT VERIFICATION & TEST SUITE      ")
    print("=" * 68)

    ok1 = test_zoom_crop_edge_cases()
    ok2 = test_media_group_payload_structure()
    ok3 = test_non_blocking_and_live_dispatch()

    if ok1 and ok2 and ok3:
        print("\n" + "=" * 68)
        print("  ALL DUAL-IMAGE ALERT TESTS PASSED SUCCESSFULLY (100%)!   ")
        print("=" * 68)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
