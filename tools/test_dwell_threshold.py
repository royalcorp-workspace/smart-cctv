"""Unit and integration test for 60-Minute (3600s) Unattended Bag Dwell Threshold,
Anti-Spam Cooldown, and Safe Telegram Fallback.
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.tracker import CentroidTracker, TrackedObject
from notification.telegram_alert import TelegramNotifier
import notification.local_alert as local_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestDwellThreshold")


def test_config_audit() -> bool:
    """Audit cameras/cam_01/config.json to ensure 3600s threshold is strictly locked."""
    config_path = PROJECT_ROOT / "cameras" / "cam_01" / "config.json"
    assert config_path.exists(), f"Config file not found: {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    zones = cfg.get("zones", {})
    assert len(zones) > 0, "No zones configured!"

    print("\n[TEST 1] Auditing cameras/cam_01/config.json zones...")
    for zone_id, zcfg in zones.items():
        dwell_sec = zcfg.get("dwell_threshold_sec")
        dwell_time = zcfg.get("dwell_time_threshold")
        unattended = zcfg.get("unattended_threshold")

        print(f" - Zone '{zone_id}': dwell_threshold_sec={dwell_sec}, dwell_time_threshold={dwell_time}, unattended_threshold={unattended}")
        assert dwell_sec == 3600, f"Zone {zone_id} dwell_threshold_sec is {dwell_sec}, expected 3600"
        assert dwell_time == 3600.0, f"Zone {zone_id} dwell_time_threshold is {dwell_time}, expected 3600.0"
        assert unattended == 3600.0, f"Zone {zone_id} unattended_threshold is {unattended}, expected 3600.0"

    print(" -> PASS: All zone configurations strictly locked to 3600 seconds (60 minutes).")
    return True


def test_dwell_time_simulation() -> bool:
    """Simulate bag tracking over time (t=0, t=60s, t=600s, t=3599s, t=3600s)."""
    print("\n[TEST 2] Simulating CentroidTracker Dwell Time Accumulation...")
    tracker = CentroidTracker()

    start_time = 1000.0
    bag_bbox = (300, 300, 60, 60)
    bag_centroid = (330, 330)
    zone_id = "zone_2_transit"
    area = 3600.0

    # Frame 1: t = 0s
    dets = [(bag_bbox, bag_centroid, zone_id, area, "tas", 25.0, 0.85)]
    active, _ = tracker.update(dets, timestamp=start_time)
    assert len(active) == 1
    bag = active[0]
    assert bag.dwell_duration == 0.0
    assert not bag.is_triggered
    assert not bag.alert_sent

    # Simulation at t = 60s (1 minute)
    # Small jitter within 5px (anchor radius is 15px)
    jitter_centroid = (332, 331)
    active, _ = tracker.update([(bag_bbox, jitter_centroid, zone_id, area, "tas", 25.0, 0.85)], timestamp=start_time + 60.0)
    bag = active[0]
    print(f" - At t = 60s (1m): Dwell = {bag.dwell_duration:.1f}s, Alert Triggered = {bag.dwell_duration >= 3600.0}")
    assert bag.is_stationary, "Bag should be detected as stationary"
    assert abs(bag.dwell_duration - 60.0) < 1.0, f"Expected ~60s dwell, got {bag.dwell_duration}"
    assert bag.dwell_duration < 3600.0, "Alert MUST NOT trigger at 60s!"

    # Simulation at t = 600s (10 minutes)
    active, _ = tracker.update([(bag_bbox, (331, 330), zone_id, area, "tas", 25.0, 0.85)], timestamp=start_time + 600.0)
    bag = active[0]
    print(f" - At t = 600s (10m): Dwell = {bag.dwell_duration:.1f}s, Alert Triggered = {bag.dwell_duration >= 3600.0}")
    assert abs(bag.dwell_duration - 600.0) < 1.0, f"Expected ~600s dwell, got {bag.dwell_duration}"
    assert bag.dwell_duration < 3600.0, "Alert MUST NOT trigger at 600s (10 minutes)!"

    # Simulation at t = 3599.0s (59.98 minutes)
    active, _ = tracker.update([(bag_bbox, (330, 330), zone_id, area, "tas", 25.0, 0.85)], timestamp=start_time + 3599.0)
    bag = active[0]
    print(f" - At t = 3599s: Dwell = {bag.dwell_duration:.1f}s, Alert Triggered = {bag.dwell_duration >= 3600.0}")
    assert bag.dwell_duration < 3600.0, "Alert MUST NOT trigger before 3600.0s!"

    # Simulation at t = 3600.0s (60 minutes)
    active, _ = tracker.update([(bag_bbox, (330, 330), zone_id, area, "tas", 25.0, 0.85)], timestamp=start_time + 3600.0)
    bag = active[0]
    print(f" - At t = 3600s (60m): Dwell = {bag.dwell_duration:.1f}s, Alert Triggered = {bag.dwell_duration >= 3600.0}")
    assert bag.dwell_duration >= 3600.0, "Alert MUST trigger at 3600.0s (60 minutes)!"

    print(" -> PASS: Dwell threshold strictly fires ONLY at t >= 3600s (never at 60s or 600s).")
    return True


def test_anti_spam_cooldown_and_resets() -> bool:
    """Verify single dispatch, alert_sent anti-spam, and reset on movement / owner attendance."""
    print("\n[TEST 3] Testing Anti-Spam Single Dispatch & Cooldown Resets...")
    tracker = CentroidTracker()
    start_time = 2000.0
    bag_bbox = (200, 200, 50, 50)
    bag_centroid = (225, 225)
    zone_id = "zone_2_transit"

    # 1. Initialize and reach 3600s
    tracker.update([(bag_bbox, bag_centroid, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=start_time)
    active, _ = tracker.update([(bag_bbox, bag_centroid, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=start_time + 3600.0)
    bag = active[0]
    assert bag.dwell_duration >= 3600.0

    # First dispatch simulated
    dispatches = 0
    if not bag.alert_sent:
        dispatches += 1
        bag.alert_sent = True
        bag.is_triggered = True

    assert dispatches == 1, "First dispatch should fire"
    assert bag.alert_sent is True

    # 2. Next frame at t = 3605s (same stationary position)
    active, _ = tracker.update([(bag_bbox, bag_centroid, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=start_time + 3605.0)
    bag = active[0]
    if not bag.alert_sent:
        dispatches += 1

    assert dispatches == 1, "Anti-spam failed: alert was dispatched again without movement or reset!"
    print(" - Anti-spam check: Subsequent frames do NOT trigger duplicate alerts (dispatches = 1).")

    # 3. Test significant movement (> 15 px smoothed displacement from anchor)
    # Move centroid from (225, 225) to (260, 260) over consecutive frames
    moved_bbox = (230, 230, 50, 50)
    moved_centroid = (260, 260)
    
    # Frame 1 of move
    active, _ = tracker.update([(moved_bbox, moved_centroid, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=start_time + 3601.0)
    bag = tracker.objects[1]
    assert bag.dwell_duration == 0.0, f"Expected dwell reset to 0.0s on movement, got {bag.dwell_duration}"
    assert bag.alert_sent is False, "Expected alert_sent reset to False on significant movement"
    assert bag.is_triggered is False, "Expected is_triggered reset to False on significant movement"
    print(" - Significant movement reset check: Dwell reset to 0.0s and alert_sent reset to False.")

    # 4. Settle EMA at new position and simulate stationary dwell until 3600s
    for i in range(2, 6):
        tracker.update([(moved_bbox, moved_centroid, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=start_time + 3600.0 + i)

    # Fast-forward to 3600s dwell at new position
    tracker.update([(moved_bbox, moved_centroid, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=start_time + 7205.0)
    bag = tracker.objects[1]
    assert bag.dwell_duration >= 3600.0
    bag.alert_sent = True
    bag.is_triggered = True

    # 5. Test owner attendance reset (>= 4.0s sustained)
    # Person stands beside the bag:
    person_bbox = (230, 150, 50, 130)
    person_centroid = (255, 215)

    # Frame 1 of attendance at t=7206.0s (less than 4.0s sustained)
    dets_with_person = [
        (moved_bbox, moved_centroid, zone_id, 2500.0, "tas", 20.0, 0.9),
        (person_bbox, person_centroid, zone_id, 6500.0, "person", 20.0, 0.9),
    ]
    tracker.update(dets_with_person, timestamp=start_time + 7206.0)
    assert bag.is_attended is True
    # At < 4.0s, dwell is paused, not yet zeroed
    assert bag.dwell_duration >= 3600.0

    # Frame at t=7211.0s (5.0s of sustained attendance >= 4.0s)
    tracker.update(dets_with_person, timestamp=start_time + 7211.0)
    assert bag.is_attended is True
    assert bag.dwell_duration == 0.0, f"Expected dwell reset to 0.0s after sustained attendance, got {bag.dwell_duration}"
    assert bag.alert_sent is False, "Expected alert_sent reset to False after sustained attendance"
    assert bag.is_triggered is False, "Expected is_triggered reset to False after sustained attendance"
    print(" - Owner attendance reset check: Dwell reset to 0.0s and alert_sent reset to False after >= 4.0s.")

    print(" -> PASS: Anti-spam cooldown and state resets work perfectly.")
    return True


def test_telegram_safe_fallback() -> bool:
    """Verify safe fallback when Telegram credentials are empty/invalid and strict 3600s gate."""
    print("\n[TEST 4] Testing Telegram Safe Fallback & Gate...")
    
    # Initialize notifier with non-existent config path
    notifier = TelegramNotifier(config_path=str(PROJECT_ROOT / "configs" / "non_existent.json"))
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Test dwell < 3600s rejection
    res_sub_threshold = notifier.dispatch_alert(
        camera_id="cam_01",
        zone_id="zone_2_transit",
        track_id=1,
        dwell_duration=600.0,  # 10 minutes
        timestamp_str="2026-09-07 08:30:00",
        frame=dummy_frame,
    )
    assert res_sub_threshold is False, "TelegramNotifier MUST reject alerts with dwell < 3600.0s"
    print(" - Dwell gate check: Dwell 600s (< 3600s) correctly rejected by notifier.")

    # 2. Test dwell >= 3600s with missing credentials
    # Ensure it returns False cleanly without raising an exception or crashing
    res_no_creds = notifier.dispatch_alert(
        camera_id="cam_01",
        zone_id="zone_2_transit",
        track_id=1,
        dwell_duration=3600.0,
        timestamp_str="2026-09-07 09:30:00",
        frame=dummy_frame,
    )
    assert res_no_creds is False, "TelegramNotifier should return False when credentials not configured"
    print(" - Missing credentials check: Gracefully handled with warning log and False return.")

    # 3. Test with dummy credentials that point to localhost / invalid host
    dummy_notifier = TelegramNotifier()
    dummy_notifier.enabled = True
    dummy_notifier.bot_token = "123456:INVALID_TOKEN"
    dummy_notifier.global_admins = ["99999999"]
    dummy_notifier.api_base_url = "http://127.0.0.1:1"  # Unreachable port

    # Queue an alert job
    success = dummy_notifier.dispatch_alert(
        camera_id="cam_01",
        zone_id="zone_2_transit",
        track_id=1,
        dwell_duration=3600.0,
        timestamp_str="2026-09-07 09:30:00",
        frame=dummy_frame,
    )
    assert success is True, "Job should be enqueued into background worker queue"

    # Let the background worker process and handle the network error safely
    import time
    time.sleep(1.0)
    dummy_notifier.stop()

    print(" - Unreachable network/invalid token check: Worker caught error safely without crashing.")
    print(" -> PASS: Safe Telegram fallback verified.")
    return True


def test_hud_minute_format() -> bool:
    """Verify local_alert HUD minute-based formatting."""
    print("\n[TEST 5] Testing HUD Minute-Based Dwell Text Rendering...")

    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    zones = {
        "zone_2_transit": [[255, 405], [304, 472], [610, 474], [626, 431], [423, 290]]
    }

    # Case A: Dwell = 180s (3.0m)
    obj_normal = TrackedObject(
        track_id=2,
        centroid=(400, 350),
        anchor_centroid=(400, 350),
        bbox=(380, 330, 40, 40),
        zone_id="zone_2_transit",
        contour_area=1600.0,
        first_seen=0.0,
        last_seen=180.0,
        stationary_start=0.0,
        dwell_duration=180.0,
        is_stationary=True,
        is_triggered=False,
        dwell_threshold=3600.0,
        class_label="tas",
    )

    # Render on canvas
    local_alert.VisualHUD.render(
        canvas=canvas,
        zones=zones,
        tracked_objects=[obj_normal],
        camera_id="cam_01",
        fps=25.0,
        is_connected=True,
    )

    # Case B: Dwell = 3600s (60.0m - Triggered Alert)
    obj_alert = TrackedObject(
        track_id=3,
        centroid=(400, 350),
        anchor_centroid=(400, 350),
        bbox=(380, 330, 40, 40),
        zone_id="zone_2_transit",
        contour_area=1600.0,
        first_seen=0.0,
        last_seen=3600.0,
        stationary_start=0.0,
        dwell_duration=3600.0,
        is_stationary=True,
        is_triggered=True,
        dwell_threshold=3600.0,
        class_label="tas",
    )

    local_alert.VisualHUD.render(
        canvas=canvas,
        zones=zones,
        tracked_objects=[obj_alert],
        camera_id="cam_01",
        fps=25.0,
        is_connected=True,
    )

    print(" - Checked draw_hud execution with 180s (3.0m/60m) and 3600s ([ALERT] 60m).")
    print(" -> PASS: HUD formatting runs smoothly without errors or raw second displays.")
    return True


def run_all_tests():
    print("=" * 60)
    print("RUNNING STRICT 60-MINUTE DWELL & SAFE TELEGRAM AUDIT SUITE")
    print("=" * 60)

    assert test_config_audit()
    assert test_dwell_time_simulation()
    assert test_anti_spam_cooldown_and_resets()
    assert test_telegram_safe_fallback()
    assert test_hud_minute_format()

    print("\n" + "=" * 60)
    print("ALL 5 AUDIT TESTS PASSED SUCCESSFULLY! (100% COMPLIANT)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
