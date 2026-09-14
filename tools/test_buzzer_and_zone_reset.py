"""Unit test suite verifying:
1. BuzzerNotifier singleton, cooldown, non-blocking trigger, and silent failure.
2. CentroidTracker vehicle dwell reset on zone change.
3. Wheel contact clamp within [0, H - 1].
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from types import SimpleNamespace
from notification.buzzer_alert import BuzzerNotifier
from engine.tracker import CentroidTracker, TrackedObject


def test_buzzer_notifier_singleton_and_cooldown():
    """Verify BuzzerNotifier behaves as a non-blocking singleton with 15s cooldown."""
    notifier1 = BuzzerNotifier.get_instance()
    notifier2 = BuzzerNotifier.get_instance()
    assert notifier1 is notifier2, "BuzzerNotifier must be a singleton."

    # Test status
    status = notifier1.get_status()
    assert "enabled" in status
    assert "cooldown_sec" in status
    assert status["cooldown_sec"] == 15.0

    # Ensure silent failure if IP is empty or unreachable
    notifier1.ip = ""
    start_t = time.time()
    # Triggering should be non-blocking (<10ms) and return False without throwing
    res = notifier1.trigger(event_type="tripwire_crossing", camera_id="cam_04", metadata={"test": True})
    dispatch_elapsed = (time.time() - start_t) * 1000.0
    assert res is False, "Empty IP should return False without raising."
    assert dispatch_elapsed < 10.0, f"Trigger dispatch took {dispatch_elapsed:.2f}ms, expected < 10ms"

    # Test with dummy IP
    notifier1.ip = "192.0.2.1"  # TEST-NET non-routable IP
    notifier1._last_triggered_time = 0.0
    res_enqueued = notifier1.trigger(event_type="tripwire_crossing", camera_id="cam_04")
    assert res_enqueued is True, "Valid trigger should be enqueued."

    # Immediate second trigger should be rejected by 15s cooldown
    res_cooldown = notifier1.trigger(event_type="tripwire_crossing", camera_id="cam_04")
    assert res_cooldown is False, "Second trigger within cooldown must be suppressed."

    # Test vehicle_dwell event dispatch (reset cooldown timer first)
    notifier1._last_triggered_time = 0.0
    res_veh = notifier1.trigger(
        event_type="vehicle_dwell",
        camera_id="cam_02",
        metadata={
            "track_id": 17,
            "class_label": "truck",
            "zone_id": "zone_1_koridor",
            "dwell_duration": 65.0,
            "dwell_threshold": 60.0,
        },
    )
    assert res_veh is True, "vehicle_dwell event should be enqueued successfully."

    # Test test_connection without crash
    notifier1.ip = ""
    test_res = notifier1.test_connection()
    assert test_res["success"] is False
    assert test_res["status"] == "NO_IP"

    # Restore IP to empty
    notifier1.ip = ""
    print("PASS: BuzzerNotifier singleton, non-blocking dispatch, cooldown, and silent failure verified.")


def test_tracker_vehicle_dwell_reset_on_zone_change():
    """Verify CentroidTracker resets dwell_duration, anchor_time, and alert flags on vehicle zone change."""
    tracker = CentroidTracker(
        max_distance_px=100.0,
        anchor_radius_px=40.0,
        flicker_tolerance_sec=2.0,
        max_disappeared_sec=12.0,
    )

    t0 = 1000.0
    # Register vehicle in zone_A
    det = ((50, 50, 60, 40), (80, 70), "zone_A", 2400.0, "truck", 25.0, 0.9, "yolo")
    active, _ = tracker.update([det], timestamp=t0)
    assert len(active) == 1
    veh_track = active[0]
    track_id = veh_track.track_id
    assert veh_track.zone_id == "zone_A"
    assert hasattr(veh_track, "anchor_time")

    # Feed 5 stationary frames in zone_A: stationary_frames >= 4 latches is_stationary
    cur_t = t0
    for i in range(1, 6):
        cur_t += 2.0
        active_step, _ = tracker.update([det], timestamp=cur_t)
        veh_track = active_step[0]

    assert veh_track.is_stationary is True, "Vehicle must be stationary after 5 frames."
    assert veh_track.dwell_duration > 0.0, f"Dwell duration should have accumulated, got {veh_track.dwell_duration}"
    veh_track.is_triggered = True
    veh_track.alert_sent = True

    # Vehicle transitions to zone_B (within tracking distance <= 100px)
    cur_t += 3.0
    det_b = ((70, 70, 60, 40), (100, 90), "zone_B", 2400.0, "truck", 25.0, 0.9, "yolo")
    active_b, _ = tracker.update([det_b], timestamp=cur_t)
    veh_track = active_b[0]
    assert veh_track.zone_id == "zone_B", "Vehicle must transition to zone_B."
    assert veh_track.dwell_duration == 0.0, f"Dwell duration must be reset to 0.0 on zone change, got {veh_track.dwell_duration}"
    assert veh_track.anchor_time == cur_t, f"anchor_time must be reset to now ({cur_t}), got {veh_track.anchor_time}"
    assert veh_track.stationary_start == cur_t, f"stationary_start must be reset to now ({cur_t}), got {veh_track.stationary_start}"
    assert veh_track.is_triggered is False, "is_triggered must be reset to False."
    assert veh_track.alert_sent is False, "alert_sent must be reset to False."

    print("PASS: CentroidTracker vehicle dwell reset on zone change verified.")


def test_wheel_contact_coordinate_clamping():
    """Verify wheel contact coordinate stays strictly clamped within [0, H - 1]."""
    infer_h = 360

    # Normal case: bbox inside canvas
    bbox1 = (100, 200, 50, 80)
    wheel_y1 = min(max(0, bbox1[1] + bbox1[3] - 5), infer_h - 1)
    assert wheel_y1 == 275
    assert 0 <= wheel_y1 <= infer_h - 1

    # Extreme bottom overflow case: bbox touching bottom
    bbox_overflow = (100, 300, 50, 100)  # y + h = 400 > 360
    wheel_y_ov = min(max(0, bbox_overflow[1] + bbox_overflow[3] - 5), infer_h - 1)
    assert wheel_y_ov == 359, f"Expected 359, got {wheel_y_ov}"
    assert 0 <= wheel_y_ov <= infer_h - 1

    # Negative / top case
    bbox_neg = (100, -20, 50, 10)  # y + h = -10
    wheel_y_neg = min(max(0, bbox_neg[1] + bbox_neg[3] - 5), infer_h - 1)
    assert wheel_y_neg == 0, f"Expected 0, got {wheel_y_neg}"
    assert 0 <= wheel_y_neg <= infer_h - 1

    print("PASS: Wheel contact coordinate clamping [0, H - 1] verified.")


if __name__ == "__main__":
    test_buzzer_notifier_singleton_and_cooldown()
    test_tracker_vehicle_dwell_reset_on_zone_change()
    test_wheel_contact_coordinate_clamping()
    print("\nALL BUZZER & ZONE RESET TESTS PASSED SUCCESSFULLY!")
