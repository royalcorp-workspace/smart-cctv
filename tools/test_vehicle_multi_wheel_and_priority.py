"""Unit test suite verifying:
1. Multi-point wheel probe detection for diagonal large trucks.
2. Zone priority resolution (60s Zebra Cross > 1800s Parking > outside_zone).
3. Stationary anti-flapping lock preserving dwell timer against boundary jitter.
4. Genuine vehicle movement resetting dwell timer upon zone change.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from engine.zone_filter import ZoneFilter
from engine.tracker import CentroidTracker, TrackedObject


def test_multi_point_wheel_probe_and_priority():
    """Verify multi-point wheel probes and priority resolution (60s Zebra Cross > 1800s Parking)."""
    # Define cam_02-like zones
    # zone_1 (Zebra cross at bottom-left: x in [0, 150], y in [240, 360])
    # zone_2 (Parking adjacent: x in [120, 260], y in [180, 280])
    zones = {
        "zone_1_koridor": [[0, 240], [150, 240], [150, 360], [0, 360]],
        "zone_2_transit": [[120, 180], [260, 180], [260, 280], [120, 280]],
    }
    zone_cfgs = {
        "zone_1_koridor": {"name": "Zebra Cross", "dwell_threshold_sec": 60.0},
        "zone_2_transit": {"name": "Parking Transit", "dwell_threshold_sec": 1800.0},
    }

    zf = ZoneFilter(
        zones=zones,
        zone_configs=zone_cfgs,
        frame_shape=(360, 640),
        base_resolution=(640, 360),
    )

    # 1. Large diagonal truck:
    # bbox = (x=50, y=100, w=200, h=180) -> bottom = 280
    # Center wheel is at x = 50 + 100 = 150, y = 275 (on the boundary or outside zone_1)
    # Left wheel is at x = 50 + 50 = 100, y = 275 -> INSIDE zone_1_koridor!
    # Right wheel is at x = 50 + 150 = 200, y = 275 -> INSIDE zone_2_transit!
    wheel_y = 275
    left_wheel = (100, wheel_y)
    center_wheel = (150, wheel_y)
    right_wheel = (200, wheel_y)

    # Multi-point probe check
    best_zone, best_dist = zf.find_best_zone_for_points([left_wheel, center_wheel, right_wheel], margin_px=8.0)
    assert best_zone == "zone_1_koridor", (
        f"Zone priority failed! 60s Zebra Cross should beat 1800s Parking, got: {best_zone}"
    )
    print("PASS: Multi-point wheel contact detects diagonal truck and awards highest priority to Zebra Cross.")

    # 2. Truck touching ONLY parking zone
    p_left = (180, 230)
    p_center = (210, 230)
    p_right = (240, 230)
    p_zone, _ = zf.find_best_zone_for_points([p_left, p_center, p_right], margin_px=8.0)
    assert p_zone == "zone_2_transit", f"Expected zone_2_transit, got {p_zone}"

    # 3. Vehicle completely outside all zones
    out_pts = [(500, 300), (520, 300), (540, 300)]
    out_zone, out_dist = zf.find_best_zone_for_points(out_pts, margin_px=8.0)
    assert out_zone is None, f"Expected None for outside points, got {out_zone}"
    print("PASS: Single-zone and outside-zone multi-point evaluations verified.")


def test_stationary_zone_lock_anti_flapping():
    """Verify stationary anti-flapping lock preserves dwell timer against 1-2px boundary jitter."""
    zone_cfgs = {
        "zone_1_koridor": {"name": "Zebra Cross", "dwell_threshold_sec": 60.0},
        "zone_2_transit": {"name": "Parking Transit", "dwell_threshold_sec": 1800.0},
    }

    tracker = CentroidTracker(
        max_distance_px=100.0,
        anchor_radius_px=40.0,
        flicker_tolerance_sec=2.0,
        max_disappeared_sec=12.0,
        zone_configs=zone_cfgs,
    )

    t0 = 1000.0
    # Truck registered in zone_1_koridor (Zebra Cross)
    det1 = ((50, 100, 200, 180), (150, 190), "zone_1_koridor", 36000.0, "truck", 25.0, 0.9, "yolo")
    active, _ = tracker.update([det1], timestamp=t0)
    assert len(active) == 1
    trk = active[0]
    assert trk.zone_id == "zone_1_koridor"

    # Feed 10 stationary frames in zone_1_koridor:
    # frames 1-4 latch stationary at t=1004, frames 5-10 accumulate dwell
    cur_t = t0
    for _ in range(10):
        cur_t += 1.0
        active, _ = tracker.update([det1], timestamp=cur_t)
        trk = active[0]

    assert trk.is_stationary is True, "Truck must be stationary."
    assert trk.dwell_duration >= 5.0, f"Dwell duration must accumulate, got {trk.dwell_duration}"
    accumulated_dwell = trk.dwell_duration

    # Frame 7: Detection boundary jitter!
    # Centroid shifts by only 1 pixel (151, 190), but raw detection says "outside_zone"!
    cur_t += 1.0
    det_jitter = ((51, 100, 200, 180), (151, 190), "outside_zone", 36000.0, "truck", -1.0, 0.9, "yolo")
    active_jit, _ = tracker.update([det_jitter], timestamp=cur_t)
    trk = active_jit[0]

    # Verification: Stationary Zone Lock MUST protect zone_1_koridor!
    assert trk.zone_id == "zone_1_koridor", (
        f"Anti-flapping failed! Stationary truck was demoted to {trk.zone_id} by 1px jitter."
    )
    assert trk.dwell_duration >= accumulated_dwell + 1.0, (
        f"Dwell timer was reset by jitter! Expected >= {accumulated_dwell + 1.0}, got {trk.dwell_duration}"
    )
    assert trk.is_stationary is True, "Truck must remain stationary during boundary jitter."

    # Frame 8: Another jitter to weaker zone (zone_2_transit 1800s)
    cur_t += 1.0
    det_weaker = ((50, 101, 200, 180), (150, 191), "zone_2_transit", 36000.0, "truck", 2.0, 0.9, "yolo")
    active_weak, _ = tracker.update([det_weaker], timestamp=cur_t)
    trk = active_weak[0]

    assert trk.zone_id == "zone_1_koridor", (
        f"Anti-flapping failed! Truck in Zebra Cross was demoted to weaker zone: {trk.zone_id}"
    )
    assert trk.dwell_duration >= accumulated_dwell + 2.0, (
        f"Dwell timer reset by weaker zone jitter! Got {trk.dwell_duration}"
    )
    print("PASS: Stationary Anti-Flapping Zone Lock verified. Jitter does NOT reset dwell timer.")


def test_moving_vehicle_zone_change_resets_dwell():
    """Verify that when a vehicle genuinely moves (>15px shift), zone change properly resets dwell."""
    zone_cfgs = {
        "zone_1_koridor": {"name": "Zebra Cross", "dwell_threshold_sec": 60.0},
        "zone_2_transit": {"name": "Parking Transit", "dwell_threshold_sec": 1800.0},
    }

    tracker = CentroidTracker(
        max_distance_px=100.0,
        anchor_radius_px=40.0,
        flicker_tolerance_sec=2.0,
        max_disappeared_sec=12.0,
        zone_configs=zone_cfgs,
    )

    t0 = 2000.0
    det_a = ((50, 100, 100, 80), (100, 140), "zone_2_transit", 8000.0, "truck", 20.0, 0.9, "yolo")
    active, _ = tracker.update([det_a], timestamp=t0)
    trk = active[0]

    # Accumulate stationary dwell in zone_2
    cur_t = t0
    for _ in range(8):
        cur_t += 2.0
        active, _ = tracker.update([det_a], timestamp=cur_t)
        trk = active[0]

    assert trk.is_stationary is True
    assert trk.dwell_duration > 0.0, f"Dwell must accumulate, got {trk.dwell_duration}"

    # Truck genuinely moves: shifts 40px (> 15px) into outside_zone
    cur_t += 2.0
    det_moved = ((100, 140, 100, 80), (150, 180), "outside_zone", 8000.0, "truck", -10.0, 0.9, "yolo")
    active_mv, _ = tracker.update([det_moved], timestamp=cur_t)
    trk = active_mv[0]

    assert trk.is_stationary is False, "Truck must not be stationary after moving 56px."
    assert trk.dwell_duration == 0.0, f"Dwell must reset to 0.0 when vehicle moves, got {trk.dwell_duration}"
    assert trk.zone_id == "outside_zone"
    print("PASS: Genuine vehicle movement properly updates zone and resets dwell timer.")


if __name__ == "__main__":
    test_multi_point_wheel_probe_and_priority()
    test_stationary_zone_lock_anti_flapping()
    test_moving_vehicle_zone_change_resets_dwell()
    print("\nALL VEHICLE MULTI-WHEEL & PRIORITY TESTS PASSED SUCCESSFULLY!")
