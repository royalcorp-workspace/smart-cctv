"""Unit test suite for Vehicle Parking & Obstruction Dwell Logic on cam_02."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.tracker import CentroidTracker, TrackedObject
from notification.local_alert import VisualHUD


def test_vehicle_tracker_stationary_and_dwell():
    """Verify that moving vehicles do not accumulate dwell, while stationary vehicles do."""
    tracker = CentroidTracker(max_distance_px=100.0)

    # Frame 1: Vehicle moving fast (x=50, y=100) -> (x=80, y=100)
    t0 = 1000.0
    det1 = [((40, 80, 40, 40), (60, 100), "zone_1_koridor", 1600.0, "car", 20.0, 0.85, "yolo")]
    active, _ = tracker.update(det1, timestamp=t0)
    assert len(active) == 1
    car_trk = active[0]
    assert car_trk.class_label == "car"
    assert car_trk.is_stationary is False
    assert car_trk.dwell_duration == 0.0

    # Frame 2: Moving (+25px shift)
    t1 = t0 + 0.2
    det2 = [((65, 80, 40, 40), (85, 100), "zone_1_koridor", 1600.0, "car", 20.0, 0.85, "yolo")]
    active, _ = tracker.update(det2, timestamp=t1)
    car_trk = active[0]
    assert car_trk.is_stationary is False
    assert car_trk.dwell_duration == 0.0

    # Frame 3-7: Stopped at same spot (shift <= 2px)
    cur_t = t1
    for step in range(6):
        cur_t += 0.5
        det_stop = [((66, 80, 40, 40), (86, 100), "zone_1_koridor", 1600.0, "car", 20.0, 0.85, "yolo")]
        active, _ = tracker.update(det_stop, timestamp=cur_t)

    car_trk = active[0]
    assert car_trk.is_stationary is True, "Vehicle should be confirmed stationary after quiet frames"
    assert car_trk.dwell_duration > 0.0, f"Stationary vehicle must accumulate dwell duration, got {car_trk.dwell_duration}"

    # Move vehicle away (+60px shift)
    cur_t += 0.5
    det_move = [((125, 80, 40, 40), (145, 100), "zone_1_koridor", 1600.0, "car", 20.0, 0.85, "yolo")]
    active, _ = tracker.update(det_move, timestamp=cur_t)
    car_trk = active[0]
    assert car_trk.is_stationary is False, "Moving vehicle must reset stationary flag"
    assert car_trk.dwell_duration == 0.0, "Moving vehicle must reset dwell duration to 0"

    print("  [PASS] test_vehicle_tracker_stationary_and_dwell")


def test_visual_hud_vehicle_badges():
    """Verify VisualHUD badge text formatting for zebra cross and parking zones."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    zones = {
        "zone_1_koridor": [[10, 10], [200, 10], [200, 200], [10, 200]],
        "zone_2_transit": [[250, 10], [500, 10], [500, 200], [250, 200]],
    }

    # 1. Moving car
    moving_car = TrackedObject(
        track_id=1,
        centroid=(100, 100),
        anchor_centroid=(100, 100),
        bbox=(80, 80, 40, 40),
        zone_id="zone_1_koridor",
        contour_area=1600.0,
        first_seen=100.0,
        last_seen=105.0,
        stationary_start=105.0,
        dwell_duration=0.0,
        is_stationary=False,
        class_label="car",
    )

    # 2. Zebra Cross stationary vehicle (dwell=35s < 60s)
    zebra_warning = TrackedObject(
        track_id=2,
        centroid=(120, 120),
        anchor_centroid=(120, 120),
        bbox=(100, 100, 40, 40),
        zone_id="zone_1_koridor",
        contour_area=1600.0,
        first_seen=100.0,
        last_seen=135.0,
        stationary_start=100.0,
        dwell_duration=35.0,
        dwell_threshold=60.0,
        is_stationary=True,
        class_label="truck",
    )

    # 3. Zebra Cross breach vehicle (dwell=75s >= 60s)
    zebra_alert = TrackedObject(
        track_id=3,
        centroid=(140, 140),
        anchor_centroid=(140, 140),
        bbox=(120, 120, 40, 40),
        zone_id="zone_1_koridor",
        contour_area=1600.0,
        first_seen=100.0,
        last_seen=175.0,
        stationary_start=100.0,
        dwell_duration=75.0,
        dwell_threshold=60.0,
        is_stationary=True,
        is_triggered=True,
        class_label="car",
    )

    # 4. Parking Zone breach vehicle (dwell=1950s >= 1800s)
    parking_alert = TrackedObject(
        track_id=4,
        centroid=(350, 100),
        anchor_centroid=(350, 100),
        bbox=(330, 80, 40, 40),
        zone_id="zone_2_transit",
        contour_area=1600.0,
        first_seen=100.0,
        last_seen=2050.0,
        stationary_start=100.0,
        dwell_duration=1950.0,
        dwell_threshold=1800.0,
        is_stationary=True,
        is_triggered=True,
        class_label="bus",
    )

    rendered = VisualHUD.render(
        canvas=canvas,
        zones=zones,
        tracked_objects=[moving_car, zebra_warning, zebra_alert, parking_alert],
        camera_id="cam_02",
        fps=12.0,
        is_connected=True,
        faces=[],
        zone_base_resolution=(640, 480),
        camera_name="Area Parkir POS-2",
    )
    assert rendered is not None
    assert rendered.shape == (480, 640, 3)
    print("  [PASS] test_visual_hud_vehicle_badges")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING CAM_02 VEHICLE PARKING & DWELL LOGIC TEST SUITE")
    print("=" * 60)
    test_vehicle_tracker_stationary_and_dwell()
    test_visual_hud_vehicle_badges()
    print("=" * 60)
    print("ALL CAM_02 VEHICLE PARKING TESTS PASSED! [OK]\n")
