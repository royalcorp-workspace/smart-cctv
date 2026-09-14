"""Unit test suite for cam_03 Logistic Road Vehicle Parking & Obstruction Dwell Logic."""

import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.tracker import CentroidTracker, TrackedObject
from notification.local_alert import VisualHUD


def test_cam03_config_integrity():
    """Verify cam_03 configuration parameters."""
    cfg_path = Path(__file__).resolve().parent.parent / "cameras" / "cam_03" / "config.json"
    assert cfg_path.exists(), "cam_03 config.json must exist"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 1. Walkway compliance must be disabled
    assert cfg.get("walkway_compliance", {}).get("enabled") is False

    # 2. Face recognition disabled
    assert cfg.get("face_detector", {}).get("enabled") is False
    assert cfg.get("face_recognizer", {}).get("enabled") is False

    # 3. Vehicle classes enabled in detector
    enabled_classes = cfg.get("detector", {}).get("enabled_classes", [])
    for vc in ("car", "bus", "truck"):
        assert vc in enabled_classes

    # 4. Zone 1 rule configured for 20 minutes (1200s)
    z1 = cfg.get("zones", {}).get("zone_1_koridor", {})
    assert z1.get("dwell_threshold_sec") == 1200.0
    assert z1.get("alert_label") == "PARKIR MELEBIHI BATAS (20M)"
    for vc in ("car", "bus", "truck"):
        assert vc in z1.get("target_classes", [])

    print("  [PASS] test_cam03_config_integrity")


def test_cam03_stationary_and_pedestrian_isolation():
    """Verify that moving vehicles and pedestrians do not trigger dwell, while stationary vehicles do."""
    tracker = CentroidTracker(max_distance_px=100.0)

    # 1. Pedestrian in zone_1_koridor
    t0 = 100.0
    ped_det = [((50, 50, 30, 80), (65, 90), "zone_1_koridor", 2400.0, "person", 20.0, 0.90, "yolo")]
    active, _ = tracker.update(ped_det, timestamp=t0)
    assert len(active) == 1
    ped = active[0]
    assert ped.class_label == "person"
    assert ped.is_stationary is False
    assert ped.dwell_duration == 0.0

    # Simulate pedestrian stationary for 30s -> person must NEVER accumulate dwell
    for step in range(10):
        t0 += 3.0
        active, _ = tracker.update(ped_det, timestamp=t0)
        assert active[0].dwell_duration == 0.0
        assert active[0].is_stationary is False

    # 2. Vehicle stops inside zone_1_koridor
    veh_det = [
        ((50, 50, 30, 80), (65, 90), "zone_1_koridor", 2400.0, "person", 20.0, 0.90, "yolo"),
        ((150, 100, 60, 50), (180, 125), "zone_1_koridor", 3000.0, "truck", 25.0, 0.88, "yolo"),
    ]
    t0 += 1.0
    active, _ = tracker.update(veh_det, timestamp=t0)
    truck_trk = [o for o in active if o.class_label == "truck"][0]

    # Vehicle stopped for 6 consecutive frames
    for _ in range(6):
        t0 += 1.0
        active, _ = tracker.update(veh_det, timestamp=t0)

    truck_trk = [o for o in active if o.class_label == "truck"][0]
    assert truck_trk.is_stationary is True
    assert truck_trk.dwell_duration > 0.0

    print("  [PASS] test_cam03_stationary_and_pedestrian_isolation")


def test_cam03_visual_hud_badges():
    """Verify VisualHUD badge formatting for cam_03."""
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    zones = {
        "zone_1_koridor": [[10, 10], [500, 10], [500, 300], [10, 300]],
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

    # 2. Stationary truck (< 20m, e.g. 500s = 8.3m)
    parked_truck = TrackedObject(
        track_id=2,
        centroid=(200, 120),
        anchor_centroid=(200, 120),
        bbox=(170, 100, 60, 40),
        zone_id="zone_1_koridor",
        contour_area=2400.0,
        first_seen=100.0,
        last_seen=600.0,
        stationary_start=100.0,
        dwell_duration=500.0,
        dwell_threshold=1200.0,
        is_stationary=True,
        class_label="truck",
    )

    # 3. Violation vehicle (>= 20m / 1200s, e.g. 1320s = 22m)
    breach_truck = TrackedObject(
        track_id=3,
        centroid=(350, 150),
        anchor_centroid=(350, 150),
        bbox=(320, 130, 60, 40),
        zone_id="zone_1_koridor",
        contour_area=2400.0,
        first_seen=100.0,
        last_seen=1420.0,
        stationary_start=100.0,
        dwell_duration=1320.0,
        dwell_threshold=1200.0,
        is_stationary=True,
        is_triggered=True,
        class_label="truck",
    )

    # 4. Pedestrian on cam_03 (should be rendered in Emerald Green)
    ped = TrackedObject(
        track_id=4,
        centroid=(50, 150),
        anchor_centroid=(50, 150),
        bbox=(40, 120, 20, 60),
        zone_id="zone_1_koridor",
        contour_area=1200.0,
        first_seen=100.0,
        last_seen=105.0,
        stationary_start=105.0,
        dwell_duration=0.0,
        is_stationary=False,
        class_label="person",
    )

    rendered = VisualHUD.render(
        canvas=canvas,
        zones=zones,
        tracked_objects=[moving_car, parked_truck, breach_truck, ped],
        camera_id="cam_03",
        fps=12.0,
        is_connected=True,
        faces=[],
        zone_base_resolution=(640, 360),
        camera_name="Jalur Logistik Arah POS-1",
    )
    assert rendered is not None
    assert rendered.shape == (360, 640, 3)

    print("  [PASS] test_cam03_visual_hud_badges")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING CAM_03 LOGISTIC ROAD VEHICLE DWELL TEST SUITE")
    print("=" * 60)
    test_cam03_config_integrity()
    test_cam03_stationary_and_pedestrian_isolation()
    test_cam03_visual_hud_badges()
    print("=" * 60)
    print("ALL CAM_03 VEHICLE DWELL TESTS PASSED! [OK]\n")
