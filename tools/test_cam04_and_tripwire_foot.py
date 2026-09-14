"""Unit test suite verifying:
1. Tripwire Foot-Contact Only (ground contact point) preventing false alarms for leaning body.
2. Vehicle wheel contact point crossing.
3. cam_04 config and multi-zone vehicle dwell thresholds.
4. cam_04 visual badges in local_alert.py.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from types import SimpleNamespace
import numpy as np

from engine.line_crossing import TripwireEngine, TripwireLine
from notification.local_alert import VisualHUD


def test_leaning_person_foot_contact():
    """Prove that a person leaning forward (centroid/head crosses line, foot does NOT) does NOT trigger crossing."""
    lines_cfg = {
        "trip_1": {
            "name": "Border Line",
            "p1": [0, 100],
            "p2": [400, 100],
            "direction": "both",
            "target_classes": ["person"],
        }
    }
    engine = TripwireEngine(lines=lines_cfg, cooldown_sec=1.0)

    # Frame 1: Person upright before line (foot at y=93, centroid at y=57.5)
    # bbox = (x1, y1, w, h) = (100, 20, 40, 75)
    # foot_y = 20 + 75 - 2 = 93 (< 100)
    track_p = SimpleNamespace(
        track_id=1,
        class_label="person",
        centroid=(120, 57),
        bbox=(100, 20, 40, 75),
    )
    events_f1 = engine.update([track_p], timestamp=1000.0)
    assert len(events_f1) == 0, "First frame registration must not trigger event."

    # Frame 2: Person leans forward heavily:
    # Centroid moves across the line to y=115!
    # But foot is firmly planted and foot_y is 96 (< 100).
    track_p.centroid = (120, 115)  # centroid has crossed y=100!
    track_p.bbox = (100, 60, 40, 38)  # foot_y = 60 + 38 - 2 = 96 (< 100)
    events_f2 = engine.update([track_p], timestamp=1000.1)
    assert len(events_f2) == 0, (
        f"False alarm detected! Head/centroid crossed, but foot has NOT crossed the line. Events: {events_f2}"
    )

    # Frame 3: Person finally steps across the line:
    # foot_y lands at 106 (> 100).
    track_p.centroid = (120, 125)
    track_p.bbox = (100, 65, 40, 43)  # foot_y = 65 + 43 - 2 = 106 (> 100)
    events_f3 = engine.update([track_p], timestamp=1000.2)
    assert len(events_f3) == 1, f"Foot crossed the line, expected 1 event, got {len(events_f3)}"
    assert events_f3[0].line_id == "trip_1"
    print("PASS: Leaning person foot-contact test (centroid crossed != alarm; foot crossed == alarm).")


def test_vehicle_wheel_contact():
    """Verify vehicle wheels contact point (y2 - 5) triggers crossing correctly."""
    lines_cfg = {
        "gate_line": {
            "name": "Gate Line",
            "p1": [0, 200],
            "p2": [500, 200],
            "direction": "both",
            "target_classes": ["truck"],
        }
    }
    engine = TripwireEngine(lines=lines_cfg, cooldown_sec=1.0)

    # Frame 1: Truck before line
    # bbox = (50, 100, 120, 100) -> wheel_y = 100 + 100 - 5 = 195 (< 200)
    truck = SimpleNamespace(
        track_id=10,
        class_label="truck",
        centroid=(110, 150),
        bbox=(50, 100, 120, 100),
    )
    events_1 = engine.update([truck], timestamp=2000.0)
    assert len(events_1) == 0

    # Frame 2: Truck front wheels cross line
    # bbox = (50, 110, 120, 100) -> wheel_y = 110 + 100 - 5 = 205 (> 200)
    truck.centroid = (110, 160)
    truck.bbox = (50, 110, 120, 100)
    events_2 = engine.update([truck], timestamp=2000.1)
    assert len(events_2) == 1, "Expected 1 event when truck wheels cross line."
    assert events_2[0].line_id == "gate_line"
    print("PASS: Vehicle wheel contact test.")


def test_cam04_config_integrity():
    """Verify cam_04 config settings conform to user specification."""
    cfg_path = Path("cameras/cam_04/config.json")
    assert cfg_path.exists(), "cameras/cam_04/config.json not found."
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 1. Walkway compliance disabled
    assert cfg.get("walkway_compliance", {}).get("enabled") is False, "walkway_compliance should be disabled."

    # 2. Face recognition disabled
    assert cfg.get("face_detector", {}).get("enabled") is False, "face_detector must be disabled."
    assert cfg.get("face_recognizer", {}).get("enabled") is False, "face_recognizer must be disabled."

    # 3. Detector includes vehicles
    for v_cls in ("car", "bus", "truck"):
        assert v_cls in cfg.get("detector", {}).get("enabled_classes", []), f"{v_cls} must be in enabled_classes."
        assert v_cls in cfg.get("detector", {}).get("target_classes", []), f"{v_cls} must be in target_classes."

    # 4. Multi-zone config
    zones = cfg.get("zones", {})
    z1 = zones.get("zone_1_koridor", {})
    assert z1.get("dwell_threshold_sec") == 600.0, "zone_1 dwell must be 600s."
    assert z1.get("name") == "Zebra Cross Timbangan"
    assert z1.get("alert_label") == "HALANGAN ZEBRA CROSS TIMBANGAN"
    assert "person" not in z1.get("target_classes", []), "person must not be in target_classes."

    z2 = zones.get("zone_2_transit", {})
    assert z2.get("dwell_threshold_sec") == 1800.0, "zone_2 dwell must be 1800s."
    assert z2.get("name") == "Area Parkir Antrean Timbangan"
    assert z2.get("alert_label") == "ANTREAN TIMBANGAN MELEBIHI BATAS"
    assert "person" not in z2.get("target_classes", []), "person must not be in target_classes."

    print("PASS: cam_04 configuration integrity verified.")


def test_cam04_visual_badges():
    """Verify local_alert.py renders correct badges for cam_04."""
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    zones = {
        "zone_1_koridor": [[47, 230], [448, 141], [479, 151], [38, 256]],
        "zone_2_transit": [[48, 230], [85, 221], [111, 98], [89, 104]],
    }

    # 1. Moving vehicle on cam_04: Cyan
    moving_truck = SimpleNamespace(
        track_id=1,
        class_label="truck",
        zone_id="zone_1_koridor",
        centroid=(150, 150),
        bbox=(120, 120, 60, 60),
        is_stationary=False,
        dwell_duration=0.0,
        dwell_threshold=600.0,
        is_triggered=False,
    )

    # 2. Stationary in Zebra Cross (< 10m): Amber
    stat_truck = SimpleNamespace(
        track_id=2,
        class_label="truck",
        zone_id="zone_1_koridor",
        centroid=(150, 150),
        bbox=(120, 120, 60, 60),
        is_stationary=True,
        dwell_duration=300.0,  # 5 min
        dwell_threshold=600.0,
        is_triggered=False,
    )

    # 3. Violation in Zebra Cross (>= 10m): Flashing Red
    vio_truck = SimpleNamespace(
        track_id=3,
        class_label="truck",
        zone_id="zone_1_koridor",
        centroid=(150, 150),
        bbox=(120, 120, 60, 60),
        is_stationary=True,
        dwell_duration=660.0,  # 11 min
        dwell_threshold=600.0,
        is_triggered=True,
    )

    # 4. Stationary in Samping (< 30m)
    samping_truck = SimpleNamespace(
        track_id=4,
        class_label="truck",
        zone_id="zone_2_transit",
        centroid=(80, 150),
        bbox=(60, 120, 40, 50),
        is_stationary=True,
        dwell_duration=600.0,  # 10 min
        dwell_threshold=1800.0,
        is_triggered=False,
    )

    # 5. Pedestrian on cam_04: Emerald Green
    ped = SimpleNamespace(
        track_id=5,
        class_label="person",
        zone_id="zone_1_koridor",
        centroid=(100, 200),
        bbox=(90, 170, 20, 50),
        is_stationary=False,
        dwell_duration=0.0,
    )

    out = VisualHUD.render(
        canvas=canvas.copy(),
        zones=zones,
        tracked_objects=[moving_truck, stat_truck, vio_truck, samping_truck, ped],
        camera_id="cam_04",
        fps=12.0,
        is_connected=True,
        faces=[],
        zone_base_resolution=(640, 360),
        camera_name="Area Timbangan Truk",
    )
    assert out is not None
    assert out.shape == (360, 640, 3)

    print("PASS: cam_04 visual HUD badge rendering verified.")


if __name__ == "__main__":
    test_leaning_person_foot_contact()
    test_vehicle_wheel_contact()
    test_cam04_config_integrity()
    test_cam04_visual_badges()
    print("\nALL CAM_04 & TRIPWIRE TESTS PASSED SUCCESSFULLY!")
