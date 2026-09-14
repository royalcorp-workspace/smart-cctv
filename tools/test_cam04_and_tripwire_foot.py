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
    engine = TripwireEngine(lines=lines_cfg, cooldown_sec=1.0, min_track_frames=1)

    # Frame 1: Person upright before line (foot at y=93, centroid at y=57.5)
    # bbox = (x1, y1, w, h) = (100, 20, 40, 75)
    # foot_y = 20 + 75 - 2 = 93 (< 100)
    track_p = SimpleNamespace(
        track_id=1,
        class_label="person",
        centroid=(120, 57),
        bbox=(100, 20, 40, 75),
        confidence=0.85,
        frame_count=4,
        missed_frames=0,
        is_active_this_frame=True,
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
    engine = TripwireEngine(lines=lines_cfg, cooldown_sec=1.0, min_track_frames=1)

    # Frame 1: Truck before line
    # bbox = (50, 100, 120, 100) -> wheel_y = 100 + 100 - 5 = 195 (< 200)
    truck = SimpleNamespace(
        track_id=10,
        class_label="truck",
        centroid=(110, 150),
        bbox=(50, 100, 120, 100),
        frame_count=4,
        missed_frames=0,
        is_active_this_frame=True,
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


def test_liang_barsky_line_overlap():
    """Verify Liang-Barsky parametric line clipping overlap computation."""
    from engine.line_crossing import compute_line_box_overlap

    # Bounding box: [100, 100, 200, 100] -> x in [100, 300], y in [100, 200]
    bbox = (100, 100, 200, 100)

    # 1. Segment fully inside
    ov_inside = compute_line_box_overlap((120, 120), (200, 150), bbox)
    assert abs(ov_inside - 1.0) < 1e-4, f"Expected 1.0, got {ov_inside}"

    # 2. Segment fully outside
    ov_outside = compute_line_box_overlap((10, 10), (50, 50), bbox)
    assert ov_outside == 0.0, f"Expected 0.0, got {ov_outside}"

    # 3. Horizontal segment through box: (0, 150) to (500, 150), total length 500, inside length 200 (x=100 to 300)
    ov_cross = compute_line_box_overlap((0, 150), (500, 150), bbox)
    assert abs(ov_cross - 0.40) < 1e-4, f"Expected 0.40, got {ov_cross}"

    # 4. Partial overlap >= 50%: (50, 150) to (250, 150), total length 200, inside length 150 (x=100 to 250) -> 75%
    ov_75 = compute_line_box_overlap((50, 150), (250, 150), bbox)
    assert abs(ov_75 - 0.75) < 1e-4, f"Expected 0.75, got {ov_75}"

    # 5. Invalid / empty bbox
    assert compute_line_box_overlap((0, 0), (10, 10), ()) == 0.0

    print("PASS: Liang-Barsky line overlap calculation verified.")


def test_stationary_truck_line_occlusion_and_recovery():
    """Verify stationary truck 15s occlusion latch, crossing trigger suppression, and auto-recovery."""
    lines_cfg = {
        "line_1": {
            "name": "Tripwire 1",
            "p1": [88, 217],
            "p2": [219, 101],
            "direction": "both",
            "target_classes": ["person", "car", "bus", "truck"],
        }
    }
    engine = TripwireEngine(
        lines=lines_cfg,
        cooldown_sec=1.0,
        occlusion_overlap_threshold=0.50,
        occlusion_duration_sec=15.0,
        min_track_frames=1,
        max_step_px=200.0,
    )

    # Stationary truck parked across Tripwire 1 (bbox covering > 80% of line_1)
    truck = SimpleNamespace(
        track_id=101,
        class_label="truck",
        centroid=(150, 160),
        bbox=(70, 90, 160, 140),
        is_stationary=True,
        frame_count=5,
        missed_frames=0,
        is_active_this_frame=True,
    )

    # t = 100.0s: first detection
    engine.update([truck], timestamp=100.0)
    assert "line_1" not in engine.get_occluded_lines(), "Line must not be occluded immediately at t=0s."

    # t = 114.0s: 14.0s elapsed (< 15.0s)
    engine.update([truck], timestamp=114.0)
    assert "line_1" not in engine.get_occluded_lines(), "Line must not be occluded before 15.0s."

    # t = 115.0s: 15.0s elapsed -> Line 1 enters OCCLUDED state!
    engine.update([truck], timestamp=115.0)
    assert "line_1" in engine.get_occluded_lines(), "Line 1 must be flagged OCCLUDED after 15.0s of stationary truck cover."

    # Test crossing trigger suppression: a vehicle crosses line_1 while it is occluded
    intruder = SimpleNamespace(
        track_id=202,
        class_label="car",
        centroid=(100, 180),
        bbox=(80, 160, 40, 40),
        is_stationary=False,
        frame_count=5,
        missed_frames=0,
        is_active_this_frame=True,
    )
    # Register frame 1 for intruder
    engine.update([truck, intruder], timestamp=116.0)
    # Move intruder across line_1 in frame 2
    intruder.bbox = (150, 130, 40, 40)
    intruder.centroid = (170, 150)
    events = engine.update([truck, intruder], timestamp=116.1)
    assert len(events) == 0, f"Crossing must be SUPPRESSED while line is OCCLUDED! Got events: {events}"

    # Test Auto-Recovery when truck moves (shift > 15px)
    truck.centroid = (200, 200)  # shifted > 15px from initial anchor (150, 160)
    truck.is_stationary = False
    engine.update([truck], timestamp=117.0)
    assert "line_1" not in engine.get_occluded_lines(), "Line 1 must auto-recover to ACTIVE immediately once truck moves."

    # Now intruder crosses line_1 again after recovery -> event must trigger!
    intruder2 = SimpleNamespace(
        track_id=203,
        class_label="car",
        centroid=(100, 200),
        bbox=(80, 180, 40, 40),
        is_stationary=False,
        frame_count=5,
        missed_frames=0,
        is_active_this_frame=True,
    )
    engine.update([intruder2], timestamp=118.0)
    intruder2.bbox = (160, 70, 40, 40)  # wheel_y = 70 + 40 - 5 = 105 (< 135.6, across the line)
    intruder2.centroid = (180, 90)
    events2 = engine.update([intruder2], timestamp=118.1)
    assert len(events2) == 1, f"Expected crossing event after line recovery, got {len(events2)}"
    assert events2[0].line_id == "line_1"

    # Test Auto-Recovery when truck leaves frame
    # Bring truck back to occlude again
    truck2 = SimpleNamespace(
        track_id=102,
        class_label="truck",
        centroid=(150, 160),
        bbox=(70, 90, 160, 140),
        is_stationary=True,
        frame_count=5,
        missed_frames=0,
        is_active_this_frame=True,
    )
    engine.update([truck2], timestamp=200.0)
    engine.update([truck2], timestamp=216.0)
    assert "line_1" in engine.get_occluded_lines()

    # Truck disappears / leaves frame:
    engine.update([], timestamp=217.0)
    assert "line_1" not in engine.get_occluded_lines(), "Line 1 must auto-recover when vehicle leaves the frame."

    print("PASS: Stationary truck line occlusion latch (15s), suppression, and auto-recovery verified.")


def test_foot_ground_filter_phantom_person():
    """Verify Foot Ground Filter ignores phantom person inside truck body while detecting real pedestrians."""
    lines_cfg = {
        "trip_test": {
            "name": "Tripwire Test",
            "p1": [0, 100],
            "p2": [600, 100],
            "direction": "both",
            "target_classes": ["person"],
        }
    }
    engine = TripwireEngine(lines=lines_cfg, cooldown_sec=1.0, min_track_frames=1)

    # Stationary truck at platform: bbox = (50, 40, 200, 100).
    # Upper/middle body: y in [40, 40 + 0.85 * 100 = 125]. Ground is at y >= 140.
    truck = SimpleNamespace(
        track_id=50,
        class_label="truck",
        centroid=(150, 90),
        bbox=(50, 40, 200, 100),
        is_stationary=True,
        frame_count=5,
        missed_frames=0,
        is_active_this_frame=True,
    )

    # 1. Phantom person from flapping tarp on top of truck:
    # bbox = (120, 50, 20, 40) -> foot_y = 50 + 40 - 2 = 88 (< 125, inside upper truck body)
    phantom_p = SimpleNamespace(
        track_id=901,
        class_label="person",
        centroid=(130, 70),
        bbox=(120, 50, 20, 40),
        confidence=0.85,
        frame_count=5,
        missed_frames=0,
        is_active_this_frame=True,
    )
    engine.update([truck, phantom_p], timestamp=500.0)

    # Frame 2: tarp shifts and foot moves across y=100 line to y=108 (still inside truck body <= 125)
    phantom_p.bbox = (120, 70, 20, 40)  # foot_y = 70 + 40 - 2 = 108 (< 125)
    phantom_p.centroid = (130, 90)
    events_phantom = engine.update([truck, phantom_p], timestamp=500.1)
    assert len(events_phantom) == 0, (
        f"Phantom person with foot inside truck body must NOT trigger line crossing! Events: {events_phantom}"
    )

    # 2. Real pedestrian walking on the asphalt ground past the truck:
    # bbox = (120, 60, 20, 80) -> foot_y = 60 + 80 - 2 = 138 (> 125, ground level)
    real_p = SimpleNamespace(
        track_id=902,
        class_label="person",
        centroid=(130, 80),
        bbox=(120, 20, 20, 75),  # foot_y = 20 + 75 - 2 = 93 (< 100)
        confidence=0.85,
        frame_count=5,
        missed_frames=0,
        is_active_this_frame=True,
    )
    engine.update([truck, real_p], timestamp=600.0)

    # Step across line to ground y=138:
    real_p.bbox = (120, 60, 20, 80)  # foot_y = 60 + 80 - 2 = 138 (> 100 and > 125)
    real_p.centroid = (130, 100)
    events_real = engine.update([truck, real_p], timestamp=600.1)
    assert len(events_real) == 1, f"Real pedestrian on ground must trigger crossing! Got: {len(events_real)}"
    assert events_real[0].line_id == "trip_test"

    print("PASS: Foot Ground Filter (phantom tarp vs real ground pedestrian) verified.")


def test_visual_hud_occluded_line():
    """Verify VisualHUD renders occluded line in dim gray with [OCCLUDED] badge, and draw alias works."""
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    lines = {
        "line_1": {
            "name": "Tripwire 1",
            "p1": [88, 217],
            "p2": [219, 101],
            "direction": "both",
        },
        "line_2": {
            "name": "Tripwire 2",
            "p1": [456, 159],
            "p2": [438, 238],
            "direction": "both",
        }
    }

    # Render with line_1 occluded and line_2 safe
    out_render = VisualHUD.render(
        canvas=canvas.copy(),
        zones={},
        tracked_objects=[],
        camera_id="cam_04",
        fps=13.0,
        is_connected=True,
        lines=lines,
        occluded_lines={"line_1"},
    )
    assert out_render is not None
    assert out_render.shape == (360, 640, 3)

    # Verify VisualHUD.draw alias functions identically
    out_draw = VisualHUD.draw(
        canvas=canvas.copy(),
        zones={},
        tracked_objects=[],
        camera_id="cam_04",
        fps=13.0,
        is_connected=True,
        lines=lines,
        occluded_lines={"line_1"},
    )
    assert out_draw is not None
    assert np.array_equal(out_render, out_draw), "VisualHUD.draw and VisualHUD.render must produce identical output."

    print("PASS: VisualHUD occluded line rendering and draw alias verified.")


def test_anti_ghost_immature_and_low_confidence():
    """Verify that immature (< 4 frames) and low-confidence (< 0.40) noise tracks are suppressed."""
    lines_cfg = {
        "line_1": {"name": "Tripwire 1", "p1": [100, 100], "p2": [200, 100], "direction": "both", "target_classes": ["person"]}
    }
    engine = TripwireEngine(lines=lines_cfg, min_track_frames=4, min_confidence=0.40)

    # Transient ghost detection (frame 1)
    ghost = SimpleNamespace(
        track_id=88, class_label="person", centroid=(150, 90), bbox=(140, 70, 20, 45),
        confidence=0.35, frame_count=1, missed_frames=0, is_active_this_frame=True
    )
    e1 = engine.update([ghost], timestamp=10.0)
    assert len(e1) == 0

    # Frame 2: shifts across line (foot from 70+45-2=113 > 100), but frame_count=2 (< 4)
    ghost.bbox = (140, 75, 20, 45)
    ghost.frame_count = 2
    e2 = engine.update([ghost], timestamp=10.1)
    assert len(e2) == 0, "Immature track (frame_count=2 < 4) must NOT trigger crossing!"

    # Frame 3: frame_count=3 (< 4)
    ghost.bbox = (140, 80, 20, 45)
    ghost.frame_count = 3
    e3 = engine.update([ghost], timestamp=10.2)
    assert len(e3) == 0, "Immature track (frame_count=3 < 4) must NOT trigger crossing!"

    # Frame 4: mature (frame_count=4), but confidence is low (0.35 < 0.40)
    ghost.bbox = (140, 85, 20, 45)
    ghost.frame_count = 4
    ghost.confidence = 0.35
    e4 = engine.update([ghost], timestamp=10.3)
    assert len(e4) == 0, "Low confidence track (0.35 < 0.40) must NOT trigger crossing!"

    print("PASS: Anti-ghost immature track and low-confidence suppression verified.")


def test_anti_ghost_micro_bbox():
    """Verify that micro noise detections (< 400 px² bbox area) are suppressed."""
    lines_cfg = {
        "line_1": {"name": "Tripwire 1", "p1": [100, 100], "p2": [200, 100], "direction": "both", "target_classes": ["person"]}
    }
    engine = TripwireEngine(lines=lines_cfg, min_track_frames=4, min_bbox_area=400.0)

    # Tiny micro-box: 15x20 = 300 px² (< 400)
    micro = SimpleNamespace(
        track_id=99, class_label="person", centroid=(150, 90), bbox=(140, 85, 15, 20),
        confidence=0.75, frame_count=5, missed_frames=0, is_active_this_frame=True
    )
    engine.update([micro], timestamp=20.0)

    # Step across line: foot_y = 95 + 20 - 2 = 113 (> 100)
    micro.bbox = (140, 95, 15, 20)
    events = engine.update([micro], timestamp=20.1)
    assert len(events) == 0, "Micro bbox (< 400 px²) must NOT trigger tripwire!"
    print("PASS: Anti-ghost micro bounding box suppression verified.")


def test_anti_ghost_abnormal_step_jump():
    """Verify that sudden abnormal position jumps (> 65 px) across tripwire are suppressed."""
    lines_cfg = {
        "line_1": {"name": "Tripwire 1", "p1": [100, 100], "p2": [200, 100], "direction": "both", "target_classes": ["person"]}
    }
    engine = TripwireEngine(lines=lines_cfg, min_track_frames=4, max_step_px=65.0)

    # Frame 1: Person at foot_y = 50 (well before y=100)
    p = SimpleNamespace(
        track_id=77, class_label="person", centroid=(150, 20), bbox=(130, 0, 40, 52),
        confidence=0.85, frame_count=5, missed_frames=0, is_active_this_frame=True
    )
    engine.update([p], timestamp=30.0)

    # Frame 2: Tracker assigns noise 90 pixels away across the line (delta y = 90 px > 65 px)
    p.bbox = (130, 90, 40, 52)
    p.centroid = (150, 110)
    events = engine.update([p], timestamp=30.1)
    assert len(events) == 0, f"Abnormal step jump (> 65 px) must NOT trigger crossing! Got {events}"
    print("PASS: Anti-ghost abnormal step jump suppression verified.")


def test_anti_ghost_coasting_track():
    """Verify that a track coasting without current YOLO detection (missed_frames > 0) is suppressed."""
    lines_cfg = {
        "line_1": {"name": "Tripwire 1", "p1": [100, 100], "p2": [200, 100], "direction": "both", "target_classes": ["person"]}
    }
    engine = TripwireEngine(lines=lines_cfg, min_track_frames=4)

    p = SimpleNamespace(
        track_id=66, class_label="person", centroid=(150, 80), bbox=(130, 40, 40, 55),
        confidence=0.85, frame_count=5, missed_frames=0, is_active_this_frame=True
    )
    engine.update([p], timestamp=40.0)

    # Frame 2: Object coasting (missed_frames=1, is_active_this_frame=False)
    p.bbox = (130, 55, 40, 55)  # foot_y = 55 + 55 - 2 = 108 (> 100)
    p.missed_frames = 1
    p.is_active_this_frame = False
    events = engine.update([p], timestamp=40.1)
    assert len(events) == 0, "Coasting track with missed_frames > 0 must NOT trigger crossing!"
    print("PASS: Anti-ghost coasting track suppression verified.")


def test_anti_ghost_valid_mature_pedestrian_crosses():
    """Verify that a legitimate mature pedestrian crossing reliably triggers the tripwire."""
    lines_cfg = {
        "line_1": {"name": "Tripwire 1", "p1": [100, 100], "p2": [200, 100], "direction": "both", "target_classes": ["person"]}
    }
    engine = TripwireEngine(lines=lines_cfg, min_track_frames=4, min_confidence=0.40, min_bbox_area=400.0, max_step_px=65.0)

    # Mature, active, high-confidence person: walking step of 15 px across line
    p = SimpleNamespace(
        track_id=55, class_label="person", centroid=(150, 70), bbox=(130, 30, 40, 65),
        confidence=0.82, frame_count=5, missed_frames=0, is_active_this_frame=True
    )
    engine.update([p], timestamp=50.0)

    # Step across line (foot_y = 45 + 65 - 2 = 108 > 100, step dx=0, dy=15 <= 65)
    p.bbox = (130, 45, 40, 65)
    p.centroid = (150, 85)
    events = engine.update([p], timestamp=50.1)
    assert len(events) == 1, f"Legitimate mature crossing must trigger 1 event, got {len(events)}"
    assert events[0].line_id == "line_1"
    assert events[0].direction == "B_to_A"
    print("PASS: Legitimate mature pedestrian crossing verified.")


if __name__ == "__main__":
    test_leaning_person_foot_contact()
    test_vehicle_wheel_contact()
    test_cam04_config_integrity()
    test_cam04_visual_badges()
    test_liang_barsky_line_overlap()
    test_stationary_truck_line_occlusion_and_recovery()
    test_foot_ground_filter_phantom_person()
    test_visual_hud_occluded_line()
    test_anti_ghost_immature_and_low_confidence()
    test_anti_ghost_micro_bbox()
    test_anti_ghost_abnormal_step_jump()
    test_anti_ghost_coasting_track()
    test_anti_ghost_valid_mature_pedestrian_crosses()
    print("\nALL CAM_04 & TRIPWIRE ANTI-GHOST TESTS PASSED SUCCESSFULLY!")

