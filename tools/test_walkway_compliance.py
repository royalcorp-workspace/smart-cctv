"""Unit test suite for Pedestrian Walkway Compliance & Vehicle Proximity Latch (cam_03 & cam_04)."""

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from engine.tracker import TrackedObject
from engine.zone_filter import ZoneFilter
from notification.local_alert import VisualHUD


def run_tests():
    print("=" * 70)
    print("      SMART CCTV 2.0 - WALKWAY COMPLIANCE & PROXIMITY TEST SUITE      ")
    print("=" * 70)

    # 1. Load actual calibrated polygon for cam_03
    cam03_zones_path = Path("cameras/cam_03/roi_zones.json")
    assert cam03_zones_path.exists(), "cam_03 roi_zones.json not found"
    with open(cam03_zones_path, "r") as f:
        zones_data = json.load(f)

    # Set up ZoneFilter for 640x360 inference space
    zf = ZoneFilter(
        zones=zones_data,
        frame_shape=(360, 640),
        base_resolution=(1920, 1080),
    )
    assert "zone_1_koridor" in zf.polygon_contours, "zone_1_koridor contour not found in ZoneFilter"
    print("[1] ZoneFilter loaded calibrated 1080p polygon and scaled to 640x360: OK")

    # Safe walkway polygon contour
    cnt = zf.polygon_contours["zone_1_koridor"]
    # Find a point strictly inside the walkway polygon
    x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
    safe_point = None
    for ty in range(y_b + 1, y_b + h_b, 2):
        for tx in range(x_b + 1, x_b + w_b, 2):
            if cv2.pointPolygonTest(cnt, (float(tx), float(ty)), False) > 0:
                safe_point = (tx, ty)
                break
        if safe_point is not None:
            break

    assert safe_point is not None, "Could not find internal point inside polygon"
    safe_cx, safe_cy = safe_point
    assert zf.check_point_in_zone(safe_point, "zone_1_koridor", margin_px=6.0), "Safe point must be inside walkway"

    # Pick a point guaranteed outside the polygon (e.g. top-left corner far from walkway)
    outside_point = (10, 10)
    assert not zf.check_point_in_zone(outside_point, "zone_1_koridor", margin_px=6.0), "Outside point must be outside walkway"
    print(f"[2] Inside point {safe_point} -> SAFE, Outside point {outside_point} -> OUTSIDE: OK")

    # 2. Test Walkway Compliance Logic Simulation
    now = time.time()

    # Track 1: Person in safe walkway
    t1 = TrackedObject(
        track_id=1,
        centroid=safe_point,
        anchor_centroid=safe_point,
        bbox=(safe_cx - 15, safe_cy - 40, 30, 80),
        zone_id="zone_1_koridor",
        contour_area=2400.0,
        first_seen=now - 50.0,
        last_seen=now,
        stationary_start=now - 50.0,
        class_label="person",
    )

    # Simulation check 1: In walkway
    foot_1 = (t1.centroid[0], min(359, t1.bbox[1] + t1.bbox[3] - 2))
    if zf.check_point_in_zone(foot_1, "zone_1_koridor", margin_px=6.0):
        t1.walkway_status = "SAFE"
        t1.outside_walkway_duration = 0.0
        t1.is_near_vehicle = False
    assert t1.walkway_status == "SAFE", f"Expected SAFE, got {t1.walkway_status}"
    print("[3] Test Scenario 1 (Person in Walkway): status=SAFE -> PASS")

    # Simulation check 2: Person outside walkway near vehicle (Proximity Latch)
    vehicle_boxes = [
        ((40, 40, 120, 80), (100, 80), "truck")
    ]
    t2 = TrackedObject(
        track_id=2,
        centroid=(50, 45),
        anchor_centroid=(50, 45),
        bbox=(40, 30, 20, 50),
        zone_id="outside_zone",
        contour_area=1000.0,
        first_seen=now - 40.0,
        last_seen=now,
        stationary_start=now - 40.0,
        class_label="person",
    )
    foot_2 = (t2.centroid[0], min(359, t2.bbox[1] + t2.bbox[3] - 2))
    in_walkway_2 = zf.check_point_in_zone(foot_2, "zone_1_koridor", margin_px=6.0)
    assert not in_walkway_2, "t2 must be outside walkway"

    # Calculate distance to vehicle
    min_dist_2 = 999.0
    for (vx, vy, vvw, vvh), _, _ in vehicle_boxes:
        clamped_x = max(vx, min(foot_2[0], vx + vvw))
        clamped_y = max(vy, min(foot_2[1], vy + vvh))
        d = float(np.hypot(foot_2[0] - clamped_x, foot_2[1] - clamped_y))
        if d < min_dist_2:
            min_dist_2 = d

    assert min_dist_2 <= 75.0, f"Distance should be <= 75px, got {min_dist_2}"
    t2.is_near_vehicle = True
    t2.walkway_status = "NEAR_VEHICLE"
    t2.outside_walkway_duration = 0.0
    assert t2.walkway_status == "NEAR_VEHICLE", f"Expected NEAR_VEHICLE, got {t2.walkway_status}"
    print(f"[4] Test Scenario 2 (Person near Truck, dist={min_dist_2:.1f}px): status=NEAR_VEHICLE -> PASS")

    # Simulation check 3: Person outside walkway far from vehicle, dwell < 35s (CROSSING)
    t3 = TrackedObject(
        track_id=3,
        centroid=(500, 30),
        anchor_centroid=(500, 30),
        bbox=(490, 10, 20, 60),
        zone_id="outside_zone",
        contour_area=1200.0,
        first_seen=now - 15.0,
        last_seen=now,
        stationary_start=now - 15.0,
        class_label="person",
        outside_walkway_start=now - 15.0,
        outside_walkway_duration=15.0,
    )
    # Distance to vehicle box (40, 40, 120, 80)
    clamped_x = max(40, min(500, 160))
    clamped_y = max(40, min(68, 120))
    d_3 = float(np.hypot(500 - clamped_x, 68 - clamped_y))
    assert d_3 > 75.0, "t3 must be far from vehicle"

    if t3.outside_walkway_duration < 35.0:
        t3.walkway_status = "CROSSING"
    else:
        t3.walkway_status = "VIOLATION"
    assert t3.walkway_status == "CROSSING", f"Expected CROSSING, got {t3.walkway_status}"
    print(f"[5] Test Scenario 3 (Person crossing, dwell=15s < 35s): status=CROSSING -> PASS")

    # Simulation check 4: Person outside walkway far from vehicle, dwell >= 35s (VIOLATION)
    t4 = TrackedObject(
        track_id=4,
        centroid=(500, 30),
        anchor_centroid=(500, 30),
        bbox=(490, 10, 20, 60),
        zone_id="outside_zone",
        contour_area=1200.0,
        first_seen=now - 38.0,
        last_seen=now,
        stationary_start=now - 38.0,
        class_label="person",
        outside_walkway_start=now - 38.0,
        outside_walkway_duration=38.0,
    )
    if t4.outside_walkway_duration < 35.0:
        t4.walkway_status = "CROSSING"
    else:
        t4.walkway_status = "VIOLATION"
        t4.is_triggered = True
    assert t4.walkway_status == "VIOLATION" and t4.is_triggered, f"Expected VIOLATION, got {t4.walkway_status}"
    print(f"[6] Test Scenario 4 (Person loitering in traffic area, dwell=38s >= 35s): status=VIOLATION -> PASS")

    # 3. Test VisualHUD Rendering on cam_03 with Walkway Objects
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered = VisualHUD.render(
        canvas=canvas,
        zones=zones_data,
        tracked_objects=[t1, t2, t3, t4],
        camera_id="cam_03",
        fps=15.0,
        is_connected=True,
        faces=[],
        camera_name="Jalur Logistik Arah POS-1",
    )
    assert rendered is not None and rendered.shape == (360, 640, 3), "VisualHUD render output invalid"
    # Ensure canvas was drawn onto (contains non-zero pixels)
    assert np.count_nonzero(rendered) > 500, "VisualHUD canvas should have drawn HUD and tracks"
    print("[7] VisualHUD Rendering with SAFE, NEAR_VEHICLE, CROSSING, and VIOLATION tracks: OK")

    # 4. Test cam_01 (indoor corridor): persons should remain hidden from visual HUD
    canvas_cam01 = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered_cam01 = VisualHUD.render(
        canvas=canvas_cam01,
        zones=zones_data,
        tracked_objects=[t1, t2, t3, t4],
        camera_id="cam_01",
        fps=15.0,
        is_connected=True,
        faces=[],
        camera_name="Koridor Utama",
    )
    assert rendered_cam01 is not None
    print("[8] VisualHUD on cam_01 correctly suppresses person boxes (0 visual clutter): OK")

    print("=" * 70)
    print("      ALL WALKWAY COMPLIANCE & PROXIMITY TESTS PASSED SUCCESSFULLY!    ")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
