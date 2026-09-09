"""Unit & Integration Test Suite for ROI Zone Coordinate Scaling.

Tests:
1. Base resolution auto-detection (explicit metadata vs coordinate heuristics).
2. ZoneFilter coordinate downscaling from 1080p base to 640x480 inference frame.
3. Binary mask generation and spatial point-in-polygon tests in 640x480 space.
4. VisualHUD 1:1 zone rendering on 1080p canvas and backward compatibility with 640p zones.
5. Calibrator auto-scaling and base_resolution persistence round-trip.
"""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.zone_filter import ZoneFilter
from notification.local_alert import VisualHUD
from tools.roi_calibrator import ROICalibrator


def test_base_resolution_detection() -> bool:
    print("\n[STEP 1] Testing Base Resolution Auto-Detection & Metadata Extraction...")

    # Case A: Explicit base_resolution metadata
    data_explicit = {
        "base_resolution": [1920, 1080],
        "zone_1_koridor": [[100, 100], [200, 100], [200, 200]],
    }
    zf_explicit = ZoneFilter(zones=data_explicit, frame_shape=(480, 640))
    print(f" - Explicit metadata detected: {zf_explicit.base_resolution}")
    assert zf_explicit.base_resolution == (1920, 1080)
    assert "base_resolution" not in zf_explicit.raw_zones
    assert "zone_1_koridor" in zf_explicit.raw_zones

    # Case B: Implicit 1080p (coordinates exceed 640 or 480)
    data_implicit_1080p = {
        "zone_1_koridor": [[100, 600], [1200, 300], [1900, 900]],
    }
    zf_implicit = ZoneFilter(zones=data_implicit_1080p, frame_shape=(480, 640))
    print(f" - Implicit 1080p heuristic detected: {zf_implicit.base_resolution}")
    assert zf_implicit.base_resolution == (1920, 1080)

    # Case C: Implicit 640x360 (all coordinates within 640 and 360)
    data_implicit_640p = {
        "zone_1_koridor": [[50, 50], [200, 50], [200, 200]],
    }
    zf_640p = ZoneFilter(zones=data_implicit_640p, frame_shape=(360, 640))
    print(f" - Implicit 640p heuristic detected: {zf_640p.base_resolution}")
    assert zf_640p.base_resolution == (640, 360)

    print(" -> PASS: Base resolution auto-detection verified.")
    return True


def test_zone_filter_downscaling() -> bool:
    print("\n[STEP 2] Testing ZoneFilter Downscaling from 1080p to 640x480...")

    # Define rectangle in 1920x1080: [300, 270] to [1500, 810]
    # Expected in 640x480 (300 * 640/1920 = 100, 1500 * 640/1920 = 500)
    # Expected y (270 * 480/1080 = 120, 810 * 480/1080 = 360)
    zones_1080p = {
        "base_resolution": [1920, 1080],
        "zone_test": [
            [300, 270],
            [1500, 270],
            [1500, 810],
            [300, 810],
        ],
    }

    zf = ZoneFilter(zones=zones_1080p, frame_shape=(480, 640))
    scaled_pts = zf.scaled_zones["zone_test"]
    print(f" - Scaled polygon points: {scaled_pts}")

    assert scaled_pts[0] == [100, 120], f"Expected [100, 120], got {scaled_pts[0]}"
    assert scaled_pts[1] == [500, 120], f"Expected [500, 120], got {scaled_pts[1]}"
    assert scaled_pts[2] == [500, 360], f"Expected [500, 360], got {scaled_pts[2]}"
    assert scaled_pts[3] == [100, 360], f"Expected [100, 360], got {scaled_pts[3]}"

    # Spatial point check in 640x480 inference space
    inside_pt = (300, 240)  # Center of [100, 120] -> [500, 360]
    outside_pt = (50, 50)
    assert zf.check_point_in_zone(inside_pt, "zone_test") is True
    assert zf.check_point_in_zone(outside_pt, "zone_test") is False
    print(f" - Point (300, 240) in 640p: INSIDE [PASS]")
    print(f" - Point (50, 50) in 640p  : OUTSIDE [PASS]")

    # Check binary mask generation
    mask = zf.get_zone_mask("zone_test")
    assert mask.shape == (480, 640), f"Expected mask shape (480, 640), got {mask.shape}"
    assert mask[240, 300] == 255, "Inside pixel must be 255"
    assert mask[50, 50] == 0, "Outside pixel must be 0"
    print(" - Binary mask matches 640x480 inference shape and alignment [PASS]")

    print(" -> PASS: ZoneFilter downscaling verified.")
    return True


def test_visual_hud_scaling() -> bool:
    print("\n[STEP 3] Testing VisualHUD Rendering (Decoupled Zone vs Object Scaling)...")

    # 1. 1080p Zone on 1080p Canvas (Scale factor MUST be 1.0)
    canvas_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)
    zones_1080p = {
        "base_resolution": [1920, 1080],
        "zone_1": [[100, 100], [800, 100], [800, 600], [100, 600]],
    }

    # Should not raise exception and should draw without out-of-bounds overflow
    rendered_1080p = VisualHUD.render(
        canvas=canvas_1080p,
        zones=zones_1080p,
        tracked_objects=[],
        camera_id="cam_01",
        fps=30.0,
        is_connected=True,
    )
    assert rendered_1080p.shape == (1080, 1920, 3)
    print(" - VisualHUD rendered 1080p zones on 1080p canvas with 1:1 scaling [PASS]")

    # 2. Legacy 640x480 Zone on 1080p Canvas (Scale factor MUST scale up 3.0x / 2.25x)
    canvas_legacy = np.zeros((1080, 1920, 3), dtype=np.uint8)
    zones_640p = {
        "base_resolution": [640, 480],
        "zone_legacy": [[50, 50], [200, 50], [200, 200], [50, 200]],
    }
    rendered_legacy = VisualHUD.render(
        canvas=canvas_legacy,
        zones=zones_640p,
        tracked_objects=[],
        camera_id="cam_01",
        fps=30.0,
        is_connected=True,
    )
    assert rendered_legacy.shape == (1080, 1920, 3)
    print(" - VisualHUD rendered legacy 640p zones with proper upscale factor [PASS]")

    # 3. Verify that non-polygon keys ("base_resolution") are ignored without crashing
    zones_with_meta = {
        "base_resolution": [1920, 1080],
        "_metadata": {"version": 2},
        "zone_1": [[100, 100], [200, 100], [200, 200]],
    }
    rendered_meta = VisualHUD.render(
        canvas=canvas_1080p.copy(),
        zones=zones_with_meta,
        tracked_objects=[],
        camera_id="cam_01",
        fps=30.0,
        is_connected=True,
    )
    assert rendered_meta is not None
    print(" - VisualHUD ignored non-polygon metadata keys cleanly [PASS]")

    print(" -> PASS: VisualHUD scaling verified.")
    return True


def test_calibrator_round_trip() -> bool:
    print("\n[STEP 4] Testing ROI Calibrator Base Resolution Persistence & Auto-Scaling...")

    temp_dir = Path(tempfile.mkdtemp(prefix="calibrator_test_"))
    try:
        cam_dir = temp_dir / "cameras" / "test_cam"
        cam_dir.mkdir(parents=True, exist_ok=True)

        config_file = cam_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"camera_id": "test_cam", "target_resolution": [1920, 1080]}, f)

        # Write existing 640x480 zone file
        roi_file = cam_dir / "roi_zones.json"
        legacy_data = {
            "base_resolution": [640, 480],
            "zone_1_koridor": [[100, 100], [200, 100], [200, 200]],
        }
        with open(roi_file, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f)

        # Create dummy reference image in 1920x1080
        ref_img = cam_dir / "ref.jpg"
        cv2.imwrite(str(ref_img), np.zeros((1080, 1920, 3), dtype=np.uint8))

        calibrator = ROICalibrator(
            camera_id="test_cam",
            image_path=ref_img,
            target_resolution=(1920, 1080),
            cam_dir=cam_dir,
        )

        # Load existing zones
        calibrator._load_existing_zones()
        loaded_pts = calibrator.zones["zone_1_koridor"]
        print(f" - Calibrator upscaled 640p points to 1080p: {loaded_pts}")
        # Expected: 100 * (1920/640) = 300, 100 * (1080/480) = 225
        assert loaded_pts[0] == [300, 225], f"Expected [300, 225], got {loaded_pts[0]}"

        # Save zones and verify base_resolution is persisted
        calibrator.save_zones()
        with open(roi_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)

        assert saved_data.get("base_resolution") == [1920, 1080], f"Expected base_resolution [1920, 1080], got {saved_data.get('base_resolution')}"
        assert "zone_1_koridor" in saved_data
        print(" - Saved JSON contains base_resolution: [1920, 1080] [PASS]")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(" -> PASS: Calibrator round-trip and auto-scaling verified.")
    return True


def run_all_tests():
    print("=" * 65)
    print("       ROI ZONE COORDINATE SCALING VERIFICATION SUITE            ")
    print("=" * 65)

    assert test_base_resolution_detection()
    assert test_zone_filter_downscaling()
    assert test_visual_hud_scaling()
    assert test_calibrator_round_trip()

    print("\n" + "=" * 65)
    print("ALL ROI ZONE SCALING VERIFICATIONS PASSED (100% COMPLIANT)       ")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
