"""Unit test suite for Dynamic Multi-ROI & Tripwire State Machine in ROI Calibrator."""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.zone_filter import load_roi_data, ZoneFilter
from tools.roi_calibrator import ROICalibrator


def test_calibrator_state_machine():
    """Verify keys 1-9, n, l, d, u, c in ROICalibrator state machine."""
    calib = ROICalibrator(camera_id="test_cam", interactive=False)

    # Initially: zone_1 and zone_2 exist
    assert "zone_1" in calib.zones
    assert "zone_2" in calib.zones
    assert calib.active_target == "zone_1"
    assert calib.mode == "zone"

    # Key 'n': Create new zone -> zone_3
    calib.handle_key(ord("n"))
    assert "zone_3" in calib.zones
    assert calib.active_target == "zone_3"

    # Add points to zone_3 via mouse click simulation
    calib.on_mouse(1, 100, 100, 0, None)  # EVENT_LBUTTONDOWN
    calib.on_mouse(1, 200, 100, 0, None)
    calib.on_mouse(1, 200, 200, 0, None)
    assert len(calib.zones["zone_3"]) == 3

    # Key 'u': Undo last point
    calib.handle_key(ord("u"))
    assert len(calib.zones["zone_3"]) == 2

    # Key 'c': Clear active zone points
    calib.handle_key(ord("c"))
    assert len(calib.zones["zone_3"]) == 0

    # Key 'l': Toggle Tripwire Mode
    calib.handle_key(ord("l"))
    assert calib.mode == "line"
    assert calib.active_target == "line_1"
    assert "line_1" in calib.lines

    # Simulate 2-click input for tripwire P1 and P2
    calib.on_mouse(1, 50, 50, 0, None)    # P1
    assert calib.lines["line_1"]["p1"] == [50, 50]
    assert not calib.lines["line_1"]["p2"]

    calib.on_mouse(1, 250, 250, 0, None)  # P2
    assert calib.lines["line_1"]["p2"] == [250, 250]

    # Key 'd': Cycle line direction (both -> A_to_B -> B_to_A -> both)
    assert calib.lines["line_1"]["direction"] == "both"
    calib.handle_key(ord("d"))
    assert calib.lines["line_1"]["direction"] == "A_to_B"
    calib.handle_key(ord("d"))
    assert calib.lines["line_1"]["direction"] == "B_to_A"
    calib.handle_key(ord("d"))
    assert calib.lines["line_1"]["direction"] == "both"

    # Key '2': Switch to line_2 (since mode is 'line')
    calib.handle_key(ord("2"))
    assert calib.active_target == "line_2"
    assert "line_2" in calib.lines

    # Key 'l': Toggle back to polygon mode
    calib.handle_key(ord("l"))
    assert calib.mode == "zone"

    # Key '1': Switch back to zone_1
    calib.handle_key(ord("1"))
    assert calib.active_target == "zone_1"

    print("  [PASS] test_calibrator_state_machine")


def test_structured_save_and_backward_compatibility():
    """Verify structured saving and 100% backward compatibility of load_roi_data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "roi_zones.json"

        # 1. Test saving structured JSON via ROICalibrator
        calib = ROICalibrator(camera_id="cam_test", interactive=False)
        calib.roi_path = tmp_path
        calib.base_w = 1920
        calib.base_h = 1080
        calib.zones = {
            "zone_1": [[100, 100], [500, 100], [500, 500], [100, 500]],
            "zone_2": [[600, 100], [900, 100], [900, 400], [600, 400]],
            "zone_3": [[1000, 200], [1200, 200], [1200, 600]],
        }
        calib.lines = {
            "line_1": {"name": "Perimeter Virtual", "p1": [200, 600], "p2": [800, 600], "direction": "A_to_B"},
        }
        calib.save_zones()

        assert tmp_path.exists(), "Saved ROI file should exist"

        # Read back raw json to verify format
        with open(tmp_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        assert "zones" in raw_data, "Structured format must contain 'zones'"
        assert "lines" in raw_data, "Structured format must contain 'lines'"
        assert raw_data["base_resolution"] == [1920, 1080]

        # 2. Test load_roi_data parser on new structured format
        base_res, zones, lines = load_roi_data(tmp_path)
        assert base_res == (1920, 1080)
        assert len(zones) == 3
        assert "zone_1" in zones
        assert "zone_3" in zones
        assert zones["zone_1"] == [[100, 100], [500, 100], [500, 500], [100, 500]]
        assert len(lines) == 1
        assert lines["line_1"]["direction"] == "A_to_B"

        # 3. Test load_roi_data parser on legacy flat dictionary format
        legacy_flat = {
            "base_resolution": [640, 480],
            "zone_1_koridor": [[10, 10], [50, 10], [50, 50], [10, 50]],
            "zone_2_transit": [[100, 100], [200, 100], [200, 200], [100, 200]],
        }
        legacy_path = Path(tmpdir) / "legacy_roi.json"
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump(legacy_flat, f, indent=2)

        leg_res, leg_zones, leg_lines = load_roi_data(legacy_path)
        assert leg_res == (640, 480)
        assert len(leg_zones) == 2
        assert "zone_1_koridor" in leg_zones
        assert "zone_2_transit" in leg_zones
        assert leg_zones["zone_1_koridor"] == [[10, 10], [50, 10], [50, 50], [10, 50]]
        assert len(leg_lines) == 0, "Legacy flat format should parse with 0 lines without crash"

        # 4. Test ZoneFilter with structured format
        zf = ZoneFilter(
            zones=zones,
            lines=lines,
            frame_shape=(360, 640),
            base_resolution=base_res,
        )
        assert len(zf.scaled_zones) == 3
        assert len(zf.scaled_lines) == 1
        # Check affine scaling: x scaled by 640/1920 = 1/3
        scaled_p1 = zf.scaled_lines["line_1"]["p1"]
        expected_x = int(round(200 * 640 / 1920.0))
        assert scaled_p1[0] == expected_x

    print("  [PASS] test_structured_save_and_backward_compatibility")


def test_calibrator_rendering():
    """Verify ROICalibrator canvas rendering with 20% transparent fill and tripwires."""
    calib = ROICalibrator(camera_id="cam_render_test", interactive=False)
    dummy_frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)

    calib.zones["zone_1"] = [[100, 100], [600, 100], [600, 600], [100, 600]]
    calib.lines["line_1"]["p1"] = [200, 800]
    calib.lines["line_1"]["p2"] = [1000, 800]
    calib.lines["line_1"]["direction"] = "both"

    rendered = calib.render(dummy_frame)
    assert rendered is not None
    assert rendered.shape == (1080, 1920, 3)
    # Check that pixels along polygon perimeter edge have changed color (crisp outline)
    assert not np.array_equal(rendered[100, 300], dummy_frame[100, 300]), "Perimeter outline should be rendered"
    # Check that interior floor pixel remains unoccluded (zero floor fill clutter)
    assert np.array_equal(rendered[300, 300], dummy_frame[300, 300]), "Interior floor should have zero fill clutter"
    # Check that pixels along the tripwire line have changed color
    assert not np.array_equal(rendered[800, 500], dummy_frame[800, 500]), "Tripwire line should be rendered"

    print("  [PASS] test_calibrator_rendering")


def run_all_tests():
    print("\n" + "=" * 60)
    print("RUNNING MULTI-ROI CALIBRATOR & BACKWARD COMPATIBILITY TESTS")
    print("=" * 60)
    test_calibrator_state_machine()
    test_structured_save_and_backward_compatibility()
    test_calibrator_rendering()
    print("=" * 60)
    print("ALL MULTI-ROI CALIBRATOR TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
