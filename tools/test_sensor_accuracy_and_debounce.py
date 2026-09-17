"""Unit test suite verifying:
1. Sensor-grade zero margin (margin_px=0.0) ground contact evaluation.
2. 3-frame temporal debounce confirmation filter in CentroidTracker.
3. Glitch suppression against 1-2 frame boundary flutter / transient shadows.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.zone_filter import ZoneFilter
from engine.tracker import CentroidTracker, TrackedObject


def test_strict_zero_margin_contact():
    """Verify that margin_px=0.0 rejects points outside the zone even by 1-2 pixels."""
    # Polygon zone: x in [100, 300], y in [100, 300]
    zones = {
        "zone_sterile": [[100, 100], [300, 100], [300, 300], [100, 300]]
    }
    zf = ZoneFilter(zones=zones, frame_shape=(360, 640), base_resolution=(640, 360))

    # Test point 2px outside the right boundary
    p_outside = (302, 200)
    # With margin_px=8.0 it would be accepted:
    z_lenient, _ = zf.find_zone_and_distance(p_outside, margin_px=8.0)
    assert z_lenient == "zone_sterile", "Lenient margin should have matched"

    # With margin_px=0.0 it MUST be rejected:
    z_strict, _ = zf.find_zone_and_distance(p_outside, margin_px=0.0)
    assert z_strict is None, f"Strict margin 0.0 must reject point outside boundary, got: {z_strict}"

    # Test point 2px inside the boundary
    p_inside = (298, 200)
    z_inside, _ = zf.find_zone_and_distance(p_inside, margin_px=0.0)
    assert z_inside == "zone_sterile", "Point inside boundary must match zone_sterile"

    print("PASS: test_strict_zero_margin_contact passed successfully!")


def test_three_frame_debounce_confirmation():
    """Verify that CentroidTracker requires 3 consecutive frames to confirm zone transition."""
    tracker = CentroidTracker(
        max_distance_px=100.0,
        anchor_radius_px=40.0,
        flicker_tolerance_sec=2.0,
        max_disappeared_sec=12.0,
    )

    t = 1000.0
    # Detection tuple: (bbox, centroid, zone_id, area, label, edge_dist, conf, source)
    # Initial frame: object appears in "outside_zone"
    det_outside = ((50, 50, 40, 60), (70, 80), "outside_zone", 2400.0, "car", -10.0, 0.9, "yolo")
    active, _ = tracker.update([det_outside], timestamp=t)
    assert len(active) == 1
    assert active[0].zone_id == "outside_zone"

    # Frame 1: Object moves into "zone_restricted"
    t += 0.1
    det_inside = ((150, 150, 40, 60), (170, 180), "zone_restricted", 2400.0, "car", 10.0, 0.9, "yolo")
    active, _ = tracker.update([det_inside], timestamp=t)
    # Debounce frame 1: should hold old zone_id
    assert active[0].zone_id == "outside_zone", "Frame 1 should hold previous zone_id"
    assert active[0].pending_zone_id == "zone_restricted"
    assert active[0].zone_confirmation_frames == 1

    # Frame 2: Still in "zone_restricted" (debounce frame 2)
    t += 0.1
    active, _ = tracker.update([det_inside], timestamp=t)
    assert active[0].zone_id == "outside_zone", "Frame 2 should hold previous zone_id"
    assert active[0].pending_zone_id == "zone_restricted"
    assert active[0].zone_confirmation_frames == 2

    # Frame 3: Still in "zone_restricted" -> Confirmed!
    t += 0.1
    active, _ = tracker.update([det_inside], timestamp=t)
    assert active[0].zone_id == "zone_restricted", "Frame 3 must confirm and update zone_id"
    assert active[0].pending_zone_id is None
    assert active[0].zone_confirmation_frames == 0

    print("PASS: test_three_frame_debounce_confirmation passed successfully!")


def test_transient_glitch_suppression():
    """Verify that a 1-frame transient detection does not trigger a zone change."""
    tracker = CentroidTracker(
        max_distance_px=100.0,
        anchor_radius_px=40.0,
        flicker_tolerance_sec=2.0,
        max_disappeared_sec=12.0,
    )

    t = 2000.0
    # Object starts in outside_zone
    det_outside = ((50, 50, 40, 60), (70, 80), "outside_zone", 2400.0, "person", -10.0, 0.9, "yolo")
    active, _ = tracker.update([det_outside], timestamp=t)
    assert active[0].zone_id == "outside_zone"

    # Frame 1: Glitch frame into zone_a (e.g. shadow or temporary flutter)
    t += 0.1
    det_glitch = ((52, 50, 40, 60), (72, 80), "zone_a", 2400.0, "person", 5.0, 0.9, "yolo")
    active, _ = tracker.update([det_glitch], timestamp=t)
    assert active[0].zone_id == "outside_zone"
    assert active[0].pending_zone_id == "zone_a"

    # Frame 2: Immediately reverts back outside
    t += 0.1
    active, _ = tracker.update([det_outside], timestamp=t)
    assert active[0].zone_id == "outside_zone"
    assert active[0].pending_zone_id is None
    assert active[0].zone_confirmation_frames == 0

    print("PASS: test_transient_glitch_suppression passed successfully!")


if __name__ == "__main__":
    test_strict_zero_margin_contact()
    test_three_frame_debounce_confirmation()
    test_transient_glitch_suppression()
    print("\nALL SENSOR ACCURACY AND DEBOUNCE TESTS PASSED!")
