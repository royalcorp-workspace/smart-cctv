"""Unit test suite for 2D Line Crossing Math, Intersection, and TripwireEngine."""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.line_crossing import (
    cross_product_2d,
    segments_intersect,
    TripwireEngine,
    TripwireLine,
    LineCrossingEvent,
)


class MockTrack:
    """Mock TrackedObject for headless unit testing."""
    def __init__(self, track_id: int, centroid: tuple, bbox: tuple, class_label: str = "person", confidence: float = 0.85, frame_count: int = 5):
        self.track_id = track_id
        self.centroid = centroid
        self.bbox = bbox
        self.class_label = class_label
        self.confidence = confidence
        self.frame_count = frame_count
        self.missed_frames = 0
        self.is_active_this_frame = True


def test_cross_product_orientation():
    """Verify 2D cross product sign orientation."""
    a = (0.0, 0.0)
    b = (10.0, 0.0)

    # Point above line AB -> left side -> positive
    c_left = (5.0, 5.0)
    cp_left = cross_product_2d(a, b, c_left)
    assert cp_left > 0, f"Expected positive cross product for left point, got {cp_left}"

    # Point below line AB -> right side -> negative
    c_right = (5.0, -5.0)
    cp_right = cross_product_2d(a, b, c_right)
    assert cp_right < 0, f"Expected negative cross product for right point, got {cp_right}"

    # Collinear point on line AB -> zero
    c_collinear = (5.0, 0.0)
    cp_collinear = cross_product_2d(a, b, c_collinear)
    assert abs(cp_collinear) < 1e-6, f"Expected zero for collinear point, got {cp_collinear}"
    print("  [PASS] test_cross_product_orientation")


def test_segments_intersect():
    """Verify 2D line segment intersection logic."""
    # Intersecting cross: (0, 5) to (10, 5) and (5, 0) to (5, 10)
    assert segments_intersect((0, 5), (10, 5), (5, 0), (5, 10)) is True

    # Parallel lines: (0, 0) to (10, 0) and (0, 5) to (10, 5)
    assert segments_intersect((0, 0), (10, 0), (0, 5), (10, 5)) is False

    # Disjoint segments (would intersect if infinite, but segments do not meet)
    assert segments_intersect((0, 0), (4, 4), (5, 5), (10, 10)) is False

    # Collinear overlapping segments: (0, 0) to (5, 0) and (3, 0) to (8, 0)
    assert segments_intersect((0, 0), (5, 0), (3, 0), (8, 0)) is True

    # Segments sharing an endpoint: (0, 0) to (5, 5) and (5, 5) to (10, 0)
    assert segments_intersect((0, 0), (5, 5), (5, 5), (10, 0)) is True

    # T-junction: segment touches the middle of another segment
    assert segments_intersect((0, 5), (10, 5), (5, 5), (5, 10)) is True

    # Zero-length point segment on the line
    assert segments_intersect((5, 5), (5, 5), (0, 5), (10, 5)) is True

    # Zero-length point segment off the line
    assert segments_intersect((5, 6), (5, 6), (0, 5), (10, 5)) is False
    print("  [PASS] test_segments_intersect")


def test_tripwire_engine_crossing_and_direction():
    """Verify TripwireEngine direction determination and event emission."""
    # Virtual line from (100, 200) to (500, 200)
    lines_cfg = {
        "line_1": {
            "name": "Gate Line",
            "p1": [100, 200],
            "p2": [500, 200],
            "direction": "both",
            "target_classes": ["person", "car"],
        }
    }
    engine = TripwireEngine(lines=lines_cfg, cooldown_sec=5.0, min_track_frames=1, max_step_px=150.0)

    # Frame 1: Person at (300, 150) -> foot_y = 150 + 40 - 2 = 188 (Above line)
    t0 = 1000.0
    track1 = MockTrack(track_id=1, centroid=(300, 170), bbox=(280, 150, 40, 40), class_label="person")
    events = engine.update([track1], timestamp=t0)
    assert len(events) == 0, "First frame should not emit events (needs 2 points for trajectory)"

    # Frame 2: Person moves to (300, 220) -> foot_y = 220 + 40 - 2 = 258 (Crossed line from Side B (top) to Side A (bottom))
    t1 = 1000.1
    track1.bbox = (280, 220, 40, 40)
    track1.centroid = (300, 240)
    events = engine.update([track1], timestamp=t1)
    assert len(events) == 1, f"Expected 1 crossing event, got {len(events)}"
    assert events[0].line_id == "line_1"
    assert events[0].line_name == "Gate Line"
    assert events[0].direction == "B_to_A"

    # Frame 3: Anti-flapping test - crossing again within cooldown
    # Move upward across line at t2 = 1002.0 (< 5.0s from t1)
    t2 = 1002.0
    track1.bbox = (280, 150, 40, 40)
    events = engine.update([track1], timestamp=t2)
    assert len(events) == 0, "Event within cooldown should be suppressed by anti-flapping"

    # Frame 4: After cooldown expires (> 5.0s from t1=1000.1)
    # Track is currently at (280, 150) (Side B). Move to (280, 250) (Side A) at t3 = 1006.0
    t3 = 1006.0
    track1.bbox = (280, 250, 40, 40)
    events = engine.update([track1], timestamp=t3)
    assert len(events) == 1, f"Expected 1 event after cooldown expiry, got {len(events)}"
    assert events[0].direction == "B_to_A"
    print("  [PASS] test_tripwire_engine_crossing_and_direction")


def test_tripwire_directional_filter():
    """Verify that lines with directional constraints only trigger for matching direction."""
    # Line only accepts B_to_A (top to bottom)
    lines_cfg = {
        "line_restricted": {
            "name": "One Way",
            "p1": [100, 200],
            "p2": [500, 200],
            "direction": "B_to_A",
            "target_classes": ["car"],
        }
    }
    engine = TripwireEngine(lines=lines_cfg, cooldown_sec=1.0, min_track_frames=1, max_step_px=250.0)

    # Move vehicle from A to B (bottom to top: y=300 -> y=100)
    v_track = MockTrack(track_id=10, centroid=(300, 300), bbox=(250, 250, 100, 100), class_label="car")
    engine.update([v_track], timestamp=10.0)

    v_track.centroid = (300, 100)
    v_track.bbox = (250, 50, 100, 100)
    events = engine.update([v_track], timestamp=10.2)
    assert len(events) == 0, "A_to_B crossing should be rejected by B_to_A filter"

    # Now move vehicle from B to A (top to bottom: y=100 -> y=300)
    t_after_cooldown = 12.0
    v_track2 = MockTrack(track_id=11, centroid=(300, 100), bbox=(250, 50, 100, 100), class_label="car")
    engine.update([v_track2], timestamp=t_after_cooldown)

    v_track2.centroid = (300, 300)
    v_track2.bbox = (250, 250, 100, 100)
    events2 = engine.update([v_track2], timestamp=t_after_cooldown + 0.2)
    assert len(events2) == 1, "B_to_A crossing should be accepted"
    assert events2[0].direction == "B_to_A"
    print("  [PASS] test_tripwire_directional_filter")


def test_visual_flash_timer():
    """Verify 2.0s pulse/flash tracking for visual HUD."""
    lines_cfg = {
        "line_flash": {
            "name": "Flash Line",
            "p1": [50, 50],
            "p2": [250, 50],
            "direction": "both",
        }
    }
    engine = TripwireEngine(lines=lines_cfg, cooldown_sec=1.0, min_track_frames=1, max_step_px=150.0)
    track = MockTrack(track_id=5, centroid=(100, 20), bbox=(80, 0, 40, 40), class_label="person")
    engine.update([track], timestamp=100.0)

    track.bbox = (80, 80, 40, 40)
    engine.update([track], timestamp=100.1)

    # Immediately after crossing
    assert engine.get_recent_flash("line_flash", now=100.2) is True
    assert "line_flash" in engine.get_flashing_lines(now=100.2)

    # At 1.9s after crossing
    assert engine.get_recent_flash("line_flash", now=101.9) is True

    # At 2.1s after crossing (> 2.0s flash window)
    assert engine.get_recent_flash("line_flash", now=102.2) is False
    assert "line_flash" not in engine.get_flashing_lines(now=102.2)
    print("  [PASS] test_visual_flash_timer")


def run_all_tests():
    print("\n" + "=" * 60)
    print("RUNNING LINE CROSSING & TRIPWIRE MATH VERIFICATION SUITE")
    print("=" * 60)
    test_cross_product_orientation()
    test_segments_intersect()
    test_tripwire_engine_crossing_and_direction()
    test_tripwire_directional_filter()
    test_visual_flash_timer()
    print("=" * 60)
    print("ALL LINE CROSSING UNIT TESTS PASSED SUCCESSFULLY! [OK]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
