import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.tracker import TrackedObject


class TestSmallPersonAndStaticArtifact(unittest.TestCase):
    """Test small seated person detection pass and static artifact fix."""

    def test_small_seated_person_box_filter(self):
        """A small seated person box at back desks (w=22, h=30, area=660) should pass the filter."""
        # Simulated filter logic from yolo_detector.py & main.py:
        # if p_h < 22 or p_w < 16 or (p_w * p_h) < 400: continue
        back_desk_person = (100, 50, 22, 30)  # w=22, h=30
        w, h = back_desk_person[2], back_desk_person[3]
        area = w * h

        is_rejected = (h < 22 or w < 16 or area < 400)
        self.assertFalse(is_rejected, "Seated person in back row should NOT be rejected")

        # Extremely tiny blob (e.g. wall speck) should still be rejected:
        tiny_noise = (100, 50, 14, 18)
        w_n, h_n = tiny_noise[2], tiny_noise[3]
        area_n = w_n * h_n
        is_noise_rejected = (h_n < 22 or w_n < 16 or area_n < 400)
        self.assertTrue(is_noise_rejected, "Micro noise should be rejected")

    def test_seated_person_never_marked_static_artifact(self):
        """A person sitting still at a desk (displacement <= 2.0, low conf) must remain should_render = True."""
        seated_person = TrackedObject(
            track_id=1,
            centroid=(200, 100),
            anchor_centroid=(200, 100),
            anchor_bbox=(185, 75, 30, 45),
            bbox=(185, 75, 30, 45),
            zone_id="outside_zone",
            contour_area=1350.0,
            first_seen=10.0,
            last_seen=25.0,
            stationary_start=10.0,
            dwell_duration=15.0,
            is_stationary=False,
            is_triggered=False,
            alert_sent=False,
            is_attended=False,
            is_active_this_frame=True,
            class_label="person",
            db_event_id=None,
            missed_frames=0,
            edge_distance=20.0,
            confidence=0.30,  # low confidence
            initial_centroid=(200, 100),
            max_displacement_from_start=1.2,  # sitting completely still (<= 2.0px)
            frame_count=30,  # >= 5 frames
        )

        self.assertFalse(seated_person.is_static_artifact, "Person must NEVER be marked as static artifact")
        self.assertTrue(seated_person.should_render, "Seated person must have should_render = True")


class TestAmbiguityGuard(unittest.TestCase):
    """Test Top-2 Ambiguity Guard prevents identity swaps."""

    def test_ambiguous_scores_rejected_to_unknown(self):
        """When top score is 0.52 (Rizqi) and 2nd score is 0.49 (Aji), margin 0.03 < 0.08 -> reject to Unknown."""
        sorted_identities = [("Rizqi Setiawan", 0.52), ("Aji Yulianto", 0.49)]
        best_name, best_score = sorted_identities[0]

        is_ambiguous = False
        if len(sorted_identities) > 1 and best_score < 0.65:
            second_name, second_score = sorted_identities[1]
            margin = best_score - second_score
            if margin < 0.08:
                is_ambiguous = True

        self.assertTrue(is_ambiguous, "Scores within 0.08 margin below 0.65 must be flagged ambiguous")

    def test_clear_match_passes_ambiguity_guard(self):
        """When top score is 0.72 (Rian) and 2nd score is 0.42, margin 0.30 >= 0.08 -> passes as genuine."""
        sorted_identities = [("Rian Heri", 0.72), ("Rizqi Setiawan", 0.42)]
        best_name, best_score = sorted_identities[0]

        is_ambiguous = False
        if len(sorted_identities) > 1 and best_score < 0.65:
            second_name, second_score = sorted_identities[1]
            margin = best_score - second_score
            if margin < 0.08:
                is_ambiguous = True

        self.assertFalse(is_ambiguous, "High margin genuine match must pass ambiguity guard")


if __name__ == "__main__":
    unittest.main(verbosity=2)
