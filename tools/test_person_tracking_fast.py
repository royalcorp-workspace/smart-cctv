import sys
import time
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.tracker import CentroidTracker, TrackedObject


class TestFastMovingPersonTracking(unittest.TestCase):
    """Test tracker resilience when persons move fast (80-120px step) in foreground."""

    def setUp(self):
        self.tracker = CentroidTracker(
            max_disappeared_sec=3.0,
            max_distance_px=60.0,
            max_age_frames=150,
            ema_alpha=0.3,
        )

    def test_foreground_fast_passer_maintains_track_id(self):
        """A person walking quickly across the foreground (y=280) jumping 90px between frames
        should retain the same track ID via adaptive distance scaling.
        """
        now = time.time()
        # Frame 1: Person enters foreground
        det_f1 = [((100, 240, 60, 100), (130, 290), "outside_zone", 6000.0, "person", -200.0, 0.85, "yolo")]
        active_1, _ = self.tracker.update(det_f1, timestamp=now)
        self.assertEqual(len(active_1), 1)
        initial_track_id = active_1[0].track_id

        # Frame 2: Person steps forward fast (dx = +85px, dy = +10px)
        now += 0.08
        det_f2 = [((185, 250, 65, 100), (215, 300), "outside_zone", 6500.0, "person", -200.0, 0.82, "yolo")]
        active_2, _ = self.tracker.update(det_f2, timestamp=now)
        self.assertEqual(len(active_2), 1)
        self.assertEqual(active_2[0].track_id, initial_track_id, "Track ID must remain identical for fast foreground movement")

        # Frame 3: Another fast step (dx = +90px, dy = +5px)
        now += 0.08
        det_f3 = [((275, 255, 65, 100), (305, 305), "outside_zone", 6500.0, "person", -200.0, 0.80, "yolo")]
        active_3, _ = self.tracker.update(det_f3, timestamp=now)
        self.assertEqual(len(active_3), 1)
        self.assertEqual(active_3[0].track_id, initial_track_id, "Track ID must be preserved across multi-frame fast motion")

    def test_full_frame_outside_zone_tracking(self):
        """Persons far outside any defined ROI zones should be registered and tracked."""
        now = time.time()
        # Person in deep center room far away from zones (dist_to_nearest = -350px)
        det = [((300, 150, 50, 90), (325, 195), "outside_zone", 4500.0, "person", -350.0, 0.75, "yolo")]
        active, _ = self.tracker.update(det, timestamp=now)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].class_label, "person")
        self.assertEqual(active[0].zone_id, "outside_zone")


class TestFaceEvalPriority(unittest.TestCase):
    """Test prioritization queue for face evaluation."""

    def test_new_moving_passerby_prioritized_over_seated(self):
        """A moving passerby with no face seen must rank higher than an existing seated person."""
        # Seated person: established, seen face, stationary
        seated_trk = TrackedObject(
            track_id=1,
            centroid=(200, 100),
            anchor_centroid=(200, 100),
            anchor_bbox=(180, 70, 40, 70),
            bbox=(180, 70, 40, 70),
            zone_id="zone_1_koridor",
            contour_area=2800.0,
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
            edge_distance=10.0,
            confidence=0.88,
            initial_centroid=(200, 100),
            max_displacement_from_start=5.0,  # barely moved
            frame_count=200,
        )

        # Passerby: newly entered, moving fast, no face
        passer_trk = TrackedObject(
            track_id=2,
            centroid=(350, 280),
            anchor_centroid=(300, 270),
            anchor_bbox=(320, 230, 60, 110),
            bbox=(320, 230, 60, 110),
            zone_id="outside_zone",
            contour_area=6600.0,
            first_seen=24.0,
            last_seen=25.0,
            stationary_start=24.0,
            dwell_duration=1.0,
            is_stationary=False,
            is_triggered=False,
            alert_sent=False,
            is_attended=False,
            is_active_this_frame=True,
            class_label="person",
            db_event_id=None,
            missed_frames=0,
            edge_distance=-100.0,
            confidence=0.78,
            initial_centroid=(250, 250),
            max_displacement_from_start=110.0,  # moving fast
            frame_count=12,
        )

        person_face_recog = {
            1: {"has_seen_face": True, "label": "Alghany", "last_eval_time": 23.0},
            2: {"has_seen_face": False, "last_eval_time": 0.0},
        }

        def _eval_priority(trk_obj):
            p_c = person_face_recog.get(trk_obj.track_id)
            has_face = p_c and p_c.get("has_seen_face", False)
            is_moving = getattr(trk_obj, "max_displacement_from_start", 0.0) > 20.0
            box_sz = float(trk_obj.bbox[2] * trk_obj.bbox[3])
            if not has_face and is_moving:
                return 20000.0 + box_sz
            elif not has_face:
                return 10000.0 + box_sz
            elif is_moving:
                return 5000.0
            else:
                return 1.0

        candidates = [seated_trk, passer_trk]
        candidates.sort(key=_eval_priority, reverse=True)

        self.assertEqual(candidates[0].track_id, 2, "Passerby must be evaluated first")
        self.assertEqual(candidates[1].track_id, 1, "Seated track should yield priority to new passerby")


if __name__ == "__main__":
    unittest.main(verbosity=2)
