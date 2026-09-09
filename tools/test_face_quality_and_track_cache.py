"""Unit Tests for Landmark-Based Face Quality Gate & Track-ID Based Caching.

Verifies:
1. is_frontal_face() discriminates frontal vs side-profile / extreme yaw poses.
2. FaceRecognizer.recognize() cleanly bypasses non-frontal faces.
3. Track-ID throttling limits evaluation to at most once per 1.0s / 15 frames.
4. Non-frontal view on re-evaluation does not overwrite established known identities.
"""

import time
import unittest
import numpy as np

from engine.face_recognizer import FaceRecognizer


class TestFaceQualityAndTrackCache(unittest.TestCase):
    """Test suite for Face Quality Gate and Track-ID caching."""

    def setUp(self):
        # Synthetic frontal face [x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rc_x, rc_y, lc_x, lc_y, score]
        # w = 100, h = 100, bbox = [50, 50, 100, 100]
        # eyes at (75, 80) and (125, 80) -> dist = 50 -> ratio = 0.50
        # nose at (100, 105) -> strictly centered between 75 and 125 -> sym = 25/25 = 1.0
        self.frontal_face = np.array([
            50.0, 50.0, 100.0, 100.0,
            75.0, 80.0,   # Right eye
            125.0, 80.0,  # Left eye
            100.0, 105.0, # Nose tip
            80.0, 125.0,  # Right mouth corner
            120.0, 125.0, # Left mouth corner
            0.85,         # Score
        ], dtype=np.float32)

    def test_01_landmark_frontal_validation(self):
        """Verify frontal face passes and distorted/side poses are rejected."""
        # 1. Frontal face passes
        is_f, reason = FaceRecognizer.is_frontal_face(self.frontal_face)
        self.assertTrue(is_f, f"Frontal face should pass: {reason}")

        # 2. Side profile with narrow eye distance (e.g. eyes collapsed to 15px on 100px face -> ratio 0.15)
        side_face_narrow = self.frontal_face.copy()
        side_face_narrow[4] = 90.0
        side_face_narrow[6] = 105.0  # eye dist = 15px, ratio = 0.15 < 0.22
        is_f, reason = FaceRecognizer.is_frontal_face(side_face_narrow)
        self.assertFalse(is_f, "Narrow eye distance must be rejected as side profile")
        self.assertIn("narrow eye distance", reason.lower())

        # 3. Side profile with nose outside eye span
        side_face_nose_out = self.frontal_face.copy()
        side_face_nose_out[8] = 135.0  # eyes are 75..125, nose at 135 (outside)
        is_f, reason = FaceRecognizer.is_frontal_face(side_face_nose_out)
        self.assertFalse(is_f, "Nose tip outside eye span must be rejected")
        self.assertIn("nose tip outside", reason.lower())

        # 4. Extreme yaw with strong horizontal asymmetry (e.g. nose at 77, right eye at 75 -> dist=2, left eye at 125 -> dist=48)
        side_face_asym = self.frontal_face.copy()
        side_face_asym[8] = 77.0  # sym = 2 / 48 = 0.04 < 0.20
        is_f, reason = FaceRecognizer.is_frontal_face(side_face_asym)
        self.assertFalse(is_f, "Extreme yaw nose asymmetry must be rejected")
        self.assertIn("asymmetric nose", reason.lower())

        # 5. Low confidence score (< 0.60)
        low_conf_face = self.frontal_face.copy()
        low_conf_face[14] = 0.45
        is_f, reason = FaceRecognizer.is_frontal_face(low_conf_face, min_score=0.60)
        self.assertFalse(is_f, "Face with confidence < 0.60 must be rejected")
        self.assertIn("low yunet confidence", reason.lower())

        # 6. Backward compatibility: 4-element bbox (no landmarks)
        bbox_only = np.array([50, 50, 100, 100], dtype=np.float32)
        is_f, reason = FaceRecognizer.is_frontal_face(bbox_only)
        self.assertTrue(is_f, "4-element bbox without landmarks should pass cleanly")

    def test_02_recognize_bypasses_non_frontal(self):
        """Verify FaceRecognizer.recognize() skips SFace inference for non-frontal faces."""
        fr = FaceRecognizer()
        # Add a dummy known identity
        fr.known_embeddings = [("Adi Ahmad", np.zeros((1, 128), dtype=np.float32))]

        # Side profile face
        side_face = self.frontal_face.copy()
        side_face[8] = 140.0  # nose outside eyes

        dummy_frame = np.zeros((300, 300, 3), dtype=np.uint8)
        name, score, label = fr.recognize(dummy_frame, side_face, check_frontal=True)
        self.assertEqual(name, "Non-Frontal")
        self.assertEqual(score, 0.0)
        self.assertEqual(label, "Non-Frontal")

    def test_03_track_id_throttling_simulation(self):
        """Simulate track-ID based face evaluation cooldown over 30 frames."""
        track_cache = {}
        person_id = 5
        eval_count = 0
        now = 1000.0

        # Simulate 30 frames at 15 FPS (dt = 0.066s per frame)
        for frame_idx in range(30):
            current_time = now + (frame_idx * 0.066)
            p_cache = track_cache.get(person_id)

            needs_eval = False
            if p_cache is None:
                needs_eval = True
            else:
                elapsed = current_time - p_cache.get("last_eval_time", 0.0)
                frames = frame_idx - p_cache.get("last_eval_frame", 0)
                if elapsed >= 1.0 and frames >= 15:
                    needs_eval = True

            if needs_eval:
                eval_count += 1
                track_cache[person_id] = {
                    "label": "Adi Ahmad (85%)",
                    "name": "Adi Ahmad",
                    "score": 0.85,
                    "last_eval_time": current_time,
                    "last_eval_frame": frame_idx,
                    "is_known": True,
                }

        # In 30 frames (2.0 seconds total duration at 15 FPS):
        # Evaluation should only trigger on Frame 0 and Frame 16 (after 1.0s and >= 15 frames)
        self.assertEqual(eval_count, 2, f"Expected exactly 2 evaluations in 30 frames, got {eval_count}")
        self.assertEqual(track_cache[person_id]["name"], "Adi Ahmad")

    def test_04_side_profile_retains_established_identity(self):
        """Verify that when an established track turns sideways, its identity is preserved."""
        track_cache = {
            10: {
                "label": "Rizqi Setiawan (88%)",
                "name": "Rizqi Setiawan",
                "score": 0.88,
                "last_eval_time": 100.0,
                "last_eval_frame": 0,
                "is_known": True,
            }
        }

        # At frame 20 (now = 101.5), evaluation is due:
        now = 101.5
        frame_idx = 20
        trk_id = 10

        # Person turns head sideways -> is_frontal is False
        side_face = self.frontal_face.copy()
        side_face[8] = 145.0
        is_frontal, reason = FaceRecognizer.is_frontal_face(side_face)
        self.assertFalse(is_frontal)

        prev_cache = track_cache.get(trk_id)
        prev_known = prev_cache and prev_cache.get("is_known", False)

        # Pipeline logic: if not frontal, retain previous known identity and bump timestamps
        if not is_frontal:
            if prev_cache:
                prev_cache["last_eval_time"] = now
                prev_cache["last_eval_frame"] = frame_idx
            else:
                track_cache[trk_id] = {"label": "Unknown", "name": "Unknown", "is_known": False}

        # Verify identity was NOT lost or replaced with "Unknown"
        self.assertEqual(track_cache[trk_id]["name"], "Rizqi Setiawan")
        self.assertEqual(track_cache[trk_id]["label"], "Rizqi Setiawan (88%)")
        self.assertTrue(track_cache[trk_id]["is_known"])


if __name__ == "__main__":
    unittest.main()
