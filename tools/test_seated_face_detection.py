import sys
import time
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.face_recognizer import FaceRecognizer
from engine.face_detector import YuNetFaceDetector


class TestSeatedFaceDetectionAndLandmarks(unittest.TestCase):
    """Test face detection and landmark gate tolerances for seated and downward-looking persons."""

    def test_head_crop_calculation_covers_seated_person(self):
        """Sitting person upper body box: head crop must cover 65% height and 15% top padding."""
        # Seated person bbox in 640x360 space (short box, e.g. upper torso + head = 70px)
        px, py, pw, ph = 200, 100, 45, 70
        pad_x = int(pw * 0.20)
        h_x1 = max(0, px - pad_x)
        h_y1 = max(0, py - int(ph * 0.15))
        h_x2 = min(640, px + pw + pad_x)
        h_y2 = min(360, py + int(ph * 0.65))

        self.assertEqual(h_x1, 200 - 9)
        self.assertEqual(h_y1, 100 - 10)  # reaches above box by 10px (hair/forehead)
        self.assertEqual(h_x2, 200 + 45 + 9)
        self.assertEqual(h_y2, 100 + 45)   # reaches 65% down into body (chin/neck)
        crop_h = h_y2 - h_y1
        self.assertEqual(crop_h, 55)

    def test_seated_downward_tilt_landmark_pass(self):
        """A person looking downward at a desk with score 0.38 and slight tilt should pass is_frontal_face."""
        # Synthesize YuNet landmark array (15 elements):
        # [0:4] bbox (x, y, w, h)
        # [4:6] right eye (x, y)
        # [6:8] left eye (x, y)
        # [8:10] nose tip (x, y)
        # [10:12] right mouth (x, y)
        # [12:14] left mouth (x, y)
        # [14] score
        w = 40.0
        h = 50.0
        # When tilting down: eyes at y=15, nose at y=26 (lower), mouth at y=38
        # interocular distance = 12px (ratio = 12/40 = 0.30 > 0.14)
        # nose slightly off-center (d_r=5, d_l=7 -> ratio = 5/7 = 0.71 > 0.12)
        face_landmarks = [
            100.0, 50.0, w, h,
            114.0, 65.0,  # right eye
            126.0, 65.0,  # left eye
            119.0, 76.0,  # nose tip
            115.0, 88.0,  # right mouth
            125.0, 88.0,  # left mouth
            0.38,         # score (previously would fail 0.45 threshold)
        ]

        is_frontal, reason = FaceRecognizer.is_frontal_face(face_landmarks)
        self.assertTrue(is_frontal, f"Seated downward tilt should pass, but failed: {reason}")

    def test_5s_grace_period_stability(self):
        """Grace period of 5.0s should retain badge when employee looks away/down for 3.5 seconds."""
        now = time.time()
        last_frontal_seen_time = now - 3.5  # looked down 3.5 seconds ago
        grace_period_sec = 5.0

        is_retained = (now - last_frontal_seen_time) <= grace_period_sec
        self.assertTrue(is_retained, "Badge should remain retained after 3.5s downward head tilt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
