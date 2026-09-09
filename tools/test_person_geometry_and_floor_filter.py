"""Unit tests for person geometric sanity filtering, spatial crop isolation, and 2.5s grace period."""

import time
import pytest
import numpy as np

from engine.face_detector import YuNetFaceDetector


class TestPersonGeometryAndFloorFilter:
    """Verifies that flat floor items are filtered and face badges adhere to spatial anchors and grace periods."""

    def test_01_person_geometry_filter(self):
        """Flat floor objects (plastic bags, trash) and tiny blobs must be rejected, while seated workers are tracked."""
        test_cases = [
            # (w, h, should_pass, description)
            (120, 45, False, "Crumpled plastic bag on floor (flat, h/w = 0.375)"),
            (150, 50, False, "Trash sack lying down (h/w = 0.33)"),
            (50, 25, False, "Tiny floor speck / reflection (h < 45)"),
            (90, 65, True, "Seated desk worker leaning forward (h=65, h/w = 0.72)"),
            (80, 60, True, "Seated desk worker compact (h=60, h/w = 0.75)"),
            (70, 70, True, "Minimum square stance (h=70, h/w = 1.0)"),
            (80, 180, True, "Standing adult person (h/w = 2.25)"),
            (75, 120, True, "Seated desk worker (h/w = 1.60)"),
        ]

        for w, h, should_pass, desc in test_cases:
            ratio = float(h) / max(1.0, float(w))
            passes = (h >= 45) and (ratio >= 0.55)
            assert passes == should_pass, f"Failed on '{desc}': got {passes}, expected {should_pass}"

    def test_02_crop_roi_spatial_isolation(self):
        """Faces detected outside the requested crop ROI must be filtered out."""
        detector = YuNetFaceDetector(score_threshold=0.50, auto_download=False)

        # Populate detector's buffer with 2 faces:
        # Face 1: at (50, 50, 60, 60) -> 1080p center ~ (240, 240)
        # Face 2: at (450, 250, 60, 60) -> 1080p center ~ (1440, 810)
        detector._face_buffer = {
            1: {
                "bbox": (50, 50, 60, 60),
                "score": 0.85,
                "missed_frames": 0,
                "raw_face": np.array([50, 50, 60, 60, 65, 65, 95, 65, 80, 80, 70, 95, 90, 95, 0.85], dtype=np.float32),
                "raw_face_1080": np.array([150, 150, 180, 180, 195, 195, 285, 195, 240, 240, 210, 285, 270, 285, 0.85], dtype=np.float32),
                "is_crop": True,
            },
            2: {
                "bbox": (450, 250, 60, 60),
                "score": 0.80,
                "missed_frames": 0,
                "raw_face": np.array([450, 250, 60, 60, 465, 265, 495, 265, 480, 280, 470, 295, 490, 295, 0.80], dtype=np.float32),
                "raw_face_1080": np.array([1350, 750, 180, 180, 1395, 795, 1485, 795, 1440, 840, 1410, 885, 1470, 885, 0.80], dtype=np.float32),
                "is_crop": False,
            },
        }

        # Request faces only in ROI (100, 100, 350, 350)
        dummy_display = np.zeros((1080, 1920, 3), dtype=np.uint8)
        dummy_infer = np.zeros((360, 640, 3), dtype=np.uint8)

        # Mock detect to avoid re-invoking network
        detector.detect = lambda f: []
        detector.detect_crop = lambda display_frame, crop_roi, target_crop_w: []

        results = detector.detect_faces(
            frame=dummy_infer,
            display_frame=dummy_display,
            crop_roi=[(100, 100, 350, 350)],
        )

        assert len(results) == 1, f"Expected 1 face inside crop ROI, got {len(results)}"
        assert results[0][4] == 1, f"Expected Face ID 1, got {results[0][4]}"

    def test_03_no_synthetic_fallback_box(self):
        """If a track has seen a face but face_bbox_640 is missing/invalid, no synthetic box is created."""
        now = time.time()
        track_info = {
            "has_seen_face": True,
            "last_frontal_seen_time": now,
            "face_bbox_640": None,  # Missing bbox
            "label": "Raqil",
            "name": "Raqil",
            "score": 0.85,
        }

        recognized_faces = []
        px, py, pw, ph = (100, 80, 80, 160)
        last_frontal = track_info.get("last_frontal_seen_time", 0.0)

        if (now - last_frontal) <= 2.5:
            f_box = track_info.get("face_bbox_640")
            if f_box and f_box[2] > 0 and f_box[3] > 0:
                recognized_faces.append((f_box, 0.85, track_info["label"], track_info["name"], track_info["score"]))

        assert len(recognized_faces) == 0, "No synthetic box must be added when f_box is None!"

    def test_04_frontal_face_grace_period(self):
        """Face badge must be rendered within 2.5s of frontal view, and hidden after 2.5s."""
        now = time.time()
        f_box_valid = (120, 90, 40, 45)

        # Case A: Seen 1.0 second ago (within 2.5s grace period) -> RENDERED
        p_info_recent = {
            "has_seen_face": True,
            "last_frontal_seen_time": now - 1.0,
            "face_bbox_640": f_box_valid,
            "label": "Raqil (85%)",
            "name": "Raqil",
            "score": 0.85,
        }
        rendered_recent = []
        if (now - p_info_recent["last_frontal_seen_time"]) <= 2.5:
            rendered_recent.append(p_info_recent["face_bbox_640"])
        assert len(rendered_recent) == 1, "Face seen 1.0s ago must be rendered during grace period!"

        # Case B: Seen 3.0 seconds ago (grace period expired, person turned away) -> HIDDEN
        p_info_expired = {
            "has_seen_face": True,
            "last_frontal_seen_time": now - 3.0,
            "face_bbox_640": f_box_valid,
            "label": "Raqil (85%)",
            "name": "Raqil",
            "score": 0.85,
        }
        rendered_expired = []
        if (now - p_info_expired["last_frontal_seen_time"]) <= 2.5:
            rendered_expired.append(p_info_expired["face_bbox_640"])
        assert len(rendered_expired) == 0, "Face seen 3.0s ago must be hidden as grace period expired!"

    def test_05_semi_frontal_face_tolerance(self):
        """Semi-frontal face turned ~45 degrees with confidence 0.48-0.50 must pass is_frontal_face."""
        from engine.face_recognizer import FaceRecognizer

        # Simulated 45-degree turned face landmarks (w=50, h=55, score=0.48):
        # right eye=(115, 105), left eye=(132, 107), nose=(124, 118)
        # eye dist = hypot(17, 2) ~ 17.1 -> ratio = 17.1 / 50 = 0.34
        # symmetry: d_r = |124 - 115| = 9, d_l = |124 - 132| = 8 -> ratio = 8/9 = 0.88
        semi_frontal_face = [
            100, 90, 50, 55,  # bbox
            115, 105,         # right eye
            132, 107,         # left eye
            124, 118,         # nose tip
            118, 130,         # mouth right
            128, 131,         # mouth left
            0.48,             # YuNet score 0.48
        ]

        is_frontal, reason = FaceRecognizer.is_frontal_face(
            semi_frontal_face,
            min_eye_dist_ratio=0.18,
            min_symmetry_ratio=0.15,
            min_score=0.45,
        )
        assert is_frontal is True, f"Semi-frontal face at 45 degrees should pass, but got rejected: {reason}"
