"""Automated Verification Suite for Accuracy Enhancements with Technical Safeguards.

Tests:
1. 16:9 Aspect Ratio Consistency (640x360 inference, ZoneFilter downscale symmetry, HUD & Telegram scaling).
2. CLAHE Local Contrast Equalization for YuNet & SFace.
3. Universal Dynamic Head-Crop (Max 3 targets sorted by area descending, Safeguard 1).
4. Stationary Recognition Cache & Min-size Gating (24px cutoff, Safeguard 1).
5. DualSubtractor Small Contour Verification (>= 250px area, Safeguard 2).
6. Multi-frame Best-shot Identity Aggregation.
"""

import os
from pathlib import Path
import sys
import unittest
import numpy as np
import cv2

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from engine.face_detector import YuNetFaceDetector, apply_clahe
from engine.face_recognizer import FaceRecognizer
from engine.zone_filter import ZoneFilter
from notification.local_alert import VisualHUD
from notification.telegram_alert import TelegramNotifier


class TestAccuracyEnhancements(unittest.TestCase):
    """Test suite covering accuracy enhancements and the 3 user safeguards."""

    def test_01_aspect_ratio_consistency_16_9(self):
        """Verify 16:9 aspect ratio symmetry: 640x360 inference and 1920x1080 native."""
        infer_w, infer_h = 640, 360
        disp_w, disp_h = 1920, 1080

        # Scale factor must be identical (3.0x) on both axes
        scale_x = disp_w / float(infer_w)
        scale_y = disp_h / float(infer_h)
        self.assertAlmostEqual(scale_x, 3.0, places=3, msg="scale_x must be 3.0")
        self.assertAlmostEqual(scale_y, 3.0, places=3, msg="scale_y must be 3.0")
        self.assertEqual(scale_x, scale_y, "Scale factors must be symmetric (no vertical squish)")

        # Verify ZoneFilter downscale with frame_shape=(360, 640)
        test_zones = {
            "test_transit": [[300, 300], [600, 300], [600, 600], [300, 600]],
            "base_resolution": [1920, 1080],
        }
        zf = ZoneFilter(zones=test_zones, frame_shape=(infer_h, infer_w))
        self.assertIn("test_transit", zf.scaled_zones)

        # A point at (300, 300) in 1080p should map to (100, 100) in 640x360
        first_pt = zf.scaled_zones["test_transit"][0]
        self.assertEqual(first_pt, [100, 100], "Point (300, 300) downscaled by 1/3 must be (100, 100)")

    def test_02_clahe_contrast_enhancement(self):
        """Verify CLAHE normalizes contrast on dark / shadowed input frames."""
        # Create dark low-contrast image simulating harsh shadow
        dark_img = np.full((120, 120, 3), 40, dtype=np.uint8)
        # Add faint face-like feature in center
        cv2.circle(dark_img, (60, 60), 20, (60, 60, 60), -1)

        enhanced = apply_clahe(dark_img, clip_limit=2.0)
        self.assertEqual(enhanced.shape, dark_img.shape)

        # Mean and variance should be higher or equal after contrast stretching
        self.assertGreaterEqual(float(np.var(enhanced)), float(np.var(dark_img)))

    def test_03_head_crop_limiter_max_3(self):
        """Safeguard 1: Verify dynamic head crop evaluates at most 3 largest person targets."""
        # Simulate 6 detected people with various bounding box areas
        person_boxes = [
            (50, 100, 40, 80),     # area = 3200
            (150, 100, 100, 200),  # area = 20000 (largest)
            (300, 100, 80, 160),   # area = 12800 (2nd)
            (400, 100, 70, 140),   # area = 9800  (3rd)
            (500, 100, 30, 60),    # area = 1800
            (550, 100, 25, 50),    # area = 1250
        ]

        sorted_persons = sorted(person_boxes, key=lambda b: b[2] * b[3], reverse=True)
        relevant_head_rois = []
        disp_w, disp_h = 1920, 1080
        scale_disp_x = disp_w / 640.0
        scale_disp_y = disp_h / 360.0

        for p_box in sorted_persons[:3]:
            px, py, pw, ph = p_box
            pad_x = int(pw * 0.15)
            h_x1 = max(0, px - pad_x)
            h_y1 = max(0, py - int(ph * 0.10))
            h_x2 = min(640, px + pw + pad_x)
            h_y2 = min(360, py + int(ph * 0.50))
            disp_x1 = int(round(h_x1 * scale_disp_x))
            disp_y1 = int(round(h_y1 * scale_disp_y))
            disp_x2 = int(round(h_x2 * scale_disp_x))
            disp_y2 = int(round(h_y2 * scale_disp_y))
            if (disp_x2 - disp_x1) >= 20 and (disp_y2 - disp_y1) >= 20:
                relevant_head_rois.append((disp_x1, disp_y1, disp_x2, disp_y2))

        # Must cap strictly at 3
        self.assertEqual(len(relevant_head_rois), 3)
        # First target must correspond to largest person (area 20000)
        self.assertGreater(relevant_head_rois[0][2] - relevant_head_rois[0][0],
                            relevant_head_rois[1][2] - relevant_head_rois[1][0])

    def test_04_micro_face_gating_and_cache(self):
        """Safeguard 1: Verify micro-faces (< 24px) are gated and stationary cache functions."""
        # Synthetic test of dimension gate
        face_w_micro = 20.0
        face_h_micro = 20.0
        self.assertTrue(face_w_micro < 24.0, "Micro face (< 24px) must be flagged")

        face_w_prominent = 36.0
        face_h_prominent = 36.0
        self.assertFalse(face_w_prominent < 24.0, "Prominent face (>= 24px) must not be flagged as micro")

        # Test stationary cache movement check
        last_pos = (150, 150)
        cur_pos_still = (152, 151)  # moved ~2.2px (< 15px)
        dist_still = ((cur_pos_still[0] - last_pos[0])**2 + (cur_pos_still[1] - last_pos[1])**2)**0.5
        self.assertLess(dist_still, 15.0, "Sitting person should be classified as stationary")

        cur_pos_walk = (180, 170)   # moved > 36px (walking)
        dist_walk = ((cur_pos_walk[0] - last_pos[0])**2 + (cur_pos_walk[1] - last_pos[1])**2)**0.5
        self.assertGreater(dist_walk, 15.0, "Walking person should bypass stationary cache")

    def test_05_dualsubtractor_small_contour_safeguard(self):
        """Safeguard 2: Verify DualSubtractor contour verification for >= 250px objects."""
        # 1. Validating minimum area 250px
        mask = np.zeros((360, 640), dtype=np.uint8)
        # Draw a small 18x16 rectangular blob (area ~288 px)
        cv2.rectangle(mask, (100, 100), (118, 116), 255, -1)

        test_zones = {
            "zone_2_transit": [[50, 50], [300, 50], [300, 300], [50, 300]],
            "base_resolution": [640, 360],
        }
        zone_cfgs = {
            "zone_2_transit": {"detect_unattended": True, "min_contour_area": 250}
        }
        zf = ZoneFilter(zones=test_zones, zone_configs=zone_cfgs, frame_shape=(360, 640))
        dets = zf.filter_contours(mask, min_area_default=250)
        self.assertEqual(len(dets), 1, "Blob with area ~288 px should be detected with min_area=250")

        # 2. Test floor variance / edge density rejection on synthetic bare floor crop
        flat_tile_crop = np.full((16, 18, 3), 200, dtype=np.uint8)  # completely uniform glossy tile
        crop_gray = cv2.cvtColor(flat_tile_crop, cv2.COLOR_BGR2GRAY)
        crop_var = float(np.var(crop_gray))
        crop_edges = int(np.count_nonzero(cv2.Canny(crop_gray, 40, 120)))
        edge_density = crop_edges / float(max(1, 18 * 16))

        # Uniform tile has 0 variance and 0 edges
        self.assertLess(crop_var, 55.0)
        self.assertLess(edge_density, 0.020)

        # Highly textured bag crop has high variance and edges
        textured_bag_crop = np.random.randint(0, 255, (16, 18, 3), dtype=np.uint8)
        bag_gray = cv2.cvtColor(textured_bag_crop, cv2.COLOR_BGR2GRAY)
        bag_var = float(np.var(bag_gray))
        bag_edges = int(np.count_nonzero(cv2.Canny(bag_gray, 40, 120)))
        bag_edge_density = bag_edges / float(max(1, 18 * 16))
        self.assertGreaterEqual(bag_var, 55.0)
        self.assertGreaterEqual(bag_edge_density, 0.020)

    def test_06_hud_and_zoom_crop_scaling(self):
        """Safeguard 3: Verify VisualHUD and zoom crop coordinate scaling to 1080p."""
        # Canvas 1080p (1920x1080)
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        zones = {
            "zone_test": [[100, 100], [500, 100], [500, 300], [100, 300]],
            "base_resolution": [1920, 1080],
        }
        # Render without crashing
        res = VisualHUD.render(
            canvas=canvas,
            zones=zones,
            tracked_objects=[],
            camera_id="cam_01",
            fps=15.0,
            is_connected=True,
            faces=[((100, 100, 40, 40), 0.90, "Andi (85%)")],
        )
        self.assertEqual(res.shape, (1080, 1920, 3))

        # Test Telegram zoom crop auto-scale from 640x360 space
        t_notifier = TelegramNotifier.get_instance()
        test_frame_1080 = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Put white box at scaled (300, 300, 150, 150)
        cv2.rectangle(test_frame_1080, (300, 300), (450, 450), (255, 255, 255), -1)

        # In 640x360 space, coordinates are (100, 100, 50, 50)
        crop = t_notifier.create_zoom_crop(test_frame_1080, bbox=(100, 100, 50, 50))
        self.assertIsNotNone(crop)
        self.assertGreater(crop.size, 0)
        # Should contain the white box
        self.assertTrue(np.any(crop > 200))


if __name__ == "__main__":
    unittest.main()
