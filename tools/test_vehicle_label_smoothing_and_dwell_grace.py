import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.tracker import CentroidTracker, TrackedObject


class TestVehicleLabelSmoothingAndDwellGrace(unittest.TestCase):
    def setUp(self):
        self.zone_cfgs = {
            "zone_1_koridor": {"name": "Zebra Cross", "dwell_threshold_sec": 60.0},
            "zone_2_transit": {"name": "Parking Transit", "dwell_threshold_sec": 1800.0},
        }
        self.tracker = CentroidTracker(
            max_distance_px=100.0,
            anchor_radius_px=40.0,
            flicker_tolerance_sec=2.0,
            max_disappeared_sec=12.0,
            zone_configs=self.zone_cfgs,
        )

    def test_label_smoothing_voting(self):
        """Test that vehicle label does not flicker when confidence fluctuates between truck and bus."""
        t = 1000.0
        # Register initial track as truck (conf=0.75)
        det_initial = ((100, 100, 200, 150), (200, 175), "zone_2_transit", 30000.0, "truck", 20.0, 0.75, "yolo")
        active, _ = self.tracker.update([det_initial], timestamp=t)
        self.assertEqual(len(active), 1)
        trk = active[0]
        self.assertEqual(trk.class_label, "truck")

        # Feed consecutive frames where YOLO fluctuates with noisy bus/car detections:
        # Frame 1: bus (0.45)
        # Frame 2: truck (0.70)
        # Frame 3: bus (0.50)
        # Frame 4: car (0.40)
        # Frame 5: truck (0.80)
        detections_stream = [
            ("bus", 0.45),
            ("truck", 0.70),
            ("bus", 0.50),
            ("car", 0.40),
            ("truck", 0.80),
            ("bus", 0.48),
            ("truck", 0.65),
        ]

        for lbl, conf in detections_stream:
            t += 0.5
            det = ((100, 100, 200, 150), (200, 175), "zone_2_transit", 30000.0, lbl, 20.0, conf, "yolo")
            active, _ = self.tracker.update([det], timestamp=t)
            trk = active[0]
            # Must remain locked as "truck" despite occasional "bus" or "car" noise
            self.assertEqual(trk.class_label, "truck", f"Flickered to {trk.class_label} on noise label {lbl}")

        print("PASS: Vehicle label smoothing successfully resisted bus/car flickering noise.")

    def test_sticky_stationary_and_jitter_grace_period(self):
        """Test that dwell timer continues accumulating during coordinate jitter and brief spikes."""
        t = 2000.0
        # Start stationary in zone_2_transit
        det_pos0 = ((100, 100, 200, 150), (200, 175), "zone_2_transit", 30000.0, "truck", 20.0, 0.85, "yolo")
        for _ in range(6):
            t += 1.0
            active, _ = self.tracker.update([det_pos0], timestamp=t)
            trk = active[0]

        self.assertTrue(trk.is_stationary, "Vehicle should be stationary after 6 stable frames.")
        self.assertGreater(trk.dwell_duration, 0.0, "Dwell duration must be accumulating.")
        dwell_before_jitter = trk.dwell_duration

        # 1. Normal CCTV bbox jitter: centroid shifts by ~18 px (e.g. from 200,175 to 215, 185)
        t += 1.0
        det_jitter1 = ((115, 110, 200, 150), (215, 185), "zone_2_transit", 30000.0, "truck", 20.0, 0.85, "yolo")
        active, _ = self.tracker.update([det_jitter1], timestamp=t)
        trk = active[0]
        self.assertTrue(trk.is_stationary, "Vehicle must stay stationary under 18px normal jitter.")
        self.assertGreater(trk.dwell_duration, dwell_before_jitter, "Dwell time must NOT reset on 18px jitter.")

        # 2. Brief coordinate spike (e.g. 28px shift) for 3 frames (grace period)
        prev_dwell = trk.dwell_duration
        for _ in range(3):
            t += 1.0
            det_spike = ((125, 120, 200, 150), (225, 195), "zone_2_transit", 30000.0, "truck", 20.0, 0.85, "yolo")
            active, _ = self.tracker.update([det_spike], timestamp=t)
            trk = active[0]
            self.assertTrue(trk.is_stationary, "Grace period must protect stationary status during brief 3-frame spike.")
            self.assertGreater(trk.dwell_duration, prev_dwell, "Dwell time must NOT reset during grace period.")
            prev_dwell = trk.dwell_duration

        # 3. Coordinate settles back to parking spot
        t += 1.0
        active, _ = self.tracker.update([det_pos0], timestamp=t)
        trk = active[0]
        self.assertTrue(trk.is_stationary)
        self.assertEqual(trk.moved_confirmation_frames, 0, "Counter must reset when still.")
        self.assertGreater(trk.dwell_duration, prev_dwell, "Dwell continues uninterrupted.")
        print(f"PASS: Sticky stationary & grace period verified. Preserved dwell={trk.dwell_duration:.1f}s.")

    def test_sustained_movement_resets_dwell(self):
        """Test that when vehicle genuinely drives away for >= 30 frames, dwell resets to 0."""
        t = 3000.0
        # Initialize stationary vehicle
        det_pos0 = ((100, 100, 200, 150), (200, 175), "zone_2_transit", 30000.0, "truck", 20.0, 0.85, "yolo")
        for _ in range(6):
            t += 1.0
            active, _ = self.tracker.update([det_pos0], timestamp=t)

        trk = active[0]
        self.assertTrue(trk.is_stationary)
        self.assertGreater(trk.dwell_duration, 0.0)

        # Vehicle drives continuously away (>40px frame shift each frame) for 35 frames
        cx, cy = 200, 175
        for i in range(35):
            t += 0.2
            cx += 25
            cy += 15
            det_moving = ((cx - 100, cy - 75, 200, 150), (cx, cy), "zone_2_transit", 30000.0, "truck", 20.0, 0.85, "yolo")
            active, _ = self.tracker.update([det_moving], timestamp=t)
            trk = active[0]

        self.assertFalse(trk.is_stationary, "Vehicle must no longer be stationary after driving away.")
        self.assertEqual(trk.dwell_duration, 0.0, "Dwell duration must reset to 0.0 after sustained departure.")
        print("PASS: Genuine departure correctly confirmed moved and reset dwell time.")


if __name__ == "__main__":
    unittest.main()
