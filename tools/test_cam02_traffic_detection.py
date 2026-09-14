"""Automated Unit and Synthetic Test for cam_02 Vehicle & Pedestrian Detection Pipeline."""

import sys
from pathlib import Path
import unittest
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engine.tracker import CentroidTracker, TrackedObject
from notification.local_alert import VisualHUD
from web.buffer import MultiCameraBuffer


class TestCam02TrafficPipeline(unittest.TestCase):
    """Test suite for vehicle and pedestrian detection & tracking on cam_02."""

    def test_tracker_cross_class_isolation(self):
        """Verify that CentroidTracker registers and separates vehicle, pedestrian, and bag tracks."""
        tracker = CentroidTracker(max_distance_px=100.0)

        # Detections: 1 truck, 1 car, 1 person, 1 bag
        # Format: (bbox, centroid, zone_id, area, label, edge_dist, conf, source)
        detections_frame1 = [
            ((100, 100, 120, 80), (160, 140), "zone_1_koridor", 9600, "truck", 30.0, 0.88, "yolo"),
            ((300, 150, 90, 60), (345, 180), "zone_1_koridor", 5400, "car", 25.0, 0.82, "yolo"),
            ((500, 200, 30, 80), (515, 240), "zone_1_koridor", 2400, "person", 40.0, 0.79, "yolo"),
            ((250, 280, 20, 20), (260, 290), "zone_1_koridor", 400, "tas", 20.0, 0.90, "yolo"),
        ]

        active_1, purged_1 = tracker.update(detections_frame1, timestamp=1000.0)
        self.assertEqual(len(active_1), 4)

        tracks_by_label = {t.class_label: t for t in active_1}
        self.assertIn("truck", tracks_by_label)
        self.assertIn("car", tracks_by_label)
        self.assertIn("person", tracks_by_label)
        self.assertIn("tas", tracks_by_label)

        truck_id = tracks_by_label["truck"].track_id
        car_id = tracks_by_label["car"].track_id
        person_id = tracks_by_label["person"].track_id

        # Frame 2: Objects move slightly
        detections_frame2 = [
            ((110, 102, 120, 80), (170, 142), "zone_1_koridor", 9600, "truck", 30.0, 0.89, "yolo"),
            ((310, 152, 90, 60), (355, 182), "zone_1_koridor", 5400, "car", 25.0, 0.84, "yolo"),
            ((505, 202, 30, 80), (520, 242), "zone_1_koridor", 2400, "person", 40.0, 0.81, "yolo"),
        ]

        active_2, purged_2 = tracker.update(detections_frame2, timestamp=1000.1)
        tracks_2_by_label = {t.class_label: t for t in active_2}

        # Verify identity preservation across frames
        self.assertEqual(tracks_2_by_label["truck"].track_id, truck_id)
        self.assertEqual(tracks_2_by_label["car"].track_id, car_id)
        self.assertEqual(tracks_2_by_label["person"].track_id, person_id)

        # Cross-class isolation check: Person moving near truck position should NOT steal truck ID
        detections_swapped = [
            ((175, 145, 30, 80), (190, 185), "zone_1_koridor", 2400, "person", 30.0, 0.80, "yolo"),
        ]
        active_3, _ = tracker.update(detections_swapped, timestamp=1000.2)
        person_active = [t for t in active_3 if t.class_label == "person" and t.is_active_this_frame]
        self.assertTrue(len(person_active) > 0)
        # Person must NOT have truck_id
        self.assertNotEqual(person_active[0].track_id, truck_id)

    def test_visual_hud_cam02_rendering(self):
        """Verify that VisualHUD renders distinct vehicle and pedestrian badges and TRAFFIC status on cam_02."""
        hud = VisualHUD()

        canvas = np.zeros((360, 640, 3), dtype=np.uint8)

        # Create dummy tracked objects
        truck_obj = TrackedObject(
            track_id=1,
            centroid=(160, 140),
            anchor_centroid=(160, 140),
            bbox=(100, 100, 120, 80),
            zone_id="zone_1_koridor",
            contour_area=9600.0,
            first_seen=1000.0,
            last_seen=1000.5,
            stationary_start=1000.0,
            class_label="truck",
            is_active_this_frame=True,
        )

        car_obj = TrackedObject(
            track_id=2,
            centroid=(345, 180),
            anchor_centroid=(345, 180),
            bbox=(300, 150, 90, 60),
            zone_id="zone_1_koridor",
            contour_area=5400.0,
            first_seen=1000.0,
            last_seen=1000.5,
            stationary_start=1000.0,
            class_label="car",
            is_active_this_frame=True,
        )

        person_obj = TrackedObject(
            track_id=3,
            centroid=(515, 240),
            anchor_centroid=(515, 240),
            bbox=(500, 200, 30, 80),
            zone_id="zone_1_koridor",
            contour_area=2400.0,
            first_seen=1000.0,
            last_seen=1000.5,
            stationary_start=1000.0,
            class_label="person",
            is_active_this_frame=True,
        )

        zones = {
            "zone_1_koridor": [(19, 817), (233, 698), (1909, 748), (1909, 1062)]
        }

        rendered = hud.render(
            canvas=canvas,
            zones=zones,
            tracked_objects=[truck_obj, car_obj, person_obj],
            camera_name="Area Parkir POS-2",
            camera_id="cam_02",
            is_connected=True,
            fps=12.5,
        )

        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.shape, (360, 640, 3))
        # Canvas should not be entirely black anymore (drawings present)
        self.assertGreater(np.count_nonzero(rendered), 500)

    def test_multicamera_buffer_traffic_telemetry(self):
        """Verify that MultiCameraBuffer telemetry accurately records traffic counts for cam_02."""
        buf = MultiCameraBuffer.get_instance()
        buf.register_camera("cam_02", "Area Parkir POS-2")

        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        telemetry = {
            "fps": 12.8,
            "online": True,
            "is_connected": True,
            "rtsp_status": "Connected",
            "violations": 0,
            "clear_area_count": 0,
            "active_tracks": 3,
            "identified_faces": [],
        }

        buf.update_frame(
            camera_id="cam_02",
            frame=dummy_frame,
            telemetry=telemetry,
            camera_name="Area Parkir POS-2",
        )

        read_telemetry = buf.get_telemetry("cam_02")
        self.assertIsNotNone(read_telemetry)
        self.assertEqual(read_telemetry.get("active_tracks"), 3)
        self.assertEqual(read_telemetry.get("camera_id"), "cam_02")


if __name__ == "__main__":
    unittest.main()
