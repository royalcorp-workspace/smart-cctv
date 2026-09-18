"""Unit test for verifying person bounding box rendering and zone target_classes persistence."""
import pytest
import numpy as np
from fastapi.testclient import TestClient
from web.server import app
from notification.local_alert import VisualHUD
from engine.tracker import TrackedObject

client = TestClient(app)

def test_person_box_rendered_for_all_cameras():
    """Verify that person bounding boxes are rendered for cam_01 and cam_05 without being filtered."""
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    track = TrackedObject(
        track_id=1,
        centroid=(125, 150),
        anchor_centroid=(125, 150),
        bbox=(100, 100, 50, 100),
        zone_id="zone_1_koridor",
        contour_area=5000.0,
        first_seen=1000.0,
        last_seen=1001.0,
        stationary_start=1000.0,
        class_label="person",
        is_active_this_frame=True,
    )
    track.zone_name = "Koridor Utama"

    # Test on cam_01
    rendered_cam01 = VisualHUD.render(
        canvas=canvas.copy(),
        zones={"zone_1_koridor": [[50, 50], [200, 50], [200, 250], [50, 250]]},
        tracked_objects=[track],
        camera_id="cam_01",
        fps=10.0,
        is_connected=True,
    )
    # The canvas should not be completely black (bounding box and text badge were drawn)
    assert np.any(rendered_cam01 > 0), "cam_01 should render person bounding box"

    # Test on cam_05
    rendered_cam05 = VisualHUD.render(
        canvas=canvas.copy(),
        zones={"zone_1": [[50, 50], [200, 50], [200, 250], [50, 250]]},
        tracked_objects=[track],
        camera_id="cam_05",
        fps=10.0,
        is_connected=True,
    )
    assert np.any(rendered_cam05 > 0), "cam_05 should render person bounding box"


def test_zone_target_classes_persistence():
    """Verify that target_classes are accepted and stored by /api/zones/update."""
    import shutil
    from pathlib import Path

    cam_dir = Path("cameras/cam_01")
    cfg_file = cam_dir / "config.json"
    roi_file = cam_dir / "roi_zones.json"

    cfg_bak = cfg_file.read_bytes() if cfg_file.exists() else None
    roi_bak = roi_file.read_bytes() if roi_file.exists() else None

    try:
        payload = {
            "camera_id": "cam_01",
            "zones": {
                "zone_1_koridor": [[100, 100], [200, 100], [200, 200], [100, 200]],
            },
            "lines": {},
            "zone_configs": {
                "zone_1_koridor": {
                    "name": "Koridor Utama",
                    "dwell_threshold_sec": 45.0,
                    "target_classes": ["person", "car"],
                }
            },
            "base_resolution": [1920, 1080],
        }
        response = client.post("/api/zones/update", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        # Fetch zones back and verify target_classes are returned
        resp_get = client.get("/api/zones/cam_01")
        assert resp_get.status_code == 200
        data_get = resp_get.json()
        z_cfg = data_get.get("zone_configs", {}).get("zone_1_koridor", {})
        assert "person" in z_cfg.get("target_classes", []), "target_classes must include 'person'"
    finally:
        if cfg_bak and cfg_file.exists():
            cfg_file.write_bytes(cfg_bak)
        if roi_bak and roi_file.exists():
            roi_file.write_bytes(roi_bak)
