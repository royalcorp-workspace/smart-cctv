"""Unit and Integration Test Suite: Dynamic Camera Management and Calibration.

Tests:
1. Safe .env handling and dynamic camera addition via /api/cameras/add.
2. Unified ROI zones & tripwires retrieval via /api/zones/{camera_id}.
3. Unified ROI zones & tripwires update via /api/zones/update with config dwell update.
4. Camera deactivation and archiving via /api/cameras/delete.
5. RTSP connection testing endpoint via /api/cameras/test_rtsp.
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient

# Ensure root workspace is on python path
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from web.server import app
from web.buffer import MultiCameraBuffer


class TestDynamicCameraAndCalibration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.test_cam_id = "cam_test_99"
        self.cam_dir = _ROOT_DIR / "cameras" / self.test_cam_id
        self.env_path = _ROOT_DIR / ".env"
        self.env_bak_path = _ROOT_DIR / ".env.testbak"

        # Backup .env if exists
        if self.env_path.exists():
            shutil.copyfile(self.env_path, self.env_bak_path)

    def tearDown(self):
        # Cleanup created camera dir if still exists
        if self.cam_dir.exists():
            shutil.rmtree(self.cam_dir, ignore_errors=True)

        # Cleanup archived dirs for this test
        for p in (_ROOT_DIR / "cameras").glob(f".archived_{self.test_cam_id}_*"):
            shutil.rmtree(p, ignore_errors=True)

        # Unregister from buffer
        MultiCameraBuffer.get_instance().unregister_camera(self.test_cam_id)

        # Restore original .env
        if self.env_bak_path.exists():
            shutil.copyfile(self.env_bak_path, self.env_path)
            self.env_bak_path.unlink()

    def test_01_add_camera_safely(self):
        """Test adding a dynamic camera: creates folder, writes .env, and registers into buffer."""
        payload = {
            "name": "Kamera Test Onboarding",
            "camera_id": self.test_cam_id,
            "ip": "192.168.1.200",
            "port": 554,
            "user": "admin",
            "pass": "Secret#123",
            "channel": 102,
        }

        resp = self.client.post("/api/cameras/add", json=payload)
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["camera_id"], self.test_cam_id)

        # Verify folder structure
        self.assertTrue(self.cam_dir.exists())
        self.assertTrue((self.cam_dir / "config.json").exists())
        self.assertTrue((self.cam_dir / "roi_zones.json").exists())
        self.assertTrue((self.cam_dir / "snapshots").exists())

        # Verify config.json content
        with open(self.cam_dir / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.assertEqual(cfg["camera_id"], self.test_cam_id)
        self.assertEqual(cfg["name"], "Kamera Test Onboarding")
        self.assertIn("CAM_TEST_99_IP", cfg["source"])

        # Verify .env appended safely
        with open(self.env_path, "r", encoding="utf-8") as f:
            env_text = f.read()
        self.assertIn("CAM_TEST_99_IP=192.168.1.200", env_text)
        self.assertIn('CAM_TEST_99_PASS="Secret#123"', env_text)

        # Verify MultiCameraBuffer registration
        buffer = MultiCameraBuffer.get_instance()
        cameras = [c["id"] for c in buffer.get_cameras()]
        self.assertIn(self.test_cam_id, cameras)

    def test_02_get_and_update_zones_unified(self):
        """Test unified ROI zones and tripwires GET and POST endpoints."""
        # 1. Create camera first
        add_payload = {
            "name": "Kamera Test Kalibrasi",
            "camera_id": self.test_cam_id,
            "ip": "192.168.1.201",
            "port": 554,
            "user": "user",
            "pass": "pass",
            "channel": 102,
        }
        self.client.post("/api/cameras/add", json=add_payload)

        # 2. Get initial zones
        get_resp = self.client.get(f"/api/zones/{self.test_cam_id}")
        self.assertEqual(get_resp.status_code, 200)
        get_data = get_resp.json()
        self.assertIn("zones", get_data)
        self.assertIn("lines", get_data)
        self.assertIn("zone_configs", get_data)

        # 3. Update zones & tripwires
        update_payload = {
            "camera_id": self.test_cam_id,
            "base_resolution": [1920, 1080],
            "zones": {
                "zone_1_koridor": [
                    [100, 100],
                    [500, 100],
                    [500, 400],
                    [100, 400]
                ]
            },
            "lines": {
                "line_1": {
                    "name": "Tripwire Pintu",
                    "p1": [200, 300],
                    "p2": [400, 300],
                    "direction": "a_to_b",
                    "target_classes": ["person"]
                }
            },
            "zone_configs": {
                "zone_1_koridor": {
                    "name": "Koridor Akses",
                    "dwell_threshold_sec": 45.0,
                    "detect_unattended": False
                }
            }
        }

        up_resp = self.client.post("/api/zones/update", json=update_payload)
        self.assertEqual(up_resp.status_code, 200, up_resp.text)
        up_data = up_resp.json()
        self.assertEqual(up_data["status"], "success")
        self.assertIn("zone_1_koridor", up_data["zones"])
        self.assertIn("line_1", up_data["lines"])

        # 4. Verify disk persistence in roi_zones.json
        with open(self.cam_dir / "roi_zones.json", "r", encoding="utf-8") as f:
            disk_roi = json.load(f)
        self.assertIn("zone_1_koridor", disk_roi["zones"])
        self.assertIn("line_1", disk_roi["lines"])
        self.assertEqual(disk_roi["lines"]["line_1"]["direction"], "a_to_b")

        # 5. Verify config.json dwell updated
        with open(self.cam_dir / "config.json", "r", encoding="utf-8") as f:
            disk_cfg = json.load(f)
        self.assertEqual(disk_cfg["zones"]["zone_1_koridor"]["dwell_threshold_sec"], 45.0)

    def test_03_delete_camera_safe_archival(self):
        """Test camera deletion: archives directory and unregisters from buffer."""
        # 1. Create camera
        add_payload = {
            "name": "Kamera Hapus",
            "camera_id": self.test_cam_id,
            "ip": "192.168.1.202",
            "port": 554,
            "user": "user",
            "pass": "pass",
            "channel": 102,
        }
        self.client.post("/api/cameras/add", json=add_payload)
        self.assertTrue(self.cam_dir.exists())

        # 2. Delete camera
        del_resp = self.client.post("/api/cameras/delete", json={"camera_id": self.test_cam_id})
        self.assertEqual(del_resp.status_code, 200)

        # 3. Verify original directory no longer exists in cameras/
        self.assertFalse(self.cam_dir.exists())

        # 4. Verify archived directory created
        archived = list((_ROOT_DIR / "cameras").glob(f".archived_{self.test_cam_id}_*"))
        self.assertTrue(len(archived) > 0)

        # 5. Verify removed from MultiCameraBuffer
        buffer = MultiCameraBuffer.get_instance()
        cameras = [c["id"] for c in buffer.get_cameras()]
        self.assertNotIn(self.test_cam_id, cameras)

    def test_04_test_rtsp_endpoint_handling(self):
        """Test RTSP endpoint input validation and graceful failure on unreachable host."""
        # Empty IP test
        resp = self.client.post("/api/cameras/test_rtsp", json={"ip": ""})
        self.assertEqual(resp.status_code, 400)

        # Unreachable IP test (should timeout gracefully within seconds without 500 crash)
        resp2 = self.client.post("/api/cameras/test_rtsp", json={
            "ip": "192.0.2.1",  # RFC 5737 TEST-NET (guaranteed unroutable)
            "port": 554,
            "user": "test",
            "pass": "test",
            "channel": 102
        })
        self.assertEqual(resp2.status_code, 200)
        data = resp2.json()
        self.assertFalse(data["success"])
        msg_lower = data["message"].lower()
        self.assertTrue("gagal" in msg_lower or "timeout" in msg_lower, f"Unexpected message: {data['message']}")

    def test_05_ssrf_and_port_whitelist_guard(self):
        """Test SSRF rejection on loopback addresses and non-whitelisted ports."""
        # 1. Loopback IP in test_rtsp
        resp_lb = self.client.post("/api/cameras/test_rtsp", json={"ip": "127.0.0.1", "port": 554})
        self.assertEqual(resp_lb.status_code, 200)
        self.assertFalse(resp_lb.json()["success"])
        self.assertIn("loopback", resp_lb.json()["message"].lower())

        # 2. Localhost string in test_rtsp
        resp_local = self.client.post("/api/cameras/test_rtsp", json={"ip": "localhost", "port": 554})
        self.assertEqual(resp_local.status_code, 200)
        self.assertFalse(resp_local.json()["success"])
        self.assertIn("loopback", resp_local.json()["message"].lower())

        # 3. Disallowed port in test_rtsp (e.g. 80, 22, 8080)
        resp_port = self.client.post("/api/cameras/test_rtsp", json={"ip": "192.168.1.100", "port": 8080})
        self.assertEqual(resp_port.status_code, 200)
        self.assertFalse(resp_port.json()["success"])
        self.assertIn("port 8080 tidak diizinkan", resp_port.json()["message"].lower())

        # 4. Loopback in add_camera (should return 400)
        resp_add_lb = self.client.post("/api/cameras/add", json={
            "camera_id": "cam_evil_lb",
            "ip": "127.0.0.1",
            "port": 554,
        })
        self.assertEqual(resp_add_lb.status_code, 400)
        self.assertIn("loopback", resp_add_lb.json()["detail"].lower())

        # 5. Disallowed port in add_camera (should return 400)
        resp_add_port = self.client.post("/api/cameras/add", json={
            "camera_id": "cam_evil_port",
            "ip": "192.168.1.100",
            "port": 22,
        })
        self.assertEqual(resp_add_port.status_code, 400)
        self.assertIn("port 22 tidak diizinkan", resp_add_port.json()["detail"].lower())

        # 6. Custom RTSP with disallowed scheme or loopback in add_camera
        resp_add_url = self.client.post("/api/cameras/add", json={
            "camera_id": "cam_evil_url",
            "rtsp_url": "http://192.168.1.100:554/feed",
        })
        self.assertEqual(resp_add_url.status_code, 400)
        self.assertIn("protokol", resp_add_url.json()["detail"].lower())

    def test_06_credential_injection_guard(self):
        """Test rejection of credentials containing newline characters (preventing .env injection)."""
        resp = self.client.post("/api/cameras/add", json={
            "camera_id": "cam_inject",
            "ip": "192.168.1.100",
            "port": 554,
            "user": "admin\nEVIL_VAR=hacked",
            "pass": "password",
        })
        self.assertEqual(resp.status_code, 400)
        self.assertIn("baris baru", resp.json()["detail"].lower())

    def test_07_csrf_origin_and_token_protection(self):
        """Test comprehensive CSRF defense: Origin verification and Double-Submit Cookie tokens."""
        # 1. Page load sets CSRF cookie and injects meta tag
        dash_resp = self.client.get("/dashboard")
        self.assertEqual(dash_resp.status_code, 200)
        csrf_cookie = dash_resp.cookies.get("csrf_token")
        self.assertTrue(csrf_cookie is not None and len(csrf_cookie) > 10, "CSRF cookie must be issued")
        self.assertIn('name="csrf-token"', dash_resp.text)
        self.assertIn(csrf_cookie, dash_resp.text)

        # 2. Cross-Origin attack with malicious Origin must be blocked (403 Forbidden)
        evil_origin_resp = self.client.post(
            "/api/cameras/test_rtsp",
            json={"ip": "192.168.1.100", "port": 554},
            headers={"Origin": "http://evil-attacker.com"},
        )
        self.assertEqual(evil_origin_resp.status_code, 403)
        self.assertIn("csrf protection", evil_origin_resp.json()["detail"].lower())

        # 3. Cross-Origin attack with malicious Referer must be blocked (403 Forbidden)
        evil_ref_resp = self.client.post(
            "/api/cameras/test_rtsp",
            json={"ip": "192.168.1.100", "port": 554},
            headers={"Referer": "http://malicious-site.org/attack.html"},
        )
        self.assertEqual(evil_ref_resp.status_code, 403)
        self.assertIn("csrf protection", evil_ref_resp.json()["detail"].lower())

        # 4. Mismatched X-CSRF-Token against cookie must be blocked (403 Forbidden)
        mismatch_resp = self.client.post(
            "/api/cameras/test_rtsp",
            json={"ip": "192.168.1.100", "port": 554},
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": "invalid_forged_token_value",
            },
            cookies={"csrf_token": csrf_cookie},
        )
        self.assertEqual(mismatch_resp.status_code, 403)
        self.assertIn("token csrf tidak valid", mismatch_resp.json()["detail"].lower())

        # 5. Legitimate request with matching Origin and CSRF token must pass (not 403)
        valid_resp = self.client.post(
            "/api/cameras/test_rtsp",
            json={"ip": "127.0.0.1", "port": 554},  # Reaches IP validator, proving CSRF passed
            headers={
                "Origin": "http://testserver",
                "X-CSRF-Token": csrf_cookie,
            },
            cookies={"csrf_token": csrf_cookie},
        )
        # Status code is 200 (SSRF handler caught loopback), proving CSRF middleware allowed request
        self.assertEqual(valid_resp.status_code, 200)
        self.assertFalse(valid_resp.json()["success"])
        self.assertIn("loopback", valid_resp.json()["message"].lower())

    def test_08_api_key_and_owasp_security_headers(self):
        """Test API Key authentication on REST endpoints and verify OWASP security headers."""
        # 1. Unauthenticated API request must be rejected with 401
        unauth_resp = self.client.get(
            "/api/status/cam_01",
            headers={"X-Force-Auth-Test": "1"},
        )
        self.assertEqual(unauth_resp.status_code, 401)
        self.assertIn("kunci api", unauth_resp.json()["detail"].lower())

        # 2. Authenticated API request via X-API-Key header must succeed
        auth_hdr_resp = self.client.get(
            "/api/status/cam_01",
            headers={
                "X-Force-Auth-Test": "1",
                "X-API-Key": "smart-cctv-royal2026",
            },
        )
        self.assertEqual(auth_hdr_resp.status_code, 200)

        # 3. Authenticated API request via query parameter ?api_key= must succeed
        auth_param_resp = self.client.get(
            "/api/status/cam_01?api_key=smart-cctv-royal2026",
            headers={"X-Force-Auth-Test": "1"},
        )
        self.assertEqual(auth_param_resp.status_code, 200)

        # 4. OWASP Security Headers check
        self.assertEqual(auth_hdr_resp.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(auth_hdr_resp.headers.get("X-Frame-Options"), "SAMEORIGIN")
        self.assertEqual(auth_hdr_resp.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")


if __name__ == "__main__":
    unittest.main(verbosity=2)



