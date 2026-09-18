"""Automated Unit and Integrity Tests: Nginx HTTPS/TLS Configuration & SSL Setup.

Validates:
1. Nginx virtual host configuration syntax & required CCTV streaming directives.
2. SSL generation script structure and SAN declarations (IP:172.16.2.185).
3. Backend proxy headers awareness (X-Forwarded-Proto sets Secure cookie).
"""

from pathlib import Path
import re
import sys
import unittest

from fastapi.testclient import TestClient

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from web.server import app


class TestNginxSSLConfiguration(unittest.TestCase):
    def setUp(self):
        self.nginx_dir = _ROOT_DIR / "scripts" / "nginx"
        self.conf_file = self.nginx_dir / "smart-cctv.conf"
        self.ssl_gen_file = self.nginx_dir / "generate_ssl_cert.sh"
        self.setup_file = self.nginx_dir / "setup_nginx_https.sh"
        self.client = TestClient(app)

    def test_01_nginx_conf_structure(self):
        """Verify Nginx virtual host contains port 80 redirect, 443 SSL, and CCTV stream tuning."""
        self.assertTrue(self.conf_file.exists(), "smart-cctv.conf must exist")
        content = self.conf_file.read_text(encoding="utf-8")

        # 1. Port 80 HTTP redirect
        self.assertIn("listen 80;", content)
        self.assertIn("return 301 https://$host$request_uri;", content)

        # 2. Port 443 HTTPS SSL
        self.assertIn("listen 443 ssl http2;", content)
        self.assertIn("ssl_certificate /etc/ssl/certs/smart-cctv.crt;", content)
        self.assertIn("ssl_certificate_key /etc/ssl/private/smart-cctv.key;", content)

        # 3. Security Headers
        self.assertIn("Strict-Transport-Security", content)
        self.assertIn("X-Content-Type-Options \"nosniff\"", content)
        self.assertIn("X-Frame-Options \"SAMEORIGIN\"", content)

        # 4. CCTV Live Streaming Zero-Latency Tuning
        self.assertIn("proxy_buffering off;", content)
        self.assertIn("proxy_read_timeout 86400s;", content)
        self.assertIn("chunked_transfer_encoding on;", content)
        self.assertIn("tcp_nodelay on;", content)

    def test_02_ssl_generation_script(self):
        """Verify SSL generator script targets 172.16.2.185 with Subject Alternative Name (SAN)."""
        self.assertTrue(self.ssl_gen_file.exists(), "generate_ssl_cert.sh must exist")
        content = self.ssl_gen_file.read_text(encoding="utf-8")

        self.assertIn("IP.1                = 172.16.2.185", content)
        self.assertIn("IP.2                = 127.0.0.1", content)
        self.assertIn("DNS.1               = localhost", content)
        self.assertIn("days 3650", content)
        self.assertIn("chmod 600", content)

    def test_03_installer_script(self):
        """Verify setup script checks root, verifies nginx, and enables service."""
        self.assertTrue(self.setup_file.exists(), "setup_nginx_https.sh must exist")
        content = self.setup_file.read_text(encoding="utf-8")

        self.assertIn("nginx -t", content)
        self.assertIn("systemctl restart nginx", content)
        self.assertIn("sites-available/smart-cctv.conf", content)

    def test_04_backend_https_cookie_security(self):
        """Verify that requests routed via HTTPS (X-Forwarded-Proto) receive Secure cookies."""
        # Request via HTTP (direct or testclient)
        resp_http = self.client.get("/dashboard")
        self.assertEqual(resp_http.status_code, 200)
        cookie_header = resp_http.headers.get("set-cookie", "")
        # Under plain HTTP, secure flag should not be forced
        self.assertIn("csrf_token=", cookie_header)

        # Request via HTTPS proxy (X-Forwarded-Proto: https)
        # Use new client to simulate fresh TLS connection
        fresh_client = TestClient(app)
        resp_https = fresh_client.get("/dashboard", headers={"X-Forwarded-Proto": "https"})
        self.assertEqual(resp_https.status_code, 200)
        cookie_https = resp_https.headers.get("set-cookie", "").lower()
        self.assertIn("csrf_token=", cookie_https)
        self.assertIn("secure", cookie_https)


if __name__ == "__main__":
    unittest.main(verbosity=2)
