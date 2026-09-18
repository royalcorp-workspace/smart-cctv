"""Unit and Integration Tests: Pure Python SSL/TLS Certificate Manager and HTTPS Web Server."""

from pathlib import Path
import sys
import unittest

from cryptography import x509
from fastapi.testclient import TestClient

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from web.ssl_manager import get_or_create_ssl_certificates
from web.server import DashboardServer, app


class TestSSLHttps(unittest.TestCase):
    def test_01_ssl_certificate_generation_and_san(self):
        """Verify pure Python SSL certificate generation with SAN extension."""
        cert_file, key_file = get_or_create_ssl_certificates(server_ip="172.16.2.185")
        self.assertTrue(cert_file.exists(), "Certificate file must exist")
        self.assertTrue(key_file.exists(), "Key file must exist")
        self.assertGreater(cert_file.stat().st_size, 500, "Cert size must be valid")
        self.assertGreater(key_file.stat().st_size, 500, "Key size must be valid")

        # Parse X.509 certificate and verify SAN
        with open(cert_file, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())

        san_ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        ip_addresses = [str(ip) for ip in san_ext.value.get_values_for_type(x509.IPAddress)]
        dns_names = san_ext.value.get_values_for_type(x509.DNSName)

        self.assertIn("172.16.2.185", ip_addresses)
        self.assertIn("127.0.0.1", ip_addresses)
        self.assertIn("localhost", dns_names)

    def test_02_dashboard_server_https_config(self):
        """Verify DashboardServer properly loads SSL certificates into Uvicorn config."""
        # Test starting with enable_https=True
        server_thread = DashboardServer.start(
            host="127.0.0.1",
            port=9999,
            enable_https=True,
            server_ip="172.16.2.185",
        )
        self.assertTrue(server_thread.is_alive())
        self.assertTrue(DashboardServer._is_https)

        uv_config = DashboardServer._uvicorn_server.config
        self.assertIsNotNone(uv_config.ssl_keyfile)
        self.assertIsNotNone(uv_config.ssl_certfile)
        self.assertTrue(Path(uv_config.ssl_keyfile).exists())
        self.assertTrue(Path(uv_config.ssl_certfile).exists())

        # Cleanup test server
        DashboardServer.stop()

    def test_03_secure_cookie_on_https_request(self):
        """Verify that requests over HTTPS receive Secure cookie flag."""
        client = TestClient(app, base_url="https://testserver")
        resp = client.get("/dashboard")
        self.assertEqual(resp.status_code, 200)
        set_cookie = resp.headers.get("set-cookie", "").lower()
        self.assertIn("csrf_token=", set_cookie)
        self.assertIn("secure", set_cookie)


if __name__ == "__main__":
    unittest.main(verbosity=2)
