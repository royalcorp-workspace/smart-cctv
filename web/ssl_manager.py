"""Smart CCTV 2.0 - Embedded SSL/TLS Certificate Manager.

Automatically manages, provisions, and verifies self-signed X.509 SSL/TLS certificates
with Subject Alternative Names (SAN) for pure Python HTTPS serving (no Nginx required).
"""

import datetime
import ipaddress
import logging
from pathlib import Path
from typing import Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

logger = logging.getLogger("smart_cctv")


def get_or_create_ssl_certificates(
    cert_dir: Path = None,
    server_ip: str = "172.16.2.185",
) -> Tuple[Path, Path]:
    """Ensure SSL certificates exist for HTTPS/TLS, generating high-security SAN certs if missing.

    Returns:
        Tuple[Path, Path]: (cert_file_path, key_file_path)
    """
    if cert_dir is None:
        cert_dir = Path(__file__).resolve().parent.parent / "data" / "certs"
    cert_dir.mkdir(parents=True, exist_ok=True)

    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"

    if (
        cert_file.exists()
        and key_file.exists()
        and cert_file.stat().st_size > 0
        and key_file.stat().st_size > 0
    ):
        return cert_file, key_file

    logger.info(f"[SSLManager] Generating self-signed SSL/TLS certificate with SAN for {server_ip}...")

    # 1. Generate RSA 2048-bit Private Key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # 2. Setup Subject and Issuer (Self-Signed)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "ID"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Jakarta"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Jakarta"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Smart CCTV Security System"),
        x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Surveillance Operations"),
        x509.NameAttribute(NameOID.COMMON_NAME, server_ip),
    ])

    # 3. Setup SAN (Subject Alternative Names) for IPs and Localhost
    san_list = [
        x509.IPAddress(ipaddress.ip_address(server_ip)),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.DNSName("localhost"),
        x509.DNSName("smart-cctv.local"),
    ]

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))  # 10 years validity
        .add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .sign(private_key, hashes.SHA256())
    )

    # 4. Write Private Key (PEM format)
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with open(key_file, "wb") as f:
        f.write(key_pem)

    # 5. Write Certificate (PEM format)
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    with open(cert_file, "wb") as f:
        f.write(cert_pem)

    logger.info(f"[SSLManager] SSL/TLS certificates generated successfully at: {cert_dir}")
    return cert_file, key_file
