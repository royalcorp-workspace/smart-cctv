#!/usr/bin/env bash
# ==============================================================================
# Script: generate_ssl_cert.sh
# Purpose: Generate Self-Signed SSL/TLS Certificate with SAN (Subject Alternative Name)
# Target: 172.16.2.185, 127.0.0.1, localhost
# Validity: 3650 days (10 years)
# ==============================================================================

set -euo pipefail

CERT_DIR="/etc/ssl/certs"
KEY_DIR="/etc/ssl/private"
CERT_FILE="${CERT_DIR}/smart-cctv.crt"
KEY_FILE="${KEY_DIR}/smart-cctv.key"
TMP_CNF="/tmp/smart_cctv_openssl_san.cnf"

echo "[SSL-GEN] Memeriksa hak akses root/sudo..."
if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] Script ini harus dijalankan dengan hak akses root atau sudo."
    exit 1
fi

echo "[SSL-GEN] Memastikan direktori sertifikat SSL tersedia..."
mkdir -p "${CERT_DIR}"
mkdir -p "${KEY_DIR}"

echo "[SSL-GEN] Menyusun konfigurasi OpenSSL dengan Subject Alternative Name (SAN)..."
cat > "${TMP_CNF}" << 'EOF'
[req]
default_bits        = 2048
prompt              = no
default_md          = sha256
distinguished_name  = dn
x509_extensions     = v3_req

[dn]
C                   = ID
ST                  = Jakarta
L                   = Jakarta
O                   = Smart CCTV Surveillance System
OU                  = Security Infrastructure
CN                  = 172.16.2.185

[v3_req]
basicConstraints    = CA:FALSE
keyUsage            = nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage    = serverAuth
subjectAltName      = @alt_names

[alt_names]
IP.1                = 172.16.2.185
IP.2                = 127.0.0.1
DNS.1               = localhost
DNS.2               = smart-cctv.local
EOF

echo "[SSL-GEN] Menghasilkan kunci privat RSA 2048-bit dan sertifikat x509..."
openssl req -x509 -nodes -days 3650 \
    -newkey rsa:2048 \
    -keyout "${KEY_FILE}" \
    -out "${CERT_FILE}" \
    -config "${TMP_CNF}"

# Hapus file konfigurasi temporer
rm -f "${TMP_CNF}"

echo "[SSL-GEN] Mengamankan izin akses file sertifikat..."
chmod 600 "${KEY_FILE}"
chmod 644 "${CERT_FILE}"

echo "=============================================================================="
echo "[SUCCESS] Sertifikat SSL/TLS berhasil dibuat!"
echo "  - Certificate: ${CERT_FILE}"
echo "  - Private Key: ${KEY_FILE}"
echo "  - Masa Berlaku: 3650 Hari (~10 Tahun)"
echo "  - SAN IPs: 172.16.2.185, 127.0.0.1"
echo "  - SAN Domains: localhost, smart-cctv.local"
echo "=============================================================================="
