#!/usr/bin/env bash
# ==============================================================================
# Script: setup_nginx_https.sh
# Purpose: One-command automated installer for Nginx Reverse Proxy with HTTPS/TLS
# Target OS: Ubuntu / Debian Linux Server (172.16.2.185)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_SRC="${SCRIPT_DIR}/smart-cctv.conf"
GEN_SSL_SRC="${SCRIPT_DIR}/generate_ssl_cert.sh"

echo "=============================================================================="
echo "    Smart CCTV 2.0 - HTTPS / TLS Nginx Reverse Proxy Setup Installer"
echo "=============================================================================="

# 1. Memeriksa izin sudo/root
if [ "$(id -u)" -ne 0 ]; then
    echo "[ERROR] Installer ini harus dijalankan dengan privilege root / sudo."
    echo "Contoh: sudo bash scripts/nginx/setup_nginx_https.sh"
    exit 1
fi

# 2. Memeriksa & menginstal Nginx dan OpenSSL jika belum terpasang
echo "[STEP 1/5] Memeriksa paket Nginx & OpenSSL..."
if ! command -v nginx >/dev/null 2>&1 || ! command -v openssl >/dev/null 2>&1; then
    echo "Paket belum lengkap. Menginstal nginx dan openssl via apt..."
    apt-get update -y
    apt-get install -y nginx openssl
else
    echo "Nginx dan OpenSSL telah terpasang di sistem."
fi

# 3. Membuat sertifikat SSL/TLS dengan SAN
echo "[STEP 2/5] Menyiapkan sertifikat SSL/TLS..."
chmod +x "${GEN_SSL_SRC}"
bash "${GEN_SSL_SRC}"

# 4. Memasang konfigurasi Virtual Host Nginx
echo "[STEP 3/5] Memasang konfigurasi Nginx Smart CCTV..."
cp -f "${CONF_SRC}" /etc/nginx/sites-available/smart-cctv.conf

# Buat symlink ke sites-enabled
mkdir -p /etc/nginx/sites-enabled
ln -sf /etc/nginx/sites-available/smart-cctv.conf /etc/nginx/sites-enabled/smart-cctv.conf

# Nonaktifkan default site jika ada konflik
if [ -f /etc/nginx/sites-enabled/default ]; then
    echo "Menonaktifkan virtual host default Nginx..."
    rm -f /etc/nginx/sites-enabled/default
fi

# 5. Menguji sintaks konfigurasi Nginx
echo "[STEP 4/5] Menguji sintaks konfigurasi Nginx..."
nginx -t

# 6. Menyalakan dan mengaktifkan service Nginx
echo "[STEP 5/5] Me-restart service Nginx..."
systemctl restart nginx
systemctl enable nginx

echo "=============================================================================="
echo "[BERHASIL] Nginx Reverse Proxy dengan HTTPS/TLS berhasil aktif!"
echo ""
echo "Akses Web Dashboard Aman:"
echo "  - HTTPS (Utama)  : https://172.16.2.185/"
echo "  - HTTP (Redirect): http://172.16.2.185/ -> otomatis dialihkan ke HTTPS"
echo "  - Local Host     : https://localhost/"
echo ""
echo "Catatan Streaming:"
echo "  - Fitur zero-latency proxy buffering telah diaktifkan untuk stream live CCTV."
echo "  - Pastikan backend Python (main.py) tetap berjalan di port 8000."
echo "=============================================================================="
