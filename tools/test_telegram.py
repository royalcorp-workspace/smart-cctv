"""Telegram Bot Diagnostic, Chat ID Detection, and Test Tool for Smart CCTV.

Usage:
    python tools/test_telegram.py --check
    python tools/test_telegram.py --get-chat-id
    python tools/test_telegram.py --send-test
    python tools/test_telegram.py --send-photo
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional
from dotenv import load_dotenv

import cv2
import numpy as np
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.config_loader import ensure_env_loaded

CONFIG_PATH = PROJECT_ROOT / "configs" / "telegram.json"


def load_config() -> dict:
    """Load configuration from .env or configs/telegram.json."""
    ensure_env_loaded()

    cfg = {
        "enabled": True,
        "bot_token": "",
        "recipients": {
            "global_admins": [],
            "cameras": {"cam_01": []}
        }
    }

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {CONFIG_PATH}: {e}")

    # Environment overrides
    env_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if env_token and not env_token.startswith("${"):
        cfg["bot_token"] = env_token

    env_admin = os.getenv("TELEGRAM_GLOBAL_ADMIN_CHAT_ID", "").strip()
    if env_admin and not env_admin.startswith("${"):
        cfg.setdefault("recipients", {}).setdefault("global_admins", [])
        if env_admin not in cfg["recipients"]["global_admins"]:
            cfg["recipients"]["global_admins"].append(env_admin)

    env_cam01 = os.getenv("TELEGRAM_CAM01_CHAT_ID", "").strip()
    if env_cam01 and not env_cam01.startswith("${"):
        cfg.setdefault("recipients", {}).setdefault("cameras", {}).setdefault("cam_01", [])
        if env_cam01 not in cfg["recipients"]["cameras"]["cam_01"]:
            cfg["recipients"]["cameras"]["cam_01"].append(env_cam01)

    return cfg


def save_config(cfg: dict) -> None:
    """Save configuration to configs/telegram.json."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"[OK] Configuration updated successfully: {CONFIG_PATH}")


def check_bot(token: str) -> bool:
    """Verify bot token validity with Telegram API getMe."""
    url = f"https://api.telegram.org/bot{token}/getMe"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            bot = data.get("result", {})
            print("=" * 60)
            print("         TELEGRAM BOT CONNECTION VERIFIED         ")
            print("=" * 60)
            print(f"Bot Name     : {bot.get('first_name')}")
            print(f"Bot Username : @{bot.get('username')}")
            print(f"Bot ID       : {bot.get('id')}")
            print(f"Can Join Grp : {bot.get('can_join_groups')}")
            print("Status       : ONLINE & READY")
            print("=" * 60)
            return True
        else:
            print(f"[FAILED] Telegram API Error ({resp.status_code}): {data.get('description')}")
            return False
    except Exception as e:
        print(f"[ERROR] Network connection failed: {e}")
        return False


def get_recent_chat_ids(token: str, auto_save: bool = False) -> List[Dict]:
    """Retrieve recent updates from Telegram and extract user Chat IDs."""
    url = f"https://api.telegram.org/bot{token}/getUpdates"
    print("\n[INFO] Contacting Telegram servers to detect incoming messages...")
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if not data.get("ok"):
            print(f"[FAILED] Could not get updates: {data.get('description')}")
            return []

        updates = data.get("result", [])
        if not updates:
            print("\n[!] Tidak ditemukan pesan masuk terbaru.")
            print("    TIPS: Buka aplikasi Telegram Anda, cari bot Anda, lalu kirim pesan '/start' atau ketik apa saja ke bot.")
            print("    Setelah mengirim pesan, jalankan kembali perintah ini.\n")
            return []

        detected_users = {}
        for u in updates:
            msg = u.get("message") or u.get("channel_post") or u.get("my_chat_member")
            if not msg:
                continue
            chat = msg.get("chat", {})
            chat_id = str(chat.get("id"))
            chat_type = chat.get("type", "private")
            name = f"{chat.get('first_name', '')} {chat.get('last_name', '')}".strip() or chat.get("title", "Unknown")
            username = chat.get("username", "-")
            text = msg.get("text", "<Media/Action>")
            date_str = datetime.datetime.fromtimestamp(msg.get("date", 0)).strftime("%Y-%m-%d %H:%M:%S")

            detected_users[chat_id] = {
                "chat_id": chat_id,
                "name": name,
                "username": username,
                "type": chat_type,
                "last_message": text,
                "date": date_str,
            }

        print("\n" + "=" * 65)
        print("          DAFTAR CHAT ID TELEGRAM YANG TERDETEKSI         ")
        print("=" * 65)
        for idx, (cid, info) in enumerate(detected_users.items(), 1):
            print(f"{idx}. Chat ID   : {cid}")
            print(f"   Nama      : {info['name']} (@{info['username']})")
            print(f"   Tipe      : {info['type']}")
            print(f"   Pesan     : \"{info['last_message']}\" ({info['date']})")
            print("-" * 65)

        if auto_save and detected_users:
            first_chat_id = list(detected_users.keys())[0]
            cfg = load_config()
            cfg["recipients"]["global_admins"] = [first_chat_id]
            cfg["recipients"]["cameras"]["cam_01"] = [first_chat_id]
            save_config(cfg)
            print(f"[OK] Chat ID '{first_chat_id}' otomatis disimpan ke configs/telegram.json!")

        return list(detected_users.values())

    except Exception as e:
        print(f"[ERROR] Failed to fetch updates: {e}")
        return []


def send_test_message(token: str, chat_id: str) -> bool:
    """Send a plain text test notification to Telegram."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": (
            "✅ <b>[TEST NOTIFIKASI] SMART CCTV 2.0</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Koneksi bot Telegram dan Chat ID Anda berhasil diverifikasi!\n"
            f"Waktu Uji: <code>{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</code>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "Sistem CCTV siap mengirimkan notifikasi foto saat terdeteksi barang tertinggal."
        ),
        "parse_mode": "HTML",
    }
    print(f"\n[INFO] Mengirim pesan uji coba ke Chat ID: {chat_id}...")
    try:
        resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            print("[SUCCESS] Pesan uji coba berhasil terkirim ke Telegram!")
            return True
        else:
            print(f"[FAILED] Telegram API Error: {data.get('description')}")
            return False
    except Exception as e:
        print(f"[ERROR] Gagal mengirim pesan: {e}")
        return False


def send_test_photo(token: str, chat_id: str) -> bool:
    """Generate mock CCTV snapshot frame and send via sendPhoto."""
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    print(f"\n[INFO] Membuat sampel snapshot foto dan mengirim ke Chat ID: {chat_id}...")

    # Create dummy 640x480 surveillance frame with simulated detection
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:] = (35, 35, 35)

    # Simulated floor grid
    for y in range(100, 480, 40):
        cv2.line(canvas, (0, y), (640, y), (50, 50, 50), 1)
    for x in range(0, 640, 60):
        cv2.line(canvas, (x, 100), (x, 480), (50, 50, 50), 1)

    # Simulated backpack box
    bx, by, bw, bh = 280, 220, 90, 80
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
    cv2.rectangle(canvas, (bx, by - 22), (bx + 190, by), (0, 0, 255), -1)
    cv2.putText(canvas, "ALERT: TAS TERTINGGAL (ID #1)", (bx + 4, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    # OSD timestamp
    ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(canvas, f"SMART CCTV | CAM: CAM_01 | ZONA: zone_2_transit | {ts_str}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

    # Encode to JPEG
    _, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_bytes = enc.tobytes()

    caption = (
        "⚠️ <b>[TEST PERINGATAN] BARANG TERTINGGAL</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "📹 <b>Kamera</b>: <code>cam_01</code> (Area Transit Depan)\n"
        "📍 <b>Zona</b>: <code>zone_2_transit</code>\n"
        "🏷️ <b>Objek</b>: Tas / Ransel (Track ID #1)\n"
        "⏱️ <b>Durasi Ditinggal</b>: <b>60 detik</b> (Ambang: 60s)\n"
        f"🕒 <b>Waktu Uji</b>: <code>{ts_str}</code>\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "🚨 <i>Ini adalah foto uji simulasi notifikasi Smart CCTV. Format foto dan caption bekerja dengan sempurna!</i>"
    )

    try:
        files = {"photo": ("test_snapshot.jpg", img_bytes, "image/jpeg")}
        data = {
            "chat_id": chat_id,
            "caption": caption,
            "parse_mode": "HTML",
        }
        resp = requests.post(url, data=data, files=files, timeout=10)
        data_resp = resp.json()
        if resp.status_code == 200 and data_resp.get("ok"):
            print("[SUCCESS] Foto snapshot simulasi berhasil terkirim ke Telegram Anda!")
            return True
        else:
            print(f"[FAILED] Telegram API Error: {data_resp.get('description')}")
            return False
    except Exception as e:
        print(f"[ERROR] Gagal mengirim foto: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic & Testing Utility for Smart CCTV Telegram Bot.")
    parser.add_argument("--check", action="store_true", help="Periksa status koneksi bot token Telegram.")
    parser.add_argument("--get-chat-id", action="store_true", help="Deteksi Chat ID pengguna secara otomatis via getUpdates.")
    parser.add_argument("--save-chat-id", action="store_true", help="Deteksi Chat ID dan otomatis simpan ke configs/telegram.json.")
    parser.add_argument("--send-test", action="store_true", help="Kirim pesan teks uji coba ke Chat ID yang terdaftar.")
    parser.add_argument("--send-photo", action="store_true", help="Kirim simulasi snapshot foto CCTV ke Chat ID yang terdaftar.")
    parser.add_argument("--chat-id", type=str, default="", help="Tentukan Chat ID manual untuk pengujian.")

    args = parser.parse_args()

    cfg = load_config()
    token = cfg.get("bot_token", "").strip()

    if not token:
        print("[ERROR] 'bot_token' belum dikonfigurasi pada configs/telegram.json!")
        sys.exit(1)

    if args.check:
        check_bot(token)
    elif args.get_chat_id or args.save_chat_id:
        get_recent_chat_ids(token, auto_save=args.save_chat_id)
    elif args.send_test:
        target_id = args.chat_id
        if not target_id:
            admins = cfg.get("recipients", {}).get("global_admins", [])
            target_id = admins[0] if admins else ""
        if not target_id:
            print("[!] Belum ada Chat ID yang dikonfigurasi. Jalankan: python tools/test_telegram.py --get-chat-id")
            sys.exit(1)
        send_test_message(token, target_id)
    elif args.send_photo:
        target_id = args.chat_id
        if not target_id:
            admins = cfg.get("recipients", {}).get("global_admins", [])
            target_id = admins[0] if admins else ""
        if not target_id:
            print("[!] Belum ada Chat ID yang dikonfigurasi. Jalankan: python tools/test_telegram.py --get-chat-id")
            sys.exit(1)
        send_test_photo(token, target_id)
    else:
        # Default interactive run
        print("\n=======================================================")
        print("      SMART CCTV 2.0 - TELEGRAM DIAGNOSTIC TOOL       ")
        print("=======================================================")
        bot_ok = check_bot(token)
        if bot_ok:
            recipients = cfg.get("recipients", {}).get("global_admins", [])
            print(f"\nChat ID Terdaftar saat ini: {recipients or 'Belum ada'}")
            print("\nPilihan perintah:")
            print("1. Deteksi Chat ID otomatis : python tools/test_telegram.py --get-chat-id")
            print("2. Simpan Chat ID otomatis  : python tools/test_telegram.py --save-chat-id")
            print("3. Uji kirim pesan teks     : python tools/test_telegram.py --send-test")
            print("4. Uji kirim foto snapshot  : python tools/test_telegram.py --send-photo\n")


if __name__ == "__main__":
    main()
