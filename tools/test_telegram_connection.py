"""Verification and connection testing script for Smart CCTV Telegram Bot."""

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CAM_CONFIG_PATH = PROJECT_ROOT / "cameras" / "cam_01" / "config.json"
GLOBAL_TELEGRAM_PATH = PROJECT_ROOT / "configs" / "telegram.json"

TEST_MESSAGE = "Smart CCTV: Koneksi Bot Telegram Berhasil Dikonfigurasi."


def load_credentials() -> tuple[str, str, bool]:
    """Load Telegram bot credentials and chat ID from camera config or global config."""
    bot_token = ""
    chat_id = ""
    enabled = False

    # 1. Primary: cameras/cam_01/config.json
    if CAM_CONFIG_PATH.exists():
        try:
            with open(CAM_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            tg_cfg = data.get("telegram", {})
            if isinstance(tg_cfg, dict):
                bot_token = str(tg_cfg.get("bot_token", "")).strip()
                chat_id = str(tg_cfg.get("chat_id", "")).strip()
                enabled = bool(tg_cfg.get("enabled", False))
        except Exception as e:
            print(f"[WARN] Gagal membaca {CAM_CONFIG_PATH}: {e}")

    # 2. Fallback: configs/telegram.json
    if (not bot_token or not chat_id) and GLOBAL_TELEGRAM_PATH.exists():
        try:
            with open(GLOBAL_TELEGRAM_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not bot_token:
                bot_token = str(data.get("bot_token", "")).strip()
            if not chat_id:
                recipients = data.get("recipients", {})
                admins = recipients.get("global_admins", [])
                cam_recipients = recipients.get("cameras", {}).get("cam_01", [])
                if cam_recipients:
                    chat_id = str(cam_recipients[0]).strip()
                elif admins:
                    chat_id = str(admins[0]).strip()
            if not enabled:
                enabled = bool(data.get("enabled", False))
        except Exception as e:
            print(f"[WARN] Gagal membaca {GLOBAL_TELEGRAM_PATH}: {e}")

    return bot_token, chat_id, enabled


def main() -> int:
    print("=" * 65)
    print("       SMART CCTV - TELEGRAM BOT CONNECTION TEST SUITE       ")
    print("=" * 65)

    bot_token, chat_id, enabled = load_credentials()

    print(f"\n[1] Pengecekan Konfigurasi Kredensial:")
    print(f" - Sumber Konfigurasi : {CAM_CONFIG_PATH}")
    print(f" - Status Enabled     : {enabled}")
    print(f" - Bot Token Terbaca  : {bot_token[:10]}...{bot_token[-6:] if len(bot_token) > 16 else ''}")
    print(f" - Chat ID Tujuan     : {chat_id}")

    if not bot_token:
        print("\n[ERROR] Bot token belum dikonfigurasi!")
        return 1

    if not chat_id or chat_id == "ISI_CHAT_ID_DISINI":
        print("\n[ERROR] Chat ID belum diisi atau masih berupa placeholder!")
        return 1

    # [2] Test getMe
    print(f"\n[2] Memverifikasi Bot Token ke Telegram API (getMe)...")
    get_me_url = f"https://api.telegram.org/bot{bot_token}/getMe"
    req_me = urllib.request.Request(get_me_url, headers={"User-Agent": "SmartCCTV/1.0"})

    try:
        with urllib.request.urlopen(req_me, timeout=10) as resp:
            status_code = resp.status
            body = json.loads(resp.read().decode("utf-8"))
            if status_code == 200 and body.get("ok"):
                bot_info = body.get("result", {})
                bot_name = bot_info.get("first_name", "")
                bot_user = bot_info.get("username", "")
                print(f" -> [PASS] HTTP {status_code} OK")
                print(f" -> Bot Name : {bot_name} (@{bot_user})")
            else:
                print(f" -> [FAIL] HTTP {status_code}: {body}")
                return 1
    except urllib.error.HTTPError as e:
        print(f" -> [FAIL] HTTP Error {e.code}: {e.read().decode('utf-8')}")
        return 1
    except Exception as e:
        print(f" -> [FAIL] Network Error: {e}")
        return 1

    # [3] Test sendMessage
    print(f"\n[3] Mengirim Pesan Uji Coba (sendMessage)...")
    print(f" - Pesan: \"{TEST_MESSAGE}\"")
    send_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": TEST_MESSAGE,
    }).encode("utf-8")

    req_send = urllib.request.Request(send_url, data=payload, headers={"User-Agent": "SmartCCTV/1.0"})

    try:
        with urllib.request.urlopen(req_send, timeout=10) as resp:
            status_code = resp.status
            body = json.loads(resp.read().decode("utf-8"))
            if status_code == 200 and body.get("ok"):
                result = body.get("result", {})
                msg_id = result.get("message_id")
                chat_info = result.get("chat", {})
                recipient_name = chat_info.get("first_name") or chat_info.get("title") or "Unknown"
                recipient_type = chat_info.get("type", "private")

                print(f" -> [PASS] HTTP {status_code} OK")
                print(f" -> Message ID    : {msg_id}")
                print(f" -> Penerima      : {recipient_name} (Tipe: {recipient_type})")
                print("\n" + "=" * 65)
                print("   SUKSES: PESAN UJI COBA BERHASIL DITERIMA DI TELEGRAM!   ")
                print("=" * 65)
                return 0
            else:
                print(f" -> [FAIL] HTTP {status_code}: {body}")
                return 1
    except urllib.error.HTTPError as e:
        print(f" -> [FAIL] HTTP Error {e.code}: {e.read().decode('utf-8')}")
        return 1
    except Exception as e:
        print(f" -> [FAIL] Network Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
