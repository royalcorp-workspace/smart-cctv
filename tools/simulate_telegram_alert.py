"""Instant Telegram Alert Simulator for Visual and Dispatch Verification.
Standardized to 1920x1080 (1080p) native resolution.
Simulates a Clear Area violation (unattended luggage > 60m) in zone_2_transit,
renders the full VisualHUD on a real 1080p camera frame,
saves storage/debug_output_1080p.jpg and storage/test_telegram_output.jpg for inspection,
and dispatches the dual-image alert directly to Telegram (with zero re-drawing).
"""

import datetime
import json
import logging
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
load_dotenv(PROJECT_ROOT / ".env")

from engine.config_loader import load_camera_config
from engine.rtsp_stream import ThreadedCapture
from engine.tracker import TrackedObject
from notification.local_alert import VisualHUD
from notification.telegram_alert import TelegramNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SimulateTelegramAlert")


def run_simulation():
    print("=================================================================")
    print("  SIMULASI SNAPSHOT ALERT TELEGRAM (1080p STANDARDIZED)          ")
    print("=================================================================")

    cam_dir = PROJECT_ROOT / "cameras" / "cam_01"
    roi_file = cam_dir / "roi_zones.json"
    config_file = cam_dir / "config.json"
    storage_dir = PROJECT_ROOT / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    debug_1080p_path = storage_dir / "debug_output_1080p.jpg"
    test_tg_path = storage_dir / "test_telegram_output.jpg"

    # 1. Load Zones & Config
    with open(roi_file, "r") as f:
        zones_data = json.load(f)

    cam_config = load_camera_config(config_file)
    source = cam_config.get("source")
    target_res = cam_config.get("target_resolution", [1920, 1080])
    target_w, target_h = int(target_res[0]), int(target_res[1])

    print(f"\n1. Target Standarisasi Resolusi: {target_w}x{target_h} (1080p)")
    print("   Koordinat ROI Zones (roi_zones.json):")
    for zid, pts in zones_data.items():
        if zid == "base_resolution" or zid.startswith("_"):
            continue
        print(f"   - {zid}: {pts}")

    # 2. Capture real frame from RTSP stream (or fallback)
    raw_frame = None
    if source:
        print(f"\n2. Mengambil frame riil dari kamera RTSP: {source}...")
        try:
            cap = ThreadedCapture(source=source)
            cap.start()
            for _ in range(30):
                ret, f = cap.read(timeout=0.3)
                if ret and f is not None:
                    raw_frame = f
                    break
            cap.stop()
        except Exception as e:
            print(f"   [Peringatan] Gagal membaca RTSP stream: {e}")

    if raw_frame is None:
        web_feed_path = storage_dir / "current_web_feed.jpg"
        if web_feed_path.exists():
            print(f"   Menggunakan live frame dari {web_feed_path}...")
            raw_frame = cv2.imread(str(web_feed_path))

    if raw_frame is None:
        print("   [Fallback] Membuat kanvas 1080p dummy...")
        raw_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # STANDARDIZE STRICTLY TO TARGET RESOLUTION (1920x1080)
    cur_h, cur_w = raw_frame.shape[:2]
    if cur_w != target_w or cur_h != target_h:
        print(f"   [Normalisasi] Menyesuaikan resolusi frame dari {cur_w}x{cur_h} ke {target_w}x{target_h} (1080p)...")
        raw_frame = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    else:
        print(f"   Frame sudah dalam resolusi native {target_w}x{target_h}.")

    # 3. Simulate TrackedObject (unattended luggage violation in zone_2_transit)
    # 640x480 inference coordinates:
    # (500, 391, 60, 53) -> scales to (1500, 880, 180, 119) in 1080p canvas (inside transit zone)
    now = time.time()
    dwell_duration = 3660.0  # 61.0 minutes (exceeds 60m threshold)
    track_id = 99

    simulated_obj = TrackedObject(
        track_id=track_id,
        centroid=(530, 417),
        anchor_centroid=(530, 417),
        bbox=(500, 391, 60, 53),
        zone_id="zone_2_transit",
        contour_area=3180.0,
        first_seen=now - dwell_duration,
        last_seen=now,
        stationary_start=now - dwell_duration,
        dwell_duration=dwell_duration,
        is_stationary=True,
        is_triggered=True,
        alert_sent=False,
        is_attended=False,
        is_active_this_frame=True,
        class_label="tas",
        dwell_threshold=3600.0,
    )

    # 4. Render complete VisualHUD on frame copy at 1080p
    print("\n3. Merender VisualHUD lengkap pada canvas 1920x1080...")
    annotated_frame = VisualHUD.render(
        canvas=raw_frame.copy(),
        zones=zones_data,
        tracked_objects=[simulated_obj],
        camera_id="cam_01",
        fps=25.0,
        is_connected=True,
        zone_base_resolution=(target_w, target_h),
    )

    # 5. Save local snapshots
    cv2.imwrite(str(debug_1080p_path), annotated_frame)
    cv2.imwrite(str(test_tg_path), annotated_frame)
    print(f"   [Saved] Snapshot 1080p tersimpan di: {debug_1080p_path}")
    print(f"   [Saved] Snapshot test tersimpan di: {test_tg_path}")

    # Verify dimensions
    saved_img = cv2.imread(str(debug_1080p_path))
    sh, sw = saved_img.shape[:2]
    print(f"   [Verifikasi Resolusi] {sw}x{sh} (Harus 1920x1080: {'OK' if (sw, sh) == (1920, 1080) else 'FAIL'})")

    # 6. Dispatch alert via TelegramNotifier
    print("\n4. Mengirim notifikasi simulasi ke Telegram Bot...")
    notifier = TelegramNotifier.get_instance()
    
    # Ensure camera config is registered
    if "telegram" in cam_config:
        notifier.register_camera_telegram("cam_01", cam_config["telegram"])

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    enqueued = notifier.dispatch_alert(
        camera_id="cam_01",
        zone_id="zone_2_transit",
        track_id=track_id,
        dwell_duration=dwell_duration,
        timestamp_str=timestamp_str,
        frame=raw_frame,
        overview_frame=annotated_frame,
        bbox=(500, 391, 60, 53),
        zone_name="Area Transit Depan (Penitipan)",
        zones=zones_data,
    )

    if enqueued:
        print("   [OK] Alert berhasil dimasukkan ke antrean worker Telegram.")
        print("   Menunggu pengiriman pesan media group oleh worker thread...")
        time.sleep(4.0)
        print("   Worker selesai memproses.")
    else:
        print("   [Info/Warning] Alert tidak dimasukkan ke antrean (cek kredensial/dwell duration).")

    print("\n=================================================================")
    print(f"Selesai! Output verifikasi tersimpan di:")
    print(f" - {debug_1080p_path} (1920x1080)")
    print(f" - {test_tg_path} (1920x1080)")
    print("Periksa Bot Telegram untuk memastikan foto alert identik dengan live feed.")
    print("=================================================================")


if __name__ == "__main__":
    run_simulation()
