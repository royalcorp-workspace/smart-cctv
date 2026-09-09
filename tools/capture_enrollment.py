#!/usr/bin/env python3
"""
Face Enrollment Capture Tool — Smart CCTV v2
=============================================
Tujuan:
    Merekam crop wajah dari stream CCTV (sudut overhead) ke folder
    `data/captured_faces/` agar operator bisa mengkurasi foto terbaik
    dan memindahkannya ke `data/known_faces/` sebagai data augmentasi.

Cara menjalankan:
    # Dari direktori root proyek:
    python tools/capture_enrollment.py

    # Gunakan video file untuk test (tanpa RTSP):
    python tools/capture_enrollment.py --source path/to/video.mp4

    # Atur interval simpan (default: 1 detik antar-crop per wajah):
    python tools/capture_enrollment.py --interval 0.5

    # Kontrol window:
    Tekan 'q' untuk berhenti.
    Tekan 's' untuk force-save frame saat ini (semua wajah).
    Tekan SPACE untuk pause/resume.

Output:
    data/captured_faces/face_HHMMSS_NNN.jpg
    (NNN = nomor wajah dalam frame jika ada lebih dari 1)
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — pastikan root proyek ada di sys.path
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_PATH = _PROJECT_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
CONFIG_PATH = _PROJECT_ROOT / "cameras" / "cam_01" / "config.json"
OUTPUT_DIR  = _PROJECT_ROOT / "data" / "captured_faces"

# UI colours (BGR)
COLOR_BOX       = (0, 220, 180)   # cyan-teal
COLOR_BOX_SAVED = (0, 255, 80)    # bright green
COLOR_PAUSED    = (0, 120, 255)   # orange
COLOR_TEXT_BG   = (20, 20, 20)
COLOR_TEXT      = (255, 255, 255)
FONT            = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_env(value: str) -> str:
    """Expand ${VAR} placeholders using environment variables."""
    def _replace(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))
    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def _load_source_from_config() -> Optional[str]:
    """Read cam_01 source URL from config.json, expanding env vars."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        source = cfg.get("source", None)
        if source:
            return _resolve_env(str(source))
    except Exception as e:
        print(f"[WARN] Tidak bisa membaca config.json: {e}")
    return None


def _download_yunet(target: Path) -> None:
    """Download YuNet weights if not present."""
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Mengunduh model YuNet ke {target} ...")
    tmp = target.with_suffix(".tmp")
    try:
        req = urllib.request.Request(
            YUNET_URL, headers={"User-Agent": "SmartCCTV-Enrollment/2.0"}
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            tmp.write_bytes(resp.read())
        tmp.rename(target)
        print(f"[INFO] YuNet berhasil diunduh ({target.stat().st_size // 1024} KB).")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(f"Gagal mengunduh YuNet: {e}") from e


def _put_label(
    img: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.55,
    thickness: int = 1,
    color: Tuple[int, int, int] = COLOR_TEXT,
    bg: bool = True,
) -> None:
    """Draw text with solid dark background badge for readability on any frame."""
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = origin
    if bg:
        cv2.rectangle(
            img, (x - 2, y - th - 4), (x + tw + 2, y + baseline), COLOR_TEXT_BG, -1
        )
    cv2.putText(img, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def _draw_status_bar(
    canvas: np.ndarray,
    saved: int,
    total_faces: int,
    paused: bool,
    fps: float,
    source_label: str,
) -> None:
    """Render a semi-transparent HUD status bar at the bottom of the canvas."""
    h, w = canvas.shape[:2]
    bar_h = 38
    overlay = canvas[h - bar_h : h, :].copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
    cv2.addWeighted(
        overlay, 0.75, canvas[h - bar_h : h, :], 0.25, 0, canvas[h - bar_h : h, :]
    )
    status = "[ PAUSE ]" if paused else "[ REC ]"
    status_col = COLOR_PAUSED if paused else (0, 200, 100)
    line = (
        f"  {status}  FPS:{fps:.1f}  Wajah:{total_faces}  "
        f"Tersimpan:{saved}  Sumber: {source_label[:55]}"
        f"  |  [q]=Keluar  [s]=Force-save  [SPACE]=Pause"
    )
    _put_label(
        canvas, line, (6, h - 10),
        font_scale=0.46, thickness=1, color=status_col, bg=False,
    )


# ---------------------------------------------------------------------------
# Minimal YuNet wrapper (no engine imports needed)
# ---------------------------------------------------------------------------

class _SimpleYuNet:
    """Thin YuNet wrapper — optimised for enrollment, no temporal smoothing."""

    def __init__(self, model_path: Path, score_threshold: float = 0.45) -> None:
        self.detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config="",
            input_size=(640, 360),
            score_threshold=score_threshold,
            nms_threshold=0.30,
            top_k=50,
        )
        self._iw = 640
        self._ih = 360

    def detect(self, frame: np.ndarray) -> List[np.ndarray]:
        """Return list of raw YuNet face arrays [x,y,w,h, lm*10, score]."""
        h, w = frame.shape[:2]
        if w != self._iw or h != self._ih:
            self.detector.setInputSize((w, h))
            self._iw, self._ih = w, h
        _, faces = self.detector.detect(frame)
        return [] if faces is None else list(faces)


# ---------------------------------------------------------------------------
# Core capture loop
# ---------------------------------------------------------------------------

def run(source: str, save_interval: float, score_threshold: float) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        _download_yunet(MODEL_PATH)

    detector = _SimpleYuNet(MODEL_PATH, score_threshold=score_threshold)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Tidak bisa membuka sumber: {source}")
        sys.exit(1)

    # Strip credentials from display label
    display_source = re.sub(r"://[^@]+@", "://<creds>@", str(source))

    print(f"\n{'='*64}")
    print(f"  Face Enrollment Capture — Smart CCTV v2")
    print(f"{'='*64}")
    print(f"  Sumber  : {display_source}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"  Interval: {save_interval:.1f}s antar-crop per wajah")
    print(f"  YuNet   : score_threshold={score_threshold}")
    print(f"  Kontrol : [q]=Keluar  [s]=Force-save  [SPACE]=Pause")
    print(f"{'='*64}\n")

    saved_count: int = 0
    paused: bool = False
    fps_t0: float = time.time()
    fps_frames: int = 0
    fps_display: float = 0.0

    # Per-face debounce: bucket centroid to 50px grid -> last save timestamp
    _face_last_saved: Dict[Tuple[int, int], float] = {}

    cv2.namedWindow("Enrollment Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Enrollment Capture", 960, 540)

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
                print(f"[INFO] {'Dijeda' if paused else 'Dilanjutkan'}.")
            elif key == ord("s"):
                _face_last_saved.clear()
                print("[INFO] Force-save: semua wajah berikutnya akan disimpan segera.")

            if paused:
                cv2.waitKey(80)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop untuk video file; reconnect delay untuk RTSP
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue

            now = time.time()
            fps_frames += 1
            if now - fps_t0 >= 1.0:
                fps_display = fps_frames / (now - fps_t0)
                fps_frames = 0
                fps_t0 = now

            # ---- Inference pada downscale 640x360 (hemat CPU) ----
            h_orig, w_orig = frame.shape[:2]
            infer_w, infer_h = 640, 360
            infer = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
            faces = detector.detect(infer)

            # Scale factor kembali ke resolusi asli
            sx = w_orig / infer_w
            sy = h_orig / infer_h

            canvas = frame.copy()
            saved_this_frame: List[int] = []

            for idx, face in enumerate(faces):
                fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                score = float(face[14])

                # Koordinat di resolusi asli
                ox = int(fx * sx)
                oy = int(fy * sy)
                ow = int(fw * sx)
                oh = int(fh * sy)

                # Margin padding:
                # 30% horizontal + 40% vertikal — menangkap dahi & dagu
                # dari sudut overhead agar SFace punya konteks cukup
                margin_x = int(ow * 0.30)
                margin_y = int(oh * 0.40)
                cx1 = max(0, ox - margin_x)
                cy1 = max(0, oy - margin_y)
                cx2 = min(w_orig, ox + ow + margin_x)
                cy2 = min(h_orig, oy + oh + margin_y)

                # Debounce per-wajah (50px grid bucket)
                bucket: Tuple[int, int] = (
                    (ox + ow // 2) // 50,
                    (oy + oh // 2) // 50,
                )
                last_t = _face_last_saved.get(bucket, 0.0)
                should_save = (now - last_t) >= save_interval

                if should_save and (cx2 > cx1) and (cy2 > cy1):
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size > 0:
                        ts = time.strftime("%H%M%S")
                        ms_part = int((now % 1) * 1000)
                        fname = OUTPUT_DIR / f"face_{ts}_{ms_part:03d}_{idx}.jpg"
                        cv2.imwrite(
                            str(fname), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95],
                        )
                        saved_count += 1
                        _face_last_saved[bucket] = now
                        saved_this_frame.append(idx)
                        print(
                            f"[SAVE] {fname.name}  "
                            f"skor={score:.2f}  "
                            f"crop={cx2 - cx1}x{cy2 - cy1}px  "
                            f"total={saved_count}"
                        )

                # Gambar kotak deteksi YuNet
                box_col = COLOR_BOX_SAVED if idx in saved_this_frame else COLOR_BOX
                cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), box_col, 2)

                # Badge skor di atas kotak
                badge = f"YuNet {score:.2f}"
                badge_y = max(oy - 6, 16)
                (bw_px, bh_px), _ = cv2.getTextSize(badge, FONT, 0.52, 1)
                cv2.rectangle(
                    canvas,
                    (ox, badge_y - bh_px - 4),
                    (ox + bw_px + 4, badge_y + 2),
                    (20, 20, 20), -1,
                )
                cv2.putText(
                    canvas, badge, (ox + 2, badge_y),
                    FONT, 0.52, box_col, 1, cv2.LINE_AA,
                )

                # Kotak abu-abu tipis menunjukkan area crop (dengan margin)
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (80, 80, 80), 1)

                if idx in saved_this_frame:
                    _put_label(
                        canvas, "SAVED",
                        (ox + 2, oy + oh - 6),
                        font_scale=0.55, color=COLOR_BOX_SAVED,
                    )

            _draw_status_bar(
                canvas, saved_count, len(faces), paused, fps_display, display_source
            )
            cv2.imshow("Enrollment Capture", canvas)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Session selesai. Total crop tersimpan: {saved_count}")
        print(f"       Lokasi folder: {OUTPUT_DIR}")
        print(
            f"\n  Langkah selanjutnya:\n"
            f"  1. Buka folder output: {OUTPUT_DIR}\n"
            f"  2. Pilih crop wajah terbaik per karyawan (cari yang paling jelas)\n"
            f"  3. Rename file menjadi: NAMA_2.jpg  (contoh: ALGHANY_2.jpg)\n"
            f"  4. Pindahkan ke: {_PROJECT_ROOT / 'data' / 'known_faces'}\n"
            f"  5. Restart main.py — embedding cache akan rebuild otomatis\n"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture face crops dari stream CCTV untuk enrollment dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", "-s",
        default=None,
        help=(
            "Sumber video: RTSP URL, path file video, atau indeks kamera (0, 1, ...)."
            " Default: baca 'source' dari cameras/cam_01/config.json."
        ),
    )
    p.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Jeda minimum (detik) antar-penyimpanan crop wajah di lokasi yang sama (default: 1.0).",
    )
    p.add_argument(
        "--score", "-t",
        type=float,
        default=0.45,
        help="YuNet detection confidence threshold (default: 0.45). Lebih rendah = lebih agresif.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    source = args.source
    if source is None:
        source = _load_source_from_config()
    if source is None:
        print(
            "[ERROR] Sumber video tidak ditemukan.\n"
            "        Pastikan env vars CAM01_USER / CAM01_PASS / CAM01_IP / CAM01_PORT sudah di-set,\n"
            "        atau gunakan: python tools/capture_enrollment.py --source 0"
        )
        sys.exit(1)

    # Coba konversi ke integer (indeks webcam)
    try:
        source = int(source)
    except (TypeError, ValueError):
        pass

    run(source=source, save_interval=args.interval, score_threshold=args.score)
