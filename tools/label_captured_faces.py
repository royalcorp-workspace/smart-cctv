#!/usr/bin/env python3
"""
Interactive Face Labeling Helper — Smart CCTV v2
================================================
Membantu operator menyortir, memfilter, dan melabeli ratusan foto hasil
`tools/capture_enrollment.py` dari `data/captured_faces/` ke `data/known_faces/<Nama>/`.

Fitur Utama:
1. Pre-Filtering Otomatis:
   - Eliminasi artefak mikro (< 35x35 px).
   - Eliminasi foto buram / motion blur (Laplacian variance < 40.0).
2. Pemetaan Identitas Dinamis:
   - Scan subfolder identitas di `data/known_faces/`.
   - Petakan otomatis ke tombol angka [1]..[9].
3. GUI Review Interaktif (OpenCV Window):
   - Menampilkan preview crop wajah resolusi nyaman dengan OSD panduan.
   - Progress bar dan informasi dimensi / skor ketajaman.
4. Pemindahan Instan & Cache Invalidation:
   - Tekan [1..9]: Pindahkan (move) ke folder identitas target.
   - Tekan [d / Spasi]: Hapus file sampah.
   - Tekan [q / ESC]: Keluar dan simpan progress.
   - Otomatis menghapus `data/known_faces/.embeddings_cache.npz` agar SFace
     langsung mengindeks embedding baru saat CCTV dijalankan.
"""

import argparse
import datetime
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Bootstrap project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from engine.face_recognizer import FaceRecognizer


def compute_laplacian_variance(img: np.ndarray) -> float:
    """Hitung skor fokus/ketajaman menggunakan variansi operator Laplacian."""
    if img is None or img.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        var_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return var_score
    except Exception:
        return 0.0


def scan_known_identities(known_dir: Path) -> List[str]:
    """Pindai folder data/known_faces/ untuk mendapatkan daftar identitas unik."""
    identities: List[str] = []
    if not known_dir.exists():
        known_dir.mkdir(parents=True, exist_ok=True)
        return identities

    # 1. Prioritaskan nama subfolder
    subdirs = [p for p in known_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    for d in subdirs:
        clean_name = FaceRecognizer.normalize_name(d.name)
        if clean_name and clean_name.lower() != "unknown" and clean_name not in identities:
            identities.append(clean_name)

    # 2. Fallback: file datar di root known_faces
    if not identities:
        flat_files = [
            p for p in known_dir.iterdir()
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith(".")
        ]
        for f in flat_files:
            clean_name = FaceRecognizer.normalize_name(f.stem)
            if clean_name and clean_name.lower() != "unknown" and clean_name not in identities:
                identities.append(clean_name)

    identities.sort()
    return identities


def pre_filter_captured_faces(
    captured_dir: Path,
    min_size: int = 35,
    min_blur: float = 40.0,
    auto_delete_trash: bool = True,
) -> Tuple[List[Path], int, int]:
    """Pindai dan eliminasi file sampah sebelum review manual dimulai.

    Returns:
        (valid_files, count_small, count_blur)
    """
    valid_extensions = {".jpg", ".jpeg", ".png"}
    all_files = [
        p for p in sorted(captured_dir.glob("*"))
        if p.is_file() and p.suffix.lower() in valid_extensions and not p.name.startswith(".")
    ]

    valid_files: List[Path] = []
    count_small = 0
    count_blur = 0

    for fpath in all_files:
        try:
            img = cv2.imread(str(fpath))
            if img is None or img.size == 0:
                count_small += 1
                if auto_delete_trash:
                    fpath.unlink(missing_ok=True)
                continue

            h, w = img.shape[:2]
            if w < min_size or h < min_size:
                count_small += 1
                if auto_delete_trash:
                    fpath.unlink(missing_ok=True)
                continue

            blur_var = compute_laplacian_variance(img)
            if blur_var < min_blur:
                count_blur += 1
                if auto_delete_trash:
                    fpath.unlink(missing_ok=True)
                continue

            valid_files.append(fpath)

        except Exception as e:
            print(f"[WARN] Gagal memvalidasi {fpath.name}: {e}")
            count_small += 1
            if auto_delete_trash:
                fpath.unlink(missing_ok=True)

    return valid_files, count_small, count_blur


def render_preview_canvas(
    img: np.ndarray,
    file_path: Path,
    index: int,
    total: int,
    identities: List[str],
    blur_score: float,
    orig_size: Tuple[int, int],
) -> np.ndarray:
    """Render GUI slate dark-mode (820x520) berisi preview crop & panel OSD kontrol."""
    canvas_w = 820
    canvas_h = 520
    canvas = np.full((canvas_h, canvas_w, 3), (28, 17, 11), dtype=np.uint8)  # Dark slate background

    # 1. Header Bar (Y: 0..65)
    cv2.rectangle(canvas, (0, 0), (canvas_w, 65), (45, 30, 22), -1)
    cv2.line(canvas, (0, 65), (canvas_w, 65), (75, 55, 42), 1, cv2.LINE_AA)

    title_text = "FACE LABELING HELPER — SMART CCTV v2"
    cv2.putText(canvas, title_text, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 220, 180), 2, cv2.LINE_AA)

    # Progress badge (Right header)
    pct = int((index / total) * 100) if total > 0 else 100
    prog_text = f"Foto {index}/{total} ({pct}%)"
    (pw, _), _ = cv2.getTextSize(prog_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(canvas, prog_text, (canvas_w - pw - 24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. Left Panel: Face Crop View Box (X: 24..384, Y: 85..445 -> 360x360 px)
    box_x = 24
    box_y = 85
    box_size = 360
    cv2.rectangle(canvas, (box_x, box_y), (box_x + box_size, box_y + box_size), (18, 12, 8), -1)
    cv2.rectangle(canvas, (box_x, box_y), (box_x + box_size, box_y + box_size), (75, 55, 42), 2, cv2.LINE_AA)

    # Fit image proportionally into 350x350 inside box
    ih, iw = img.shape[:2]
    target_box = 340
    scale = min(target_box / float(iw), target_box / float(ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA)

    off_x = box_x + (box_size - nw) // 2
    off_y = box_y + (box_size - nh) // 2
    canvas[off_y : off_y + nh, off_x : off_x + nw] = resized

    # Image metadata footer under preview box
    meta_str = f"Dim: {orig_size[0]}x{orig_size[1]} px | Blur: {blur_score:.1f}"
    cv2.putText(canvas, meta_str, (box_x + 10, box_y + box_size + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 200, 200), 1, cv2.LINE_AA)
    fname_display = file_path.name if len(file_path.name) <= 35 else (file_path.name[:32] + "...")
    cv2.putText(canvas, fname_display, (box_x + 10, box_y + box_size + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 140, 140), 1, cv2.LINE_AA)

    # 3. Right Panel: Dynamic Identity Keybindings (X: 410..796)
    panel_rx = 410
    panel_ry = 85
    cv2.putText(canvas, "PILIH IDENTITAS PEMILIK WAJAH:", (panel_rx, panel_ry + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 220, 100), 2, cv2.LINE_AA)

    line_y = panel_ry + 45
    for idx, name in enumerate(identities[:9], start=1):
        # Key button pill
        key_label = f"[{idx}]"
        cv2.putText(canvas, key_label, (panel_rx, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 180), 2, cv2.LINE_AA)
        cv2.putText(canvas, name, (panel_rx + 45, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        line_y += 32

    # Action Keys Separator
    line_y = max(line_y + 10, 360)
    cv2.line(canvas, (panel_rx, line_y), (canvas_w - 24, line_y), (55, 45, 35), 1, cv2.LINE_AA)
    line_y += 28

    cv2.putText(canvas, "[D / Space]  Hapus / Lewati (Sampah)", (panel_rx, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (80, 110, 240), 2, cv2.LINE_AA)
    line_y += 28
    cv2.putText(canvas, "[Q / ESC]    Simpan Progress & Selesai", (panel_rx, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 240), 2, cv2.LINE_AA)

    # Progress bar at bottom
    bar_y1 = canvas_h - 10
    bar_y2 = canvas_h
    cv2.rectangle(canvas, (0, bar_y1), (canvas_w, bar_y2), (35, 25, 20), -1)
    if total > 0:
        fill_w = int((index / total) * canvas_w)
        cv2.rectangle(canvas, (0, bar_y1), (fill_w, bar_y2), (0, 220, 180), -1)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart CCTV v2 — Interactive Face Labeling Helper")
    parser.add_argument(
        "--captured-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "captured_faces"),
        help="Direktori sumber crop hasil enrollment (default: data/captured_faces)",
    )
    parser.add_argument(
        "--known-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "known_faces"),
        help="Direktori target dataset referensi (default: data/known_faces)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=35,
        help="Ambang batas minimal lebar/tinggi crop wajah (default: 35 px)",
    )
    parser.add_argument(
        "--min-blur",
        type=float,
        default=40.0,
        help="Ambang batas minimal variansi Laplacian ketajaman (default: 40.0)",
    )
    parser.add_argument(
        "--keep-trash",
        action="store_true",
        help="Jangan hapus file yang gagal pre-filter (default: otomatis dihapus)",
    )
    args = parser.parse_args()

    captured_dir = Path(args.captured_dir).resolve()
    known_dir = Path(args.known_dir).resolve()

    print("\n" + "=" * 64)
    print("      INTERACTIVE FACE LABELING HELPER — SMART CCTV v2      ")
    print("=" * 64)
    print(f"  Direktori Captured : {captured_dir}")
    print(f"  Direktori Target   : {known_dir}")
    print(f"  Filter Min Size    : >= {args.min_size}x{args.min_size} px")
    print(f"  Filter Min Blur    : Laplacian var >= {args.min_blur:.1f}")
    print("=" * 64 + "\n")

    if not captured_dir.exists():
        print(f"[INFO] Direktori {captured_dir} belum ada. Membuat direktori...")
        captured_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pindai identitas terdaftar di data/known_faces/
    identities = scan_known_identities(known_dir)
    if not identities:
        print("[WARN] Belum ada subfolder karyawan di data/known_faces/.")
        print("       Silakan buat minimal 1 subfolder nama karyawan terlebih dahulu,")
        print("       contoh: data/known_faces/Alghany/")
        sys.exit(0)

    print(f"[INFO] Terdeteksi {len(identities)} identitas karyawan terdaftar:")
    for idx, name in enumerate(identities[:9], start=1):
        print(f"   [{idx}] {name}")
    print()

    # 2. Jalankan Pre-Filtering Otomatis
    print("[1/2] Menjalankan Pre-Filtering otomatis pada file crop...")
    auto_delete = not args.keep_trash
    valid_files, count_small, count_blur = pre_filter_captured_faces(
        captured_dir=captured_dir,
        min_size=args.min_size,
        min_blur=args.min_blur,
        auto_delete_trash=auto_delete,
    )

    total_trash = count_small + count_blur
    action_trash = "dieliminasi/dihapus otomatis" if auto_delete else "dilewati"
    print(f"   - File terlalu kecil (< {args.min_size}px) : {count_small} {action_trash}")
    print(f"   - File buram (Laplacian < {args.min_blur}) : {count_blur} {action_trash}")
    print(f"   - Total file sampah disaring     : {total_trash}")
    print(f"   - File LOLOS kualifikasi review  : {len(valid_files)}")

    if not valid_files:
        print("\n[INFO] Tidak ada file crop yang perlu direview di data/captured_faces/.")
        print("       Jalankan `python tools/capture_enrollment.py` terlebih dahulu untuk merekam crop.")
        sys.exit(0)

    # 3. Mulai Sesi Review Interaktif GUI
    print("\n[2/2] Membuka sesi review interaktif. Periksa jendela gambar...")
    print("   Kontrol keyboard:")
    print("   - Tekan [1..9] : Pindahkan foto ke nama karyawan bersangkutan")
    print("   - Tekan [D/Spasi]: Hapus foto (sampah/bukan wajah)")
    print("   - Tekan [Q/ESC]  : Selesai & Simpan progress\n")

    win_name = "Face Labeling Helper - Smart CCTV v2"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 820, 520)

    stats: Dict[str, int] = {name: 0 for name in identities}
    deleted_during_review = 0
    total_valid = len(valid_files)
    files_moved = False

    try:
        for idx, fpath in enumerate(valid_files, start=1):
            if not fpath.exists():
                continue

            img = cv2.imread(str(fpath))
            if img is None:
                continue

            orig_h, orig_w = img.shape[:2]
            blur_score = compute_laplacian_variance(img)

            canvas = render_preview_canvas(
                img=img,
                file_path=fpath,
                index=idx,
                total=total_valid,
                identities=identities,
                blur_score=blur_score,
                orig_size=(orig_w, orig_h),
            )
            cv2.imshow(win_name, canvas)

            while True:
                key = cv2.waitKey(0) & 0xFF

                # Exit
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[INFO] Sesi review dihentikan oleh operator.")
                    raise KeyboardInterrupt

                # Delete / Skip
                elif key in (ord("d"), ord("D"), ord(" "), 8, 127):  # 'd', Space, Backspace, Delete
                    fpath.unlink(missing_ok=True)
                    deleted_during_review += 1
                    print(f"[{idx}/{total_valid}] Dihapus (sampah): {fpath.name}")
                    break

                # Mapped numbers [1..9]
                elif ord("1") <= key <= ord("9"):
                    num = key - ord("1")
                    if num < len(identities):
                        target_name = identities[num]
                        target_folder = known_dir / target_name
                        target_folder.mkdir(parents=True, exist_ok=True)

                        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                        new_filename = f"cctv_crop_{now_str}.jpg"
                        target_path = target_folder / new_filename

                        shutil.move(str(fpath), str(target_path))
                        stats[target_name] += 1
                        files_moved = True
                        print(f"[{idx}/{total_valid}] [SIMPAN] -> {target_name}: {new_filename}")
                        break

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()

    # 4. Invalidate SFace Embeddings Cache if any files were moved
    cache_path = known_dir / ".embeddings_cache.npz"
    if files_moved:
        if cache_path.exists():
            try:
                cache_path.unlink()
                print(f"\n[CACHE] File cache {cache_path.name} otomatis DIHAPUS (invalidated).")
                print("        SFace akan otomatis meregenerasi embedding baru saat CCTV dijalankan.")
            except Exception as e:
                print(f"[WARN] Gagal menghapus cache file: {e}")
        else:
            print("\n[CACHE] Cache .embeddings_cache.npz bersih. Siap re-index.")

    # 5. Rekap Statistik
    total_processed = sum(stats.values()) + deleted_during_review
    remaining_files = len(list(captured_dir.glob("*.jpg"))) + len(list(captured_dir.glob("*.png")))

    print("\n" + "=" * 64)
    print("                  REKAP HASIL LABELING                  ")
    print("=" * 64)
    for name, count in stats.items():
        if count > 0:
            print(f"   [+] {name:<22}: {count:>3} foto ditambahkan")
    print("-" * 64)
    print(f"   Total foto berhasil dilabeli : {sum(stats.values()):>3} foto")
    print(f"   Total foto dihapus (review)  : {deleted_during_review:>3} foto")
    print(f"   Total sampah pre-filter      : {total_trash:>3} foto")
    print(f"   Sisa foto di captured_faces  : {remaining_files:>3} foto")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
