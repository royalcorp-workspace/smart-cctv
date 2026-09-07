"""Tool to render and inspect ROI zones on actual camera frame.
Draws polygons and labels each vertex index to detect self-intersecting or misaligned perimeter points.
Saves rendered image to storage/debug_zones_clean.jpg.
"""

import json
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.config_loader import load_camera_config
from engine.rtsp_stream import ThreadedCapture


def debug_render_zones():
    print("=================================================================")
    print("        INSPEKSI & VALIDASI KOORDINAT ROI ZONES                  ")
    print("=================================================================")

    cam_dir = PROJECT_ROOT / "cameras" / "cam_01"
    roi_file = cam_dir / "roi_zones.json"
    config_file = cam_dir / "config.json"

    with open(roi_file, "r") as f:
        zones_data = json.load(f)

    base_res = zones_data.get("base_resolution", [1920, 1080])
    target_w, target_h = base_res[0], base_res[1]

    print(f"Base Resolution: {target_w}x{target_h}")
    print("\nKoordinat Zona Saat Ini:")
    for zid, pts in zones_data.items():
        if zid == "base_resolution" or zid.startswith("_"):
            continue
        print(f" - [{zid}]: {pts}")

    # 1. Obtain real frame
    config = load_camera_config(config_file)
    source = config.get("source")
    frame = None

    if source:
        print(f"\nMengambil 1 frame riil dari kamera RTSP: {source}...")
        try:
            cap = ThreadedCapture(source=source)
            cap.start()
            for _ in range(30):
                ret, f = cap.read(timeout=0.3)
                if ret and f is not None:
                    frame = f
                    break
            cap.stop()
        except Exception as e:
            print(f"Peringatan koneksi RTSP: {e}")

    if frame is None:
        print("Mencari frame cadangan dari storage/...")
        snapshots = sorted(Path("storage").glob("*.jpg"), key=lambda x: x.stat().st_mtime)
        if snapshots:
            latest = snapshots[-1]
            print(f"Menggunakan snapshot lokal: {latest}")
            frame = cv2.imread(str(latest))
        else:
            print("Membuat frame kanvas sintetis...")
            frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Normalize to 1920x1080
    if frame.shape[1] != target_w or frame.shape[0] != target_h:
        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    annotated = frame.copy()

    # Distinct colors per zone (BGR)
    colors = {
        "zone_1_koridor": (0, 255, 0),    # Bright Green
        "zone_2_transit": (0, 140, 255),  # Orange
    }
    default_color = (255, 0, 255)

    scale_x = target_w / float(base_res[0])
    scale_y = target_h / float(base_res[1])

    for zid, pts in zones_data.items():
        if zid == "base_resolution" or zid.startswith("_"):
            continue
        if not isinstance(pts, list) or len(pts) < 3:
            continue

        color = colors.get(zid, default_color)
        scaled_pts = []
        for idx, pt in enumerate(pts):
            px = int(round(pt[0] * scale_x))
            py = int(round(pt[1] * scale_y))
            scaled_pts.append([px, py])

            # Draw vertex marker and index
            cv2.circle(annotated, (px, py), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"P{idx}({px},{py})",
                (px + 8, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"P{idx}({px},{py})",
                (px + 8, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                lineType=cv2.LINE_AA,
            )

        poly_arr = np.array(scaled_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [poly_arr], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        # Label zone name at centroid
        cx = int(np.mean([p[0] for p in scaled_pts]))
        cy = int(np.mean([p[1] for p in scaled_pts]))
        cv2.putText(
            annotated,
            f"[{zid}]",
            (cx - 60, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            lineType=cv2.LINE_AA,
        )

    out_path = PROJECT_ROOT / "storage" / "debug_zones_clean.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    print(f"\n[SUKSES] Hasil visualisasi poligon disimpan ke: {out_path}")
    print("=================================================================")


if __name__ == "__main__":
    debug_render_zones()
