"""Diagnostic Tool: Capture camera frame and run raw YOLO11n inference at conf=0.15.

Prints all detections and identifies what is detected inside zone_2_transit.
"""

import json
from pathlib import Path
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Workspace Paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
CONFIG_PATH = BASE_DIR / "cameras" / "cam_01" / "config.json"
ROI_PATH = BASE_DIR / "cameras" / "cam_01" / "roi_zones.json"
MODEL_DIR = BASE_DIR / "yolo11n_openvino_model"
SCRATCH_DIR = BASE_DIR / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)


def get_camera_frame() -> Tuple[np.ndarray, str]:
    """Attempt to capture live frame from camera RTSP via TCP, fallback to latest snapshot."""
    import os
    from engine.config_loader import load_camera_config

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cfg = load_camera_config(CONFIG_PATH)
    rtsp_url = cfg.get("source", "")

    print(f"[INFO] Connecting to camera RTSP with TCP transport...")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = None
    if cap.isOpened():
        # Read frames to reach current live frame
        for _ in range(10):
            ret, tmp = cap.read()
            if ret and tmp is not None:
                frame = tmp
        cap.release()

    if frame is not None:
        print("[OK] Live RTSP frame captured successfully from camera!")
        return frame, "Live Camera Stream (RTSP TCP)"

    print("[WARN] Could not capture live RTSP stream. Searching for latest saved snapshot...")
    # Check storage/ or snapshots/
    candidates = list((BASE_DIR / "storage").glob("*.jpg")) + list((BASE_DIR / "cameras" / "cam_01" / "snapshots").glob("*.jpg"))
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        print(f"[INFO] Loading latest snapshot: {latest.name}")
        img = cv2.imread(str(latest))
        if img is not None:
            return img, str(latest.name)

    print("[ERROR] No frame could be retrieved.")
    sys.exit(1)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Raw YOLO11n Detection Diagnostic Tool")
    parser.add_argument("--image", type=str, default="", help="Path to image snapshot (optional, defaults to live camera)")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default 0.15)")
    args = parser.parse_args()

    print("=" * 70)
    print(f"      YOLO11n RAW DETECTION DIAGNOSTIC (CONF={args.conf})       ")
    print("=" * 70)

    # 1. Load ROI Zones
    with open(ROI_PATH, "r", encoding="utf-8") as f:
        zones = json.load(f)
    zone_2_pts = zones.get("zone_2_transit", [])
    zone_2_poly = np.array(zone_2_pts, dtype=np.int32).reshape((-1, 1, 2))

    # 2. Get Frame
    if args.image and Path(args.image).exists():
        raw_frame = cv2.imread(args.image)
        source_info = f"Snapshot File: {args.image}"
    else:
        raw_frame, source_info = get_camera_frame()

    if raw_frame.shape[1] != 640 or raw_frame.shape[0] != 480:
        frame = cv2.resize(raw_frame, (640, 480), interpolation=cv2.INTER_AREA)
    else:
        frame = raw_frame.copy()

    # 3. Load YOLO11n model
    model_path = str(MODEL_DIR) if MODEL_DIR.exists() else str(BASE_DIR / "yolo11n.pt")
    print(f"[INFO] Loading model from: {model_path} ...")
    model = YOLO(model_path, task="detect")

    # 4. Run raw inference at specified conf across ALL classes (not filtered)
    print(f"[INFO] Running raw inference with conf={args.conf} across ALL classes...")
    results = model(frame, imgsz=640, conf=args.conf, verbose=False)

    annotated = frame.copy()
    # Draw zone_2_transit polygon
    cv2.polylines(annotated, [zone_2_poly], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.putText(annotated, "zone_2_transit", (zone_2_pts[0][0], zone_2_pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    all_detections = []
    zone_2_detections = []

    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        names = results[0].names

        for box, conf_val, cls_val in zip(xyxy, confs, classes):
            cid = int(cls_val)
            cname = names.get(cid, f"class_{cid}")
            conf_f = float(conf_val)
            x1, y1, x2, y2 = [int(v) for v in box]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w // 2
            cy = y1 + h // 2

            # Point polygon test for zone_2_transit (feet or center)
            dist_center = float(cv2.pointPolygonTest(zone_2_poly, (float(cx), float(cy)), True))
            dist_bottom = float(cv2.pointPolygonTest(zone_2_poly, (float(cx), float(y2)), True))

            in_zone_2 = dist_center >= -5.0 or dist_bottom >= -5.0

            det_info = {
                "class_id": cid,
                "class_name": cname,
                "confidence": conf_f,
                "bbox": (x1, y1, w, h),
                "center": (cx, cy),
                "dist_center_to_zone2": dist_center,
                "in_zone_2": in_zone_2,
            }
            all_detections.append(det_info)
            if in_zone_2:
                zone_2_detections.append(det_info)

            # Draw on annotated image
            color = (0, 0, 255) if in_zone_2 else (255, 165, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            lbl = f"{cname} {conf_f:.2f}"
            cv2.putText(annotated, lbl, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

    print("\n" + "=" * 70)
    print(f"Frame Source: {source_info}")
    print(f"Total Raw Detections Found in Entire Frame : {len(all_detections)}")
    print(f"Detections inside / touching zone_2_transit : {len(zone_2_detections)}")
    print("=" * 70)

    if zone_2_detections:
        print("\n--- DETEKSI DI AREA ZONE_2_TRANSIT ---")
        for i, d in enumerate(zone_2_detections, 1):
            print(f"[{i}] Class: {d['class_name']} (ID {d['class_id']}) | Conf: {d['confidence']:.3f} | BBox: {d['bbox']} | Center: {d['center']} | Dist to Poly: {d['dist_center_to_zone2']:+.1f}px")
    else:
        print("\n[!] TIDAK ADA deteksi dengan conf >= 0.15 di area zone_2_transit.")

    if all_detections:
        print("\n--- SEMUA DETEKSI LAINNYA DI FRAME ---")
        for i, d in enumerate(all_detections, 1):
            if not d['in_zone_2']:
                print(f"[{i}] Class: {d['class_name']} (ID {d['class_id']}) | Conf: {d['confidence']:.3f} | BBox: {d['bbox']} | Center: {d['center']}")

    # Save output image
    out_path = SCRATCH_DIR / "debug_detect_bag.jpg"
    cv2.imwrite(str(out_path), annotated)
    print("\n" + "=" * 70)
    print(f"[OK] Visual annotated frame saved to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
