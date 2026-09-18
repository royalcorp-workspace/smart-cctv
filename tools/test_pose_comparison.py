"""Standalone benchmark comparing YOLOv11n standard detection vs YOLOv11n-pose for seated/occluded persons."""
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Project root setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from engine.config_loader import load_camera_config

CONFIG_PATH = BASE_DIR / "cameras" / "cam_01" / "config.json"
OUTPUT_DIR = BASE_DIR / "storage"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG = OUTPUT_DIR / "pose_comparison_result.jpg"


def get_test_frame():
    """Attempt to capture live frame from local web stream, then RTSP, fallback to latest cam_01 office snapshot."""
    # 1. Try reading from running local MJPEG stream
    try:
        cap_web = cv2.VideoCapture("http://127.0.0.1:8000/stream/cam_01")
        if cap_web.isOpened():
            ret, frame = cap_web.read()
            cap_web.release()
            if ret and frame is not None:
                print("[OK] Berhasil menangkap live frame langsung dari Web Stream cam_01!")
                return frame, "Live Web Stream cam_01"
    except Exception:
        pass

    # 2. Try direct RTSP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;3000000"
    cfg = load_camera_config(CONFIG_PATH)
    rtsp_url = cfg.get("source", "")

    if rtsp_url:
        print("[INFO] Menghubungkan ke RTSP cam_01...")
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            for _ in range(8):
                ret, tmp = cap.read()
                if ret and tmp is not None:
                    cap.release()
                    print("[OK] Berhasil menangkap live frame langsung dari RTSP cam_01!")
                    return tmp, "Live RTSP cam_01"
            cap.release()

    # 3. Fallback to latest cam_01 office snapshot
    print("[WARN] RTSP/WebStream tidak merespon. Mencari snapshot kantor cam_01...")
    cam01_snaps = sorted(
        list((BASE_DIR / "storage").glob("*cam01*.jpg")) + list((BASE_DIR / "cameras" / "cam_01" / "snapshots").glob("*.jpg")),
        key=lambda p: p.stat().st_mtime,
    )
    if cam01_snaps:
        latest = cam01_snaps[-1]
        print(f"[INFO] Menggunakan snapshot kantor cam_01: {latest.name}")
        img = cv2.imread(str(latest))
        if img is not None:
            return img, latest.name

    print("[WARN] Tidak ada snapshot cam_01, mencari kandidat apapun di storage...")
    candidates = list((BASE_DIR / "storage").glob("*.jpg"))
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return cv2.imread(str(latest)), latest.name

    return np.full((360, 640, 3), 40, dtype=np.uint8), "Mock Canvas"


def run_benchmark():
    print("=" * 70)
    print("   BENCHMARK EVALUASI: YOLOv11n (DETEKSI) VS YOLOv11n-POSE (RANGKA)   ")
    print("=" * 70)

    frame, source_info = get_test_frame()
    orig_h, orig_w = frame.shape[:2]
    # Resize to 640x360 inference resolution
    infer_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)

    # 1. Standard YOLOv11n (Object Detection)
    print("\n[1/2] Menyiapkan Model Standar: yolo11n.pt...")
    model_det = YOLO("yolo11n.pt")

    # Warmup
    _ = model_det(infer_frame, verbose=False)

    # Benchmark Deteksi
    t0 = time.perf_counter()
    results_det = model_det(infer_frame, conf=0.20, verbose=False)[0]
    lat_det = (time.perf_counter() - t0) * 1000.0

    # Filter person only (class 0)
    det_boxes = []
    if results_det.boxes is not None:
        for b in results_det.boxes:
            cls_id = int(b.cls[0].item())
            conf_val = float(b.conf[0].item())
            if cls_id == 0:  # person
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                det_boxes.append((xyxy, conf_val))

    # 2. YOLOv11n-pose (Pose Estimation)
    print("\n[2/2] Menyiapkan Model Pose: yolo11n-pose.pt...")
    model_pose = YOLO("yolo11n-pose.pt")

    # Warmup
    _ = model_pose(infer_frame, verbose=False)

    # Benchmark Pose
    t0 = time.perf_counter()
    results_pose = model_pose(infer_frame, conf=0.20, verbose=False)[0]
    lat_pose = (time.perf_counter() - t0) * 1000.0

    pose_persons = []
    if results_pose.boxes is not None:
        for idx_p, b in enumerate(results_pose.boxes):
            conf_val = float(b.conf[0].item())
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            kpts = None
            if results_pose.keypoints is not None and idx_p < len(results_pose.keypoints):
                kpts = results_pose.keypoints[idx_p].data[0].cpu().numpy()
            pose_persons.append((xyxy, conf_val, kpts))

    # 3. Visual Rendering Comparison
    panel_det = infer_frame.copy()
    panel_pose = infer_frame.copy()

    # Draw Standard Detection Panel
    for (x1, y1, x2, y2), cval in det_boxes:
        cv2.rectangle(panel_det, (x1, y1), (x2, y2), (0, 255, 127), 2)
        lbl = f"PERSON {cval:.2f}"
        cv2.putText(panel_det, lbl, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 127), 1, cv2.LINE_AA)

    cv2.putText(panel_det, f"STANDAR: yolo11n ({len(det_boxes)} orang, {lat_det:.1f} ms)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # Draw Pose Panel
    for (x1, y1, x2, y2), cval, kpts in pose_persons:
        cv2.rectangle(panel_pose, (x1, y1), (x2, y2), (255, 200, 0), 2)
        lbl = f"POSE {cval:.2f}"
        cv2.putText(panel_pose, lbl, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)
        
        # Draw keypoints (head/ears/shoulders)
        if kpts is not None:
            # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear, 5: left_shoulder, 6: right_shoulder
            for k_idx in range(min(7, len(kpts))):
                kx, ky, kconf = kpts[k_idx]
                if kconf > 0.3:
                    cv2.circle(panel_pose, (int(kx), int(ky)), 3, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.putText(panel_pose, f"POSE: yolo11n-pose ({len(pose_persons)} orang, {lat_pose:.1f} ms)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    # Combine side-by-side
    combined = np.hstack([panel_det, panel_pose])
    cv2.imwrite(str(OUTPUT_IMG), combined)
    print(f"\n[HASIL VISUAL] Gambar komparasi berhasil disimpan di:\n  -> {OUTPUT_IMG}")

    # 4. Summary Table
    print("\n" + "=" * 70)
    print(f"{'PARAMETER EVALUASI':<30} | {'YOLOv11n (STANDAR)':<18} | {'YOLOv11n-POSE':<18}")
    print("-" * 70)
    print(f"{'Jumlah Orang Terdeteksi':<30} | {len(det_boxes):<18} | {len(pose_persons):<18}")
    print(f"{'Latensi Komputasi (ms)':<30} | {lat_det:<18.1f} | {lat_pose:<18.1f}")
    fps_det_est = 1000.0 / max(1.0, lat_det)
    fps_pose_est = 1000.0 / max(1.0, lat_pose)
    print(f"{'Estimasi Throughput (FPS)':<30} | ~{fps_det_est:<17.1f} | ~{fps_pose_est:<17.1f}")
    diff_count = len(pose_persons) - len(det_boxes)
    if diff_count > 0:
        print(f"\n[KESIMPULAN] Model Pose mendeteksi +{diff_count} orang lebih banyak dibanding model standar!")
    elif diff_count == 0:
        print(f"\n[KESIMPULAN] Kedua model mendeteksi jumlah orang yang sama ({len(det_boxes)} orang).")
    else:
        print(f"\n[KESIMPULAN] Model standar mendeteksi {abs(diff_count)} objek lebih banyak.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
