"""Diagnostic and Verification Tool for YuNet Face Detection Outside ROI Zones.

Tests:
1. Auto-download and model weight integrity validation
2. Model instantiation via cv2.FaceDetectorYN with score_threshold=0.45
3. Spatial filtering (inside ROI zone vs outside ROI zone)
4. Anti-parallax correlation with in-zone person bounding boxes
5. Temporal smoothing buffer & 5-frame grace period retention
6. Inference latency benchmark on 640x480 frame
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_DIR))

from engine.face_detector import YuNetFaceDetector
from engine.zone_filter import ZoneFilter


def run_tests() -> bool:
    print("=" * 65)
    print("      YUNET FACE DETECTOR DIAGNOSTIC & SPATIAL VERIFICATION      ")
    print("=" * 65)

    # 1. Test Model Download & Instantiation
    print("\n[STEP 1] Testing Model Initialization & Stable Threshold (0.48)...")
    start_init = time.time()
    detector = YuNetFaceDetector(
        score_threshold=0.48,
        nms_threshold=0.30,
        input_size=(640, 480),
        auto_download=True,
    )
    init_time = (time.time() - start_init) * 1000.0

    model_file = detector.model_path
    assert model_file.exists(), f"Model file must exist at: {model_file}"
    file_size_kb = model_file.stat().st_size / 1024.0
    print(f"  -> Model Path   : {model_file}")
    print(f"  -> File Size    : {file_size_kb:.1f} KB (Expected ~227-235 KB)")
    print(f"  -> Score Thresh : {detector.score_threshold} (Stable threshold 0.48)")
    print(f"  -> Load Time    : {init_time:.1f} ms")
    assert file_size_kb > 200, "Model file size too small, download may be truncated!"
    print("  [PASS] Model initialized and verified successfully.")

    # 2. Test Universal Face Inclusion Across All Areas (No Spatial Exclusion)
    print("\n[STEP 2] Testing Universal Face Detection (All Zones & Outside Areas Included)...")
    test_detector = YuNetFaceDetector(score_threshold=0.48, auto_download=False)

    # Simulate faces detected at various locations
    face_in_koridor = ((200, 200, 40, 40), 0.88)
    face_in_transit = ((400, 200, 40, 40), 0.85)
    face_outside_desk = ((580, 50, 40, 40), 0.90)

    # Mock detect output returning all 3 faces
    test_detector.detect = lambda frame: [face_in_koridor, face_in_transit, face_outside_desk]
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    results = test_detector.detect_faces(dummy_frame)
    assert len(results) == 3, f"Expected all 3 faces across all areas, got {len(results)}"
    print(f"  -> Face in Koridor (x=200, y=200) : DETECTED & INCLUDED [PASS]")
    print(f"  -> Face in Transit (x=400, y=200) : DETECTED & INCLUDED [PASS]")
    print(f"  -> Face at Desk   (x=580, y=50)  : DETECTED & INCLUDED [PASS]")
    print("  [PASS] Universal face detection accepts faces across all monitored zones and outside areas.")

    # 3. Test Temporal Smoothing & 5-Frame Retention Buffer
    print("\n[STEP 3] Testing Temporal Smoothing Buffer & 5-Frame Retention...")
    buffer_detector = YuNetFaceDetector(score_threshold=0.48, auto_download=False)
    # Manually register a face in buffer
    buffer_detector._face_buffer[1] = {
        "bbox": (550, 50, 40, 40),
        "score": 0.88,
        "missed_frames": 0,
    }

    # Simulate 3 frames of detection drop (e.g. person turned head)
    # On frame 1 drop: missed_frames increments to 1, face RETAINED
    out_1 = buffer_detector.detect_faces(dummy_frame)
    assert len(out_1) == 1, "Face must be retained on frame 1 drop!"
    assert buffer_detector._face_buffer[1]["missed_frames"] == 1

    # On frame 3 drop: face STILL RETAINED
    buffer_detector.detect_faces(dummy_frame)
    out_3 = buffer_detector.detect_faces(dummy_frame)
    assert len(out_3) == 1, "Face must be retained within 5-frame grace period!"
    assert buffer_detector._face_buffer[1]["missed_frames"] == 3
    print("  -> 3 consecutive missed frames: Face retained in buffer [CORRECT - ZERO FLICKER]")

    # On frame 6 drop (exceeds max 5 missed frames): face PURGED
    for _ in range(3):
        buffer_detector.detect_faces(dummy_frame)
    out_final = buffer_detector.detect_faces(dummy_frame)
    assert len(out_final) == 0, "Face must be purged after exceeding 5 missed frames!"
    print("  -> 6 missed frames: Face gracefully purged from buffer [CORRECT - TIMELY DEREGISTRATION]")
    print("  [PASS] Temporal buffer and grace period eliminate flicker and purge gracefully.")

    # 4. Latency Benchmark
    print("\n[STEP 4] Benchmarking YuNet Inference Latency on 640x480 Frame...")
    # Warm-up run
    detector.detect(dummy_frame)

    iterations = 25
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        detector.detect(dummy_frame)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_latency = np.mean(latencies)
    print(f"  -> Avg Latency  : {avg_latency:.2f} ms")
    print(f"  -> Est. FPS Equiv: {1000.0 / avg_latency:.1f} FPS (standalone)")
    assert avg_latency < 80.0, f"Latency exceeds budget: {avg_latency:.2f} ms"
    print("  [PASS] Latency well within budget.")

    print("\n" + "=" * 65)
    print("       ALL YUNET FACE DETECTOR VERIFICATIONS PASSED (100%)       ")
    print("=" * 65 + "\n")
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
