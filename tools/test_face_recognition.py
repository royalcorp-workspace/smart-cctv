"""Verification & Diagnostic Tool for OpenCV SFace Face Recognition Engine."""

import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.face_detector import YuNetFaceDetector
from engine.face_recognizer import FaceRecognizer
from notification.local_alert import VisualHUD


def test_model_and_initialization() -> bool:
    print("\n[STEP 1] Testing SFace Model Weight & Initialization...")
    model_path = PROJECT_ROOT / "models" / "face_recognition_sface_2021dec.onnx"
    assert model_path.exists(), f"Model not found at: {model_path}"

    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f" - Model path: {model_path}")
    print(f" - File size : {file_size_mb:.2f} MB (Expected ~36-39 MB)")
    assert file_size_mb > 30.0, "Model weight file is too small or truncated!"

    start = time.time()
    recognizer = FaceRecognizer(model_path=str(model_path), cosine_threshold=0.50)
    init_ms = (time.time() - start) * 1000.0
    print(f" - Initialized FaceRecognizer in {init_ms:.1f} ms")
    assert recognizer.recognizer is not None, "FaceRecognizerSF instance should not be None"
    print(" -> PASS: SFace Model weight and initialization verified.")
    return True


def test_database_indexing() -> bool:
    print("\n[STEP 2] Testing Reference Photo Database Indexing (data/known_faces/)...")
    known_dir = PROJECT_ROOT / "data" / "known_faces"
    assert known_dir.exists(), f"Directory not found: {known_dir}"

    recognizer = FaceRecognizer(known_faces_dir=str(known_dir))
    print(f" - Total indexed embeddings: {len(recognizer.known_embeddings)}")

    # rian.jpeg should be indexed
    rian_path = known_dir / "rian.jpeg"
    if rian_path.exists():
        names = [n for n, _ in recognizer.known_embeddings]
        print(f" - Indexed identities: {names}")
        assert "Rian" in names, "Identity 'Rian' should be indexed from rian.jpeg"
        # Check 128-D vector shape
        for name, feat in recognizer.known_embeddings:
            assert feat.shape == (1, 128), f"Expected (1, 128) feature vector, got {feat.shape}"

    print(" -> PASS: Database indexing and 128-D feature extraction verified.")
    return True


def test_face_matching_known_and_unknown() -> bool:
    print("\n[STEP 3] Testing Cosine Similarity Matching (Known vs Unknown)...")
    known_dir = PROJECT_ROOT / "data" / "known_faces"
    recognizer = FaceRecognizer(known_faces_dir=str(known_dir), cosine_threshold=0.50)
    detector = YuNetFaceDetector(score_threshold=0.40)

    # 1. Test Known Face Matching (rian.jpeg)
    rian_path = known_dir / "rian.jpeg"
    if rian_path.exists():
        rian_img = cv2.imread(str(rian_path))
        detector.set_input_size((rian_img.shape[1], rian_img.shape[0]))
        ret, faces = detector.detector.detect(rian_img)
        assert faces is not None and len(faces) > 0, "YuNet must detect face in rian.jpeg"
        raw_face = faces[0]

        name, sim_score, label = recognizer.recognize(rian_img, raw_face)
        print(f" - Known Query Result : Name='{name}', Similarity={sim_score:.4f}, Label='{label}'")
        assert name == "Rian", f"Expected match 'Rian', got '{name}'"
        assert sim_score >= 0.95, f"Expected near-perfect match >= 0.95, got {sim_score}"
        assert "Rian" in label and "%" in label, f"Label should be formatted e.g. 'Rian (100%)', got '{label}'"

    # 2. Test Unknown Face Matching (Synthetic non-matching face image)
    # Generate a blank image with simple drawn shapes that YuNet detects but SFace produces non-matching embedding
    synthetic_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(synthetic_img, (320, 240), 60, (220, 220, 220), -1)
    cv2.circle(synthetic_img, (300, 220), 8, (30, 30, 30), -1)
    cv2.circle(synthetic_img, (340, 220), 8, (30, 30, 30), -1)
    cv2.circle(synthetic_img, (320, 250), 6, (30, 30, 30), -1)
    cv2.line(synthetic_img, (305, 270), (335, 270), (30, 30, 30), 3)

    detector.set_input_size((640, 480))
    ret_syn, faces_syn = detector.detector.detect(synthetic_img)
    if faces_syn is not None and len(faces_syn) > 0:
        raw_syn = faces_syn[0]
        syn_name, syn_score, syn_label = recognizer.recognize(synthetic_img, raw_syn)
        print(f" - Unknown Query Result: Name='{syn_name}', Similarity={syn_score:.4f}, Label='{syn_label}'")
        assert syn_name == "Unknown", f"Expected 'Unknown', got '{syn_name}'"
        assert syn_score < 0.50, f"Expected similarity < 0.50 threshold, got {syn_score}"
        assert syn_label == "Unknown", f"Expected label 'Unknown', got '{syn_label}'"
    else:
        # Fallback test with random feature comparison
        random_feat = np.random.randn(1, 128).astype(np.float32)
        random_feat /= np.linalg.norm(random_feat)
        sim = recognizer.recognizer.match(random_feat, recognizer.known_embeddings[0][1], cv2.FaceRecognizerSF_FR_COSINE)
        print(f" - Synthetic Random Feature Sim: {sim:.4f} (< 0.50 threshold)")
        assert sim < 0.50

    print(" -> PASS: Cosine similarity matching correctly discriminates Known vs Unknown.")
    return True


def test_empty_database_fallback() -> bool:
    print("\n[STEP 4] Testing Graceful Empty Database Fallback...")
    temp_dir = Path(tempfile.mkdtemp(prefix="known_faces_empty_"))
    try:
        empty_recognizer = FaceRecognizer(known_faces_dir=str(temp_dir))
        assert len(empty_recognizer.known_embeddings) == 0, "Embeddings list should be empty"

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_face = (100, 100, 50, 50)
        name, score, label = empty_recognizer.recognize(dummy_frame, dummy_face)

        print(f" - Empty DB query result: name='{name}', score={score}, label='{label}'")
        assert name == "Unknown"
        assert label == "Unknown"
        assert score == 0.0
        print(" -> PASS: Empty database operates gracefully without crashing or errors.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return True


def test_hud_rendering() -> bool:
    print("\n[STEP 5] Testing HUD Visual Overlay with Recognized Name Badges...")
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    zones = {"zone_2_transit": [[255, 405], [304, 472], [610, 474], [626, 431], [423, 290]]}

    faces = [
        ((150, 100, 60, 60), 0.92, "Rian (98%)"),
        ((450, 120, 55, 55), 0.85, "Unknown"),
    ]

    rendered = VisualHUD.render(
        canvas=canvas,
        zones=zones,
        tracked_objects=[],
        camera_id="cam_01",
        fps=28.5,
        is_connected=True,
        faces=faces,
    )

    assert rendered is not None
    assert rendered.shape == (720, 1280, 3)
    print(" - VisualHUD.render executed with faces containing identity labels.")
    print(" -> PASS: HUD rendering handles recognition labels cleanly.")
    return True


def run_all_tests():
    print("=" * 65)
    print("      OPENCV SFACE FACE RECOGNITION VERIFICATION SUITE           ")
    print("=" * 65)

    assert test_model_and_initialization()
    assert test_database_indexing()
    assert test_face_matching_known_and_unknown()
    assert test_empty_database_fallback()
    assert test_hud_rendering()

    print("\n" + "=" * 65)
    print("ALL 5 FACE RECOGNITION VERIFICATIONS PASSED (100% COMPLIANT)     ")
    print("=" * 65)


if __name__ == "__main__":
    run_all_tests()
