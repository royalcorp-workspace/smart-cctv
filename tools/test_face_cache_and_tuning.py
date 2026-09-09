"""Automated Verification Suite for Face Cache, String Normalization, and Threshold Tuning.

Tests:
1. Regex name parsing & Title Case normalization.
2. Fast startup .npz caching & SHA-256 signature invalidation.
3. Cold vs Hot startup performance benchmark (< 10 ms hot load).
4. Cosine similarity threshold tuning (0.60) & False Positive suppression.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
import numpy as np

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from engine.face_recognizer import FaceRecognizer


class TestFaceCacheAndTuning(unittest.TestCase):
    """Test suite for fast npz caching, regex name parsing, and 0.60 threshold."""

    def test_01_name_normalization_regex(self):
        """Verify regex transforms raw filenames into clean Title Case names."""
        cases = [
            ("ADI-AHMAD", "Adi Ahmad"),
            ("AJI_YULIANTO", "Aji Yulianto"),
            ("ALDY", "Aldy"),
            ("ALGHANY", "Alghany"),
            ("RIZQI-SETIAWAN", "Rizqi Setiawan"),
            ("budi_santoso_2", "Budi Santoso"),
            ("john-doe-1", "John Doe"),
            ("jane___smith__3", "Jane Smith"),
            ("SINGLE", "Single"),
        ]
        for raw, expected in cases:
            cleaned = FaceRecognizer.normalize_name(raw)
            self.assertEqual(cleaned, expected, f"Failed for raw input '{raw}': got '{cleaned}'")

    def test_02_default_threshold_0_60(self):
        """Verify default cosine threshold is updated to 0.60."""
        # Using empty temp directory to avoid heavy disk loading
        with tempfile.TemporaryDirectory() as tmp_dir:
            fr = FaceRecognizer(known_faces_dir=tmp_dir)
            self.assertEqual(fr.cosine_threshold, 0.60, "Default cosine threshold must be 0.60")

    def test_03_npz_caching_roundtrip_and_speedup(self):
        """Verify .npz caching creates valid file, invalidates on change, and loads in < 10ms."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create synthetic reference image files
            img1 = tmp_path / "ALDY.jpg"
            img2 = tmp_path / "AJI-YULIANTO.jpeg"
            img1.write_bytes(b"dummy1")
            img2.write_bytes(b"dummy2")

            fr = FaceRecognizer(known_faces_dir=str(tmp_path))
            cache_file = tmp_path / ".embeddings_cache.npz"

            # Mock extraction so we don't depend on actual face detection
            mock_emb1 = np.ones((1, 128), dtype=np.float32) * 0.1
            mock_emb2 = np.ones((1, 128), dtype=np.float32) * 0.2
            fr.known_embeddings = [("Aldy", mock_emb1), ("Aji Yulianto", mock_emb2)]

            # Save cache
            sig1 = fr._compute_dataset_signature([img1, img2])
            fr._save_cache(sig1)
            self.assertTrue(cache_file.exists(), "Cache file must exist after _save_cache")

            # Test hot load
            fr_hot = FaceRecognizer(known_faces_dir=str(tmp_path))
            t0 = time.perf_counter()
            loaded = fr_hot._load_cache(sig1)
            load_time_ms = (time.perf_counter() - t0) * 1000.0

            self.assertTrue(loaded, "Hot load with matching signature must succeed")
            self.assertEqual(len(fr_hot.known_embeddings), 2)
            self.assertEqual(fr_hot.known_embeddings[0][0], "Aldy")
            self.assertEqual(fr_hot.known_embeddings[1][0], "Aji Yulianto")
            self.assertLess(load_time_ms, 10.0, f"Hot load must be < 10ms, took {load_time_ms:.2f}ms")

            # Test cache invalidation when signature changes (e.g. new file added)
            img3 = tmp_path / "RIZQI-SETIAWAN.png"
            img3.write_bytes(b"dummy3")
            sig2 = fr._compute_dataset_signature([img1, img2, img3])
            self.assertNotEqual(sig1, sig2, "Signature must change when new file is added")

            # Mismatched signature must reject stale cache
            stale_load = fr_hot._load_cache(sig2)
            self.assertFalse(stale_load, "Stale cache with mismatched signature must be rejected")

    def test_04_cosine_threshold_discrimination(self):
        """Verify 0.60 threshold rejects 0.560 cross-similarity (Alghany vs Rizqi) and accepts match."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            fr = FaceRecognizer(known_faces_dir=tmp_dir, cosine_threshold=0.60)

            # Known identity
            known_emb = np.zeros((1, 128), dtype=np.float32)
            known_emb[0, 0] = 1.0
            fr.known_embeddings = [("Alghany", known_emb)]

            # Query 1: Alghany herself (sim = 1.000 >= 0.60) -> MATCH
            name, score, label = fr.recognize(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                face_data=(0, 0, 50, 50),
                min_size=24,
            )
            # Override feature extraction for deterministic unit testing
            fr.extract_feature = lambda frame, data, min_size=24: known_emb
            name, score, label = fr.recognize(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                face_data=(0, 0, 50, 50),
                min_size=24,
            )
            self.assertEqual(name, "Alghany")
            self.assertGreaterEqual(score, 0.60)
            self.assertIn("Alghany", label)

            # Query 2: Cross-identity resembling Rizqi Setiawan (sim = 0.560 < 0.60) -> REJECT / UNKNOWN
            # Vector with cosine similarity exactly 0.560
            cross_emb = np.zeros((1, 128), dtype=np.float32)
            cross_emb[0, 0] = 0.560
            cross_emb[0, 1] = np.sqrt(1.0 - 0.560**2)
            fr.extract_feature = lambda frame, data, min_size=24: cross_emb

            name, score, label = fr.recognize(
                frame=np.zeros((100, 100, 3), dtype=np.uint8),
                face_data=(0, 0, 50, 50),
                min_size=24,
            )
            self.assertEqual(name, "Unknown", "Cross-match with sim=0.560 must be rejected under 0.60 threshold")
            self.assertEqual(label, "Unknown")
            self.assertAlmostEqual(score, 0.560, places=2)


if __name__ == "__main__":
    unittest.main()
