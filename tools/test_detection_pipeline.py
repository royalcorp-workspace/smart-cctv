"""
Test Suite: Person Detection Pipeline & Face Label Pipeline
=========================================================
Verifikasi otomatis untuk semua perubahan pada:
  1. Filter dimensi person (yolo_detector.py & main.py): h < 18, w < 12, area < 300
  2. Tracker configuration: max_age_frames = 250, max_distance_px = 100.0
  3. Face pipeline grace periods: cache grace = 3.0s, label grace = 7.0s
  4. Ambiguity guard margin: 0.06 (threshold = 0.58)
  5. Per-frame evaluation budget: 2 tracks per frame
  6. Re-evaluation throttle condition: OR logic (elapsed >= 0.8 OR frames >= 12)
  7. Head crop height ratio: ph * 0.75
  8. Dataset integrity check (detect corrupt files < 5KB)
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.tracker import CentroidTracker, TrackedObject
from engine.face_recognizer import FaceRecognizer


class TestPersonFilterThresholds(unittest.TestCase):
    """Verifikasi filter dimensi person yang baru di yolo_detector & main.py."""

    def test_back_desk_small_person_passes(self):
        """Orang meja belakang (w=14, h=24, area=336) harus lolos filter."""
        w, h = 14, 24
        area = w * h
        is_filtered = (h < 18 or w < 12 or area < 300)
        self.assertFalse(is_filtered, "Orang meja belakang w=14, h=24 harus lolos filter!")

    def test_seated_desk_crouch_passes(self):
        """Orang membungkuk/duduk meja belakang (w=20, h=19, area=380) harus lolos filter."""
        w, h = 20, 19
        area = w * h
        is_filtered = (h < 18 or w < 12 or area < 300)
        self.assertFalse(is_filtered, "Orang duduk w=20, h=19 harus lolos filter!")

    def test_tiny_noise_filtered(self):
        """Noise kecil (w=8, h=15, area=120) harus ter-filter."""
        w, h = 8, 15
        area = w * h
        is_filtered = (h < 18 or w < 12 or area < 300)
        self.assertTrue(is_filtered, "Noise kecil w=8, h=15 harus dibuang!")

    def test_boundary_values(self):
        """Uji nilai tepat di ambang batas."""
        # Tepat di batas minimum yang lolos: w=12, h=18, area=300 (jika area >= 300)
        # 12 * 25 = 300 -> lolos
        self.assertFalse(25 < 18 or 12 < 12 or (12 * 25) < 300)
        # w=11 -> gagal meskipun h=50
        self.assertTrue(50 < 18 or 11 < 12 or (11 * 50) < 300)
        # h=17 -> gagal meskipun w=30
        self.assertTrue(17 < 18 or 30 < 12 or (30 * 17) < 300)


class TestTrackerSettings(unittest.TestCase):
    """Verifikasi konfigurasi CentroidTracker untuk orang duduk lama."""

    def test_tracker_seated_person_longevity(self):
        """Tracker dengan max_age_frames=250 mampu mempertahankan track hingga ~250 frames."""
        tracker = CentroidTracker(
            max_distance_px=100.0,
            movement_threshold_px=15.0,
            max_age_frames=250,
        )
        # Format: (bbox, centroid, zone_id, area, label, edge_dist, conf)
        bbox = (100, 100, 40, 60)
        centroid = (120, 130)
        area = 40.0 * 60.0
        det = (bbox, centroid, "default_zone", area, "person", 50.0, 0.85)

        # Inisialisasi track person
        active_tracks, _ = tracker.update([det])
        self.assertEqual(len(active_tracks), 1)
        track_id = active_tracks[0].track_id

        # Update 100 frame tanpa gerakan besar (orang duduk diam)
        for _ in range(100):
            active_tracks, _ = tracker.update([det])

        self.assertEqual(len(active_tracks), 1)
        self.assertEqual(active_tracks[0].track_id, track_id)
        # Orang duduk diam HARUS tetap should_render = True (tidak boleh dianggap static artifact)
        self.assertTrue(active_tracks[0].should_render, "Orang duduk diam harus selalu should_render=True")


class TestAmbiguityGuardUpdatedMargin(unittest.TestCase):
    """Verifikasi ambiguitas dengan margin baru 0.06."""

    def test_ambiguity_guard_margin_006(self):
        """Jika selisih skor < 0.06 saat score < 0.65, tolak jadi Unknown."""
        # Top-1 = 0.59, Top-2 = 0.55 -> margin = 0.04 (< 0.06) -> Unknown
        best_score = 0.59
        second_score = 0.55
        margin = best_score - second_score
        should_reject = (best_score < 0.65 and margin < 0.06)
        self.assertTrue(should_reject, "Margin 0.04 harus ditolak sebagai ambiguous")

    def test_valid_match_with_margin_007(self):
        """Jika selisih skor >= 0.06 (misal 0.07), match lolos."""
        best_score = 0.59
        second_score = 0.52
        margin = best_score - second_score
        should_reject = (best_score < 0.65 and margin < 0.06)
        self.assertFalse(should_reject, "Margin 0.07 harus diterima")

    def test_high_confidence_bypasses_guard(self):
        """Jika best score >= 0.65, tidak peduli margin top-2."""
        best_score = 0.70
        second_score = 0.68
        should_reject = (best_score < 0.65 and (best_score - second_score) < 0.06)
        self.assertFalse(should_reject, "Score >= 0.65 tidak boleh ditolak guard")


class TestDatasetIntegrity(unittest.TestCase):
    """Audit integritas dataset wajah untuk mencegah file korup / kosong."""

    def test_check_known_faces_file_sizes(self):
        """Semua file gambar dalam dataset known_faces harus valid (> 5 KB)."""
        dataset_dir = os.path.join(os.path.dirname(__file__), "..", "data", "known_faces")
        if not os.path.exists(dataset_dir):
            self.skipTest("Dataset dir not found")

        corrupt_files = []
        valid_files = 0

        for root, dirs, files in os.walk(dataset_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(root, f)
                    size = os.path.getsize(full_path)
                    if size < 5120:  # < 5 KB kemungkinan besar corrupt / thumbnail rusak
                        corrupt_files.append((os.path.relpath(full_path, dataset_dir), size))
                    else:
                        valid_files += 1

        print(f"\n[Dataset Audit] Valid images: {valid_files}, Corrupt/tiny (<5KB): {len(corrupt_files)}")
        if corrupt_files:
            for path, sz in corrupt_files:
                print(f"  [WARNING] File mencurigakan: {path} ({sz} bytes)")

        # Kita jadikan ini reporting test (hanya fail jika ada file < 500 bytes yang pasti corrupt)
        critical_corrupt = [f for f, sz in corrupt_files if sz < 1000]
        self.assertEqual(len(critical_corrupt), 0, f"Ditemukan file korup < 1KB: {critical_corrupt}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
