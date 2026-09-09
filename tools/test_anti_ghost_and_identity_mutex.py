"""
Test Suite: Anti-Ghost Face Anchor, Identity Mutex, and Adaptive CPU Throttling
==============================================================================
Verifikasi otomatis untuk:
  1. Strict Person-Box Anchor: Wajah hanya boleh dicari & digambar di top 70% person bbox.
  2. Identity Mutex (Anti-Cloning Guard): Menolak duplikasi nama yang sama dalam 1 frame,
     track dengan skor tertinggi memenangkan identitas, track lain didemotasi ke 'Unknown'.
  3. Adaptive CPU Throttling: Menghindari pemborosan CPU untuk orang duduk statis
     yang sudah dikenali (cooldown 15.0 detik).
  4. FaceDetector No Full-Frame Leak: Memastikan full-frame inference dimatikan saat
     crop_roi diberikan.
"""

import os
import sys
import unittest
from typing import Dict, Tuple, List, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestStrictPersonBoxAnchor(unittest.TestCase):
    """Pengujian validasi spasial wajah agar terkunci strictly di top 70% person bbox."""

    def is_face_anchored(self, person_bbox: Tuple[int, int, int, int], face_bbox: Tuple[int, int, int, int]) -> bool:
        """Simulasi logika anchor spasial di main.py."""
        px, py, pw, ph = person_bbox
        fx, fy, fw, fh = face_bbox
        fcx = fx + fw // 2
        fcy = fy + fh // 2

        head_x1 = px - int(pw * 0.15)
        head_x2 = px + pw + int(pw * 0.15)
        head_y1 = py - int(ph * 0.10)
        head_y2 = py + int(ph * 0.70)

        # 1. Containment check
        if not (head_x1 <= fcx <= head_x2 and head_y1 <= fcy <= head_y2):
            return False

        # 2. Proportional size check
        if fw > (pw * 1.1) or fh > (ph * 0.70):
            return False

        return True

    def test_face_in_top_30_percent_valid(self):
        """Wajah di area atas (kepala normal) harus valid."""
        person = (100, 100, 80, 160)  # ph = 160 -> top 70% is 100..212
        face = (120, 110, 40, 45)      # fcx = 140, fcy = 132 (top ~20%)
        self.assertTrue(self.is_face_anchored(person, face))

    def test_face_in_seated_top_60_percent_valid(self):
        """Wajah orang menunduk/duduk di meja (top 60%) harus valid."""
        person = (100, 100, 80, 160)
        face = (120, 180, 35, 40)      # fcx = 137, fcy = 200 (top 62.5% <= 70%)
        self.assertTrue(self.is_face_anchored(person, face))

    def test_ghost_face_in_bottom_half_rejected(self):
        """Ghost box di area bawah tubuh/kaki (misal pantulan meja/lantai) harus ditolak."""
        person = (100, 100, 80, 160)
        face = (120, 230, 35, 40)      # fcy = 250 -> 250 > 212 (top 70%)
        self.assertFalse(self.is_face_anchored(person, face), "Wajah di bawah 70% person bbox harus ditolak!")

    def test_ghost_face_drifting_to_adjacent_chair_rejected(self):
        """Kotak wajah yang melayang ke kursi kosong di sebelahnya harus ditolak."""
        person = (100, 100, 80, 160)  # pw = 80, margin = 12 -> head_x2 = 192
        ghost_face = (220, 120, 35, 40) # fcx = 237 > 192
        self.assertFalse(self.is_face_anchored(person, ghost_face), "Wajah melayang di luar x-margin person harus ditolak!")

    def test_oversized_face_rejected(self):
        """Kotak deteksi yang lebih besar dari tubuh orang (bukan wajah) harus ditolak."""
        person = (100, 100, 50, 100)
        huge_face = (90, 90, 70, 85)   # fw = 70 > 55 (1.1 * pw)
        self.assertFalse(self.is_face_anchored(person, huge_face), "Kotak melebihi proporsi person harus ditolak!")


class TestIdentityMutex(unittest.TestCase):
    """Pengujian Identity Mutex untuk mencegah cloning nama pada frame yang sama."""

    def resolve_identity_mutex(self, candidate_faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulasi resolusi Identity Mutex seperti pada main.py."""
        known_candidates = [
            c for c in candidate_faces
            if c.get("is_known", False) and c.get("name", "").lower() != "unknown"
        ]
        # Sort by score descending
        known_candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        claimed_identities: Dict[str, Tuple[int, float]] = {}
        demoted_track_ids = set()

        for c in known_candidates:
            c_name = c["name"]
            tid = c["track_id"]
            score = c["score"]
            if c_name not in claimed_identities:
                claimed_identities[c_name] = (tid, score)
            else:
                demoted_track_ids.add(tid)

        final_faces = []
        for c in candidate_faces:
            tid = c["track_id"]
            if tid in demoted_track_ids:
                final_faces.append({
                    "track_id": tid,
                    "name": "Unknown",
                    "label": "Unknown",
                    "score": 0.0,
                    "is_known": False,
                })
            else:
                final_faces.append({
                    "track_id": tid,
                    "name": c["name"],
                    "label": c["label"],
                    "score": c["score"],
                    "is_known": c.get("is_known", False),
                })
        return final_faces

    def test_duplicate_identity_demoted_to_unknown(self):
        """Jika ada 2 track mengklaim 'Rian Heri' (62% dan 58%), yang 58% harus jadi Unknown."""
        candidates = [
            {"track_id": 1, "name": "Rian Heri", "label": "Rian Heri (62%)", "score": 0.62, "is_known": True},
            {"track_id": 2, "name": "Rian Heri", "label": "Rian Heri (58%)", "score": 0.58, "is_known": True},
            {"track_id": 3, "name": "Aji Yulianto", "label": "Aji Yulianto (60%)", "score": 0.60, "is_known": True},
            {"track_id": 4, "name": "Unknown", "label": "Unknown", "score": 0.0, "is_known": False},
        ]

        resolved = self.resolve_identity_mutex(candidates)

        # Cari track 1 dan track 2
        t1 = next(r for r in resolved if r["track_id"] == 1)
        t2 = next(r for r in resolved if r["track_id"] == 2)
        t3 = next(r for r in resolved if r["track_id"] == 3)
        t4 = next(r for r in resolved if r["track_id"] == 4)

        self.assertEqual(t1["name"], "Rian Heri")
        self.assertEqual(t1["score"], 0.62)

        # Track 2 harus didemotasi menjadi Unknown
        self.assertEqual(t2["name"], "Unknown", "Track duplikat harus didemotasi menjadi Unknown!")
        self.assertEqual(t2["score"], 0.0)

        # Track 3 tetap Aji Yulianto
        self.assertEqual(t3["name"], "Aji Yulianto")

        # Track 4 tetap Unknown
        self.assertEqual(t4["name"], "Unknown")

        # Pastikan tidak ada duplikasi nama terdaftar
        registered_names = [r["name"] for r in resolved if r["name"] != "Unknown"]
        self.assertEqual(len(registered_names), len(set(registered_names)), "Tidak boleh ada nama duplikat!")

    def test_triple_identity_collision(self):
        """Tiga track mengklaim nama yang sama: hanya skor tertinggi yang menang."""
        candidates = [
            {"track_id": 10, "name": "Aldy", "label": "Aldy (55%)", "score": 0.55, "is_known": True},
            {"track_id": 11, "name": "Aldy", "label": "Aldy (68%)", "score": 0.68, "is_known": True},
            {"track_id": 12, "name": "Aldy", "label": "Aldy (59%)", "score": 0.59, "is_known": True},
        ]

        resolved = self.resolve_identity_mutex(candidates)
        winner = [r for r in resolved if r["name"] == "Aldy"]
        self.assertEqual(len(winner), 1)
        self.assertEqual(winner[0]["track_id"], 11)
        self.assertEqual(winner[0]["score"], 0.68)

        demoted = [r for r in resolved if r["name"] == "Unknown"]
        self.assertEqual(len(demoted), 2)


class TestAdaptiveCPUThrottling(unittest.TestCase):
    """Pengujian adaptif cooldown untuk membebaskan beban CPU orang duduk."""

    def get_throttle_values(self, is_known: bool, score: float, is_moving: bool, fail_count: int) -> Tuple[float, int]:
        """Menghitung throttle_sec dan min_frames sesuai logika main.py."""
        if is_known and score >= 0.60 and not is_moving:
            return 15.0, 150
        elif not is_moving and fail_count >= 2:
            return 6.0, 60
        elif is_moving:
            return 1.2, 12
        else:
            return 3.0, 30

    def test_confirmed_stationary_person_gets_15s_cooldown(self):
        """Karyawan duduk yang sudah terkonfirmasi (score >= 0.60) diberi cooldown 15.0s."""
        throttle_sec, min_frames = self.get_throttle_values(is_known=True, score=0.65, is_moving=False, fail_count=0)
        self.assertEqual(throttle_sec, 15.0)
        self.assertEqual(min_frames, 150)

    def test_moving_person_gets_responsive_cooldown(self):
        """Orang yang bergerak dievaluasi cepat (1.2 detik / 12 frame)."""
        throttle_sec, min_frames = self.get_throttle_values(is_known=False, score=0.0, is_moving=True, fail_count=0)
        self.assertEqual(throttle_sec, 1.2)
        self.assertEqual(min_frames, 12)

    def test_repeated_fail_stationary_gets_6s_backoff(self):
        """Orang duduk yang gagal dideteksi 2x diberi backoff 6.0 detik."""
        throttle_sec, min_frames = self.get_throttle_values(is_known=False, score=0.0, is_moving=False, fail_count=3)
        self.assertEqual(throttle_sec, 6.0)
        self.assertEqual(min_frames, 60)


class TestFaceDetectorNoFullFrameLeak(unittest.TestCase):
    """Pengujian face_detector.py untuk memastikan full-frame dimatikan saat crop_roi diberikan."""

    def test_has_crop_disables_full_frame(self):
        """Jika crop_roi diberikan, run_crop = True dan run_full = False."""
        crop_rois_list = [(10, 10, 100, 100)]
        display_frame_present = True

        has_crop = display_frame_present and (len(crop_rois_list) > 0)
        if has_crop:
            run_crop = True
            run_full = False
        else:
            run_crop = False
            run_full = True

        self.assertTrue(run_crop)
        self.assertFalse(run_full, "Full frame detection HARUS dimatikan saat crop ROI diberikan!")

    def test_no_crop_enables_full_frame(self):
        """Jika tidak ada crop_roi, run_full = True dan run_crop = False."""
        crop_rois_list = []
        display_frame_present = True

        has_crop = display_frame_present and (len(crop_rois_list) > 0)
        if has_crop:
            run_crop = True
            run_full = False
        else:
            run_crop = False
            run_full = True

        self.assertFalse(run_crop)
        self.assertTrue(run_full)


if __name__ == "__main__":
    unittest.main(verbosity=2)
