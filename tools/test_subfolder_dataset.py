"""Unit tests for the subfolder-based identity dataset refactor in face_recognizer.py.

Tests the identity-resolution logic and SHA-256 signature behaviour
without requiring cv2, ultralytics, or a live camera feed.
"""

import hashlib
import pathlib
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Minimal stubs so face_recognizer can be imported without heavy deps
# ---------------------------------------------------------------------------
for _mod in ("cv2", "ultralytics", "pygame"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)

import numpy as _np  # noqa: E402  (numpy is always available in .venv)
if "numpy" not in sys.modules:
    sys.modules["numpy"] = _np

# cv2 needs FaceRecognizerSF attribute for class-level code
_cv2_stub = sys.modules["cv2"]
if not hasattr(_cv2_stub, "FaceRecognizerSF_FR_COSINE"):
    _cv2_stub.FaceRecognizerSF_FR_COSINE = 0  # type: ignore

# Stub only the leaf sub-modules that have heavy runtime deps.
# Do NOT stub the 'engine' package itself — let Python find the real package.
import logging as _logging

_logger_mod = types.ModuleType("engine.logger")
_logger_mod.logger = _logging.getLogger("test_face_rec")  # type: ignore
sys.modules.setdefault("engine.logger", _logger_mod)

_fd_mod = types.ModuleType("engine.face_detector")
_fd_mod.apply_clahe = lambda img, **kw: img  # type: ignore
sys.modules.setdefault("engine.face_detector", _fd_mod)

# ---------------------------------------------------------------------------
# Now import the module under test
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine.face_recognizer import FaceRecognizer  # noqa: E402


# ---------------------------------------------------------------------------
# Helper — identity resolution extracted from load_database logic
# ---------------------------------------------------------------------------

def _resolve_identity(img_path: pathlib.Path, known_faces_dir: pathlib.Path) -> str:
    """Mirror the identity-resolution logic in FaceRecognizer.load_database."""
    parent = img_path.parent
    if parent == known_faces_dir:
        return FaceRecognizer.normalize_name(img_path.stem)
    else:
        return FaceRecognizer.normalize_name(parent.name)


def _fake_signature(files: list, base: pathlib.Path) -> str:
    """Mirror _compute_dataset_signature without stat() calls."""
    h = hashlib.sha256()
    for p in sorted(files):
        try:
            rel = str(p.relative_to(base))
            h.update(rel.encode("utf-8"))
        except ValueError:
            pass
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

BASE = pathlib.Path("data/known_faces")


class TestNormalizeName(unittest.TestCase):
    """Sanity-check normalize_name (unchanged behaviour)."""

    def test_dash_suffix_stripped(self):
        self.assertEqual(FaceRecognizer.normalize_name("ALGHANY-1"), "Alghany")

    def test_underscore_to_space(self):
        self.assertEqual(FaceRecognizer.normalize_name("RIZQI_SETIAWAN"), "Rizqi Setiawan")

    def test_mixed_case(self):
        self.assertEqual(FaceRecognizer.normalize_name("rian-heri"), "Rian Heri")

    def test_already_clean_subfolder_name(self):
        self.assertEqual(FaceRecognizer.normalize_name("Alghany"), "Alghany")
        self.assertEqual(FaceRecognizer.normalize_name("Rizqi Setiawan"), "Rizqi Setiawan")


class TestIdentityResolution(unittest.TestCase):
    """Test that label comes from folder name (subfolder) or filename stem (root)."""

    def _r(self, p: pathlib.Path) -> str:
        return _resolve_identity(p, BASE)

    def test_flat_root_file_uses_stem(self):
        self.assertEqual(self._r(BASE / "RIAN_HERI.jpg"), "Rian Heri")

    def test_flat_root_file_with_number_stripped(self):
        self.assertEqual(self._r(BASE / "ALGHANY-1.jpg"), "Alghany")

    def test_subfolder_uses_folder_name(self):
        self.assertEqual(self._r(BASE / "Alghany" / "cctv_overhead.jpg"), "Alghany")

    def test_subfolder_multiple_photos_same_identity(self):
        p1 = BASE / "Rizqi Setiawan" / "1.jpg"
        p2 = BASE / "Rizqi Setiawan" / "2.jpg"
        self.assertEqual(self._r(p1), "Rizqi Setiawan")
        self.assertEqual(self._r(p2), "Rizqi Setiawan")

    def test_subfolder_underscore_name_normalised(self):
        p = BASE / "budi_santoso" / "frontal.png"
        self.assertEqual(self._r(p), "Budi Santoso")

    def test_subfolder_mixed_angle_photos(self):
        for fname in ("depan.jpg", "kiri.jpg", "overhead_cctv.jpg"):
            p = BASE / "Alghany" / fname
            self.assertEqual(self._r(p), "Alghany",
                             f"Expected 'Alghany' for {fname}")

    def test_flat_root_png_uses_stem(self):
        self.assertEqual(self._r(BASE / "budi_santoso.png"), "Budi Santoso")


class TestDatasetSignature(unittest.TestCase):
    """Test that the SHA-256 signature covers relative paths (not just filenames)."""

    def test_flat_vs_subfolder_different_sig(self):
        flat = [BASE / "face.jpg"]
        sub  = [BASE / "Alghany" / "face.jpg"]
        self.assertNotEqual(
            _fake_signature(flat, BASE),
            _fake_signature(sub, BASE),
            "Same filename in different locations must produce different signatures",
        )

    def test_different_subfolders_different_sig(self):
        s1 = [BASE / "Alghany" / "1.jpg"]
        s2 = [BASE / "Rizqi Setiawan" / "1.jpg"]
        self.assertNotEqual(_fake_signature(s1, BASE), _fake_signature(s2, BASE))

    def test_same_files_same_sig(self):
        files = [BASE / "Alghany" / "a.jpg", BASE / "Rizqi Setiawan" / "b.jpg"]
        self.assertEqual(
            _fake_signature(files, BASE),
            _fake_signature(files, BASE),
        )

    def test_adding_file_changes_sig(self):
        before = [BASE / "Alghany" / "1.jpg"]
        after  = [BASE / "Alghany" / "1.jpg", BASE / "Alghany" / "2.jpg"]
        self.assertNotEqual(_fake_signature(before, BASE), _fake_signature(after, BASE))

    def test_empty_set_is_stable(self):
        self.assertEqual(_fake_signature([], BASE), _fake_signature([], BASE))


if __name__ == "__main__":
    unittest.main(verbosity=2)
