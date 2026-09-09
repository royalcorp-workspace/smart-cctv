"""OpenCV SFace Face Recognition Engine with Local Photo Database."""

import hashlib
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import urllib.error

import cv2
import numpy as np

from engine.face_detector import apply_clahe

logger = logging.getLogger("smart_cctv")

# Official OpenCV Zoo SFace weights URL (~38.7 MB)
SFACE_DEFAULT_URL: str = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
EXPECTED_MIN_BYTES: int = 30_000_000  # Official file is ~38.6 MB


class FaceRecognizer:
    """OpenCV SFace 128-D embedding extractor and cosine similarity matcher."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        known_faces_dir: Optional[str] = None,
        cosine_threshold: float = 0.60,
        auto_download: bool = True,
        detector: Optional[Any] = None,
    ) -> None:
        self.cosine_threshold: float = float(cosine_threshold)
        if self.cosine_threshold > 1.0:
            self.cosine_threshold = self.cosine_threshold / 100.0
        base_dir = Path(__file__).resolve().parent.parent

        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = base_dir / "models" / "face_recognition_sface_2021dec.onnx"

        if known_faces_dir:
            self.known_faces_dir = Path(known_faces_dir)
        else:
            self.known_faces_dir = base_dir / "data" / "known_faces"

        self.cache_path: Path = self.known_faces_dir / ".embeddings_cache.npz"

        # Auto-download SFace weights if missing
        if not self.model_path.exists():
            if auto_download:
                self._download_weights(self.model_path)
            else:
                raise FileNotFoundError(f"SFace model weights not found at: {self.model_path}")

        # Initialize OpenCV FaceRecognizerSF
        try:
            self.recognizer = cv2.FaceRecognizerSF.create(model=str(self.model_path), config="")
            logger.info(f"[FaceRecognizer] Loaded SFace model from {self.model_path.name}")
        except Exception as e:
            logger.error(f"[FaceRecognizer] Failed to initialize FaceRecognizerSF: {e}")
            raise RuntimeError(f"FaceRecognizerSF initialization failed: {e}") from e

        # Auxiliary detector for registration
        self._detector = detector

        # In-memory database of registered faces: list of (identity_name, 128-D feature)
        self.known_embeddings: List[Tuple[str, np.ndarray]] = []
        self.load_database()

    def _download_weights(self, target_path: Path, max_retries: int = 3, timeout_sec: float = 45.0) -> None:
        """Download SFace weights with retry, timeout, and file integrity validation."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".tmp")

        logger.info(f"[FaceRecognizer] Weights not found. Downloading SFace model to {target_path}...")
        headers = {"User-Agent": "Smart-CCTV/2.0 FaceRecognizerSF"}

        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(SFACE_DEFAULT_URL, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                    data = response.read()

                if len(data) < EXPECTED_MIN_BYTES:
                    raise IOError(
                        f"Downloaded file too small: {len(data)} bytes (expected >= {EXPECTED_MIN_BYTES} bytes)"
                    )

                with open(temp_path, "wb") as f:
                    f.write(data)

                # Rename atomic replacement
                if target_path.exists():
                    target_path.unlink()
                temp_path.rename(target_path)
                logger.info(
                    f"[FaceRecognizer] Download complete! Size: {target_path.stat().st_size / (1024 * 1024):.1f} MB"
                )
                return

            except Exception as e:
                logger.warning(
                    f"[FaceRecognizer] Download attempt {attempt}/{max_retries} failed: {e}"
                )
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                if attempt < max_retries:
                    time.sleep(2.0 * attempt)
                else:
                    raise IOError(
                        f"Failed to download SFace model from {SFACE_DEFAULT_URL} after {max_retries} attempts: {e}"
                    ) from e

    def _get_detector(self) -> Any:
        """Lazy load or return YuNet detector for registering reference photos."""
        if self._detector is not None:
            return self._detector

        from engine.face_detector import YuNetFaceDetector
        self._detector = YuNetFaceDetector(
            score_threshold=0.40,
            nms_threshold=0.30,
            auto_download=True,
        )
        return self._detector

    @staticmethod
    def normalize_name(raw_name: str) -> str:
        """Normalize reference photo stem into Title Case name with spaces.

        Examples:
            'ADI-AHMAD' -> 'Adi Ahmad'
            'AJI_YULIANTO' -> 'Aji Yulianto'
            'budi_santoso_2' -> 'Budi Santoso'
            'ALGHANY-1' -> 'Alghany'
            'RIZQI-SETIAWAN' -> 'Rizqi Setiawan'
        """
        if not raw_name or not str(raw_name).strip():
            return "Unknown"
        clean = re.sub(r"[-_]\d+$", "", str(raw_name))
        clean = re.sub(r"[-_]+", " ", clean)
        clean_name = " ".join(clean.split()).title()
        return clean_name if (clean_name and clean_name.strip()) else str(raw_name).strip()

    @staticmethod
    def is_frontal_face(
        face_data: Any,
        min_eye_dist_ratio: float = 0.18,
        min_symmetry_ratio: float = 0.15,
        min_score: float = 0.45,
    ) -> Tuple[bool, str]:
        """Validate if face pose is sufficiently frontal for reliable SFace recognition.

        YuNet landmarks (15 elements):
        [0:4] bbox (x, y, w, h)
        [4:6] right eye (x, y)
        [6:8] left eye (x, y)
        [8:10] nose tip (x, y)
        [10:12] right mouth corner (x, y)
        [12:14] left mouth corner (x, y)
        [14] score (confidence)

        Returns:
            (is_frontal: bool, reason: str)
        """
        try:
            if face_data is None:
                return False, "Null face data"

            arr = np.asarray(face_data, dtype=np.float32).ravel()
            if len(arr) < 14:
                # If only bbox (4 elements) provided, bypass landmark check for backward compatibility
                return True, "No landmarks available (pass bbox)"

            # Check detection score if present (15th element)
            if len(arr) >= 15:
                score = float(arr[14])
                if score < min_score:
                    return False, f"Low YuNet confidence ({score:.2f} < {min_score:.2f})"

            w = float(arr[2])
            h = float(arr[3])
            if w <= 0.0 or h <= 0.0:
                return False, "Invalid face dimensions"

            re_x, re_y = float(arr[4]), float(arr[5])
            le_x, le_y = float(arr[6]), float(arr[7])
            nt_x, nt_y = float(arr[8]), float(arr[9])

            # 1. Interocular distance check (distance between eyes)
            d_eyes = float(np.hypot(le_x - re_x, le_y - re_y))
            eye_ratio = d_eyes / w
            if eye_ratio < min_eye_dist_ratio:
                return False, f"Side profile: narrow eye distance ratio ({eye_ratio:.2f} < {min_eye_dist_ratio:.2f})"

            # 2. Horizontal eye span
            min_eye_x = min(re_x, le_x)
            max_eye_x = max(re_x, le_x)
            eye_span = max_eye_x - min_eye_x
            if eye_span < (0.15 * w):
                return False, f"Side profile: collapsed horizontal eye span ({eye_span:.1f}px)"

            # 3. Nose horizontal placement between eyes (with 8% width tolerance for 45-deg semi-frontal)
            if nt_x < (min_eye_x - 0.08 * w) or nt_x > (max_eye_x + 0.08 * w):
                return False, f"Side profile: nose tip outside eye span (nt_x={nt_x:.1f}, eyes=[{min_eye_x:.1f}, {max_eye_x:.1f}])"

            # 4. Nose-eye horizontal symmetry ratio
            d_r = abs(nt_x - re_x)
            d_l = abs(nt_x - le_x)
            min_d = min(d_r, d_l)
            max_d = max(d_r, d_l)
            if max_d > 0.0:
                sym_ratio = min_d / max_d
                if sym_ratio < min_symmetry_ratio:
                    return False, f"Extreme yaw: asymmetric nose position ({sym_ratio:.2f} < {min_symmetry_ratio:.2f})"

            return True, "Frontal face pass"
        except Exception as e:
            return False, f"Landmark validation error: {e}"

    def _compute_dataset_signature(self, image_files: List[Path]) -> str:
        """Compute composite SHA-256 hash over sorted image paths, sizes, and mtimes.

        Accepts a flat list of absolute paths (may span multiple subdirectories),
        so the signature reflects any addition, deletion, or modification anywhere
        in the known_faces tree.
        """
        hasher = hashlib.sha256()
        for p in sorted(image_files):
            try:
                st = p.stat()
                # Include relative path from known_faces root so renames are detected
                rel = str(p.relative_to(self.known_faces_dir))
                hasher.update(rel.encode("utf-8"))
                hasher.update(str(st.st_size).encode("utf-8"))
                hasher.update(str(st.st_mtime_ns).encode("utf-8"))
            except OSError:
                continue
        return hasher.hexdigest()

    def _load_cache(self, expected_sig: str) -> bool:
        """Load precomputed 128-D embeddings from .embeddings_cache.npz if signature matches."""
        if not self.cache_path.exists():
            return False
        try:
            data = np.load(str(self.cache_path), allow_pickle=True)
            cached_sig = str(data.get("signature", ""))
            if cached_sig != expected_sig:
                logger.info("[FaceRecognizer] Cache signature mismatch, re-indexing reference photos...")
                return False
            names = data.get("names", [])
            embeddings = data.get("embeddings", [])
            if len(names) == 0 or len(embeddings) == 0 or len(names) != len(embeddings):
                return False
            self.known_embeddings.clear()
            for n, emb in zip(names, embeddings):
                self.known_embeddings.append((str(n), np.asarray(emb, dtype=np.float32)))
            logger.info(
                f"[FaceRecognizer] Fast startup: Loaded {len(self.known_embeddings)} face embeddings "
                f"from cache ({self.cache_path.name}) in < 5ms."
            )
            return True
        except Exception as e:
            logger.warning(f"[FaceRecognizer] Failed loading embeddings cache: {e}. Re-indexing.")
            return False

    def _save_cache(self, sig: str) -> None:
        """Persist current 128-D embeddings and identity names to compressed .npz file."""
        if not self.known_embeddings:
            return
        try:
            names = np.array([n for n, _ in self.known_embeddings], dtype=object)
            embeddings = np.array([emb for _, emb in self.known_embeddings], dtype=np.float32)
            np.savez_compressed(
                str(self.cache_path),
                signature=sig,
                names=names,
                embeddings=embeddings,
            )
            logger.info(
                f"[FaceRecognizer] Saved {len(self.known_embeddings)} embeddings to cache ({self.cache_path.name})."
            )
        except Exception as e:
            logger.warning(f"[FaceRecognizer] Failed saving embeddings cache: {e}")

    def load_database(self) -> int:
        """Scan data/known_faces/ recursively, extract 128-D embeddings, and cache in memory.

        Supports two dataset layouts (can be mixed):

        Subfolder-per-identity (preferred for multi-angle enrollment):
            data/known_faces/
                Alghany/
                    depan.jpg          -> identity = "Alghany"
                    cctv_overhead.jpg  -> identity = "Alghany"
                Rizqi Setiawan/
                    1.jpg              -> identity = "Rizqi Setiawan"

        Flat root (legacy, still supported):
            data/known_faces/
                RIAN_HERI.jpg          -> identity = "Rian Heri" (normalize_name)

        Uses fast startup .npz caching when the recursive file signature matches.
        Returns total number of successfully indexed photos.
        """
        self.known_embeddings.clear()
        if not self.known_faces_dir.exists():
            self.known_faces_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[FaceRecognizer] Created known faces directory at: {self.known_faces_dir}")
            return 0

        valid_extensions = {".jpg", ".jpeg", ".png"}

        # Recursive scan — picks up both root-level files and subfolder files.
        # Exclude hidden files and the cache file itself.
        image_files = [
            p for p in self.known_faces_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in valid_extensions
            and not p.name.startswith(".")
            and p != self.cache_path
        ]

        if not image_files:
            # Clean up stale cache if directory is now empty
            if self.cache_path.exists():
                try:
                    self.cache_path.unlink()
                except OSError:
                    pass
            logger.info(
                f"[FaceRecognizer] Known faces directory '{self.known_faces_dir}' is empty. "
                f"System will operate normally with 'Unknown' labels."
            )
            return 0

        dataset_sig = self._compute_dataset_signature(image_files)
        if self._load_cache(dataset_sig):
            return len(self.known_embeddings)

        logger.info(
            f"[FaceRecognizer] Indexing {len(image_files)} reference photo(s) from "
            f"{self.known_faces_dir} (including subfolders)..."
        )
        detector = self._get_detector()
        loaded_count = 0

        for img_path in sorted(image_files):
            try:
                # --- Identity label resolution ---
                # If image is inside a direct subfolder of known_faces/, use folder name.
                # If image is directly in known_faces/ root, use filename stem (legacy).
                parent = img_path.parent
                if parent == self.known_faces_dir:
                    # Flat root file: normalize filename stem
                    raw_label = img_path.stem
                    identity = self.normalize_name(raw_label)
                else:
                    # Subfolder file: use folder name directly (already human-readable)
                    # Still run through normalize_name to ensure consistent Title Case
                    identity = self.normalize_name(parent.name)

                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    logger.warning(f"[FaceRecognizer] Failed to read image: {img_path.relative_to(self.known_faces_dir)}")
                    continue

                h, w = img.shape[:2]
                detector.set_input_size((w, h))
                retval, faces = detector.detector.detect(img)

                if retval is None or faces is None or len(faces) == 0:
                    logger.warning(
                        f"[FaceRecognizer] No face detected in reference photo "
                        f"'{img_path.relative_to(self.known_faces_dir)}'. Skipping."
                    )
                    continue

                # Take highest scoring face
                best_face = max(faces, key=lambda f: float(f[-1]))
                aligned_face = self.recognizer.alignCrop(img, best_face)
                if aligned_face is not None and aligned_face.size > 0:
                    aligned_face = apply_clahe(aligned_face, clip_limit=2.0)
                feature = self.recognizer.feature(aligned_face)

                self.known_embeddings.append((identity, feature))
                loaded_count += 1
                logger.info(
                    f"[FaceRecognizer] Loaded reference face: '{identity}' "
                    f"(from {img_path.relative_to(self.known_faces_dir)})"
                )

            except Exception as e:
                logger.error(
                    f"[FaceRecognizer] Error processing reference photo "
                    f"{img_path.relative_to(self.known_faces_dir)}: {e}"
                )

        logger.info(
            f"[FaceRecognizer] Database indexing complete: {loaded_count}/{len(image_files)} faces loaded."
        )
        if loaded_count > 0:
            self._save_cache(dataset_sig)
        return loaded_count

    def extract_feature(
        self,
        frame: np.ndarray,
        face_data: Union[np.ndarray, Tuple[int, int, int, int], List],
        min_size: int = 24,
    ) -> Optional[np.ndarray]:
        """Align crop face and extract 128-D feature embedding vector.

        Skip feature extraction for distant/micro faces (width < min_size or height < min_size)
        to prevent unnecessary ONNX inference load on background rows.
        """
        try:
            if isinstance(face_data, (tuple, list)):
                face_arr = np.array(face_data, dtype=np.float32)
            else:
                face_arr = np.asarray(face_data, dtype=np.float32)

            # Minimum size gating: reject distant micro-faces (< 24px)
            if len(face_arr) >= 4:
                fw = float(face_arr[2])
                fh = float(face_arr[3])
                if fw < float(min_size) or fh < float(min_size):
                    return None

            aligned_face = self.recognizer.alignCrop(frame, face_arr)
            if aligned_face is not None and aligned_face.size > 0:
                aligned_face = apply_clahe(aligned_face, clip_limit=2.0)
            feature = self.recognizer.feature(aligned_face)
            return feature
        except Exception as e:
            logger.debug(f"[FaceRecognizer] Feature extraction failed: {e}")
            return None

    def recognize(
        self,
        frame: np.ndarray,
        face_data: Union[np.ndarray, Tuple[int, int, int, int], List],
        min_size: int = 24,
        check_frontal: bool = False,
    ) -> Tuple[str, float, str]:
        """Match detected face against database embeddings with minimum size & landmark quality gating.

        Returns:
            (name: str, confidence_score: float, display_label: str)
            e.g. ("Rian", 0.78, "Rian (78%)") or ("Unknown", 0.32, "Unknown") or ("Non-Frontal", 0.0, "Non-Frontal")
        """
        if not self.known_embeddings:
            return ("Unknown", 0.0, "Unknown")

        # 1. Landmark-Based Face Quality Gate (Bypass SFace on side-profile / extreme yaw)
        if check_frontal:
            is_frontal, reason = self.is_frontal_face(face_data)
            if not is_frontal:
                logger.debug(f"[FaceRecognizer] Bypassed non-frontal face: {reason}")
                return ("Non-Frontal", 0.0, "Non-Frontal")

        # 2. Fast dimension gate before array conversion (< 24px)
        if isinstance(face_data, (tuple, list)) and len(face_data) >= 4:
            if float(face_data[2]) < float(min_size) or float(face_data[3]) < float(min_size):
                return ("Unknown", 0.0, "Unknown")
        elif hasattr(face_data, "__len__") and len(face_data) >= 4:
            if float(face_data[2]) < float(min_size) or float(face_data[3]) < float(min_size):
                return ("Unknown", 0.0, "Unknown")

        feature = self.extract_feature(frame, face_data, min_size=min_size)
        if feature is None:
            return ("Unknown", 0.0, "Unknown")

        best_name: str = "Unknown"
        best_score: float = -1.0

        for name, known_feat in self.known_embeddings:
            try:
                score = float(self.recognizer.match(feature, known_feat, cv2.FaceRecognizerSF_FR_COSINE))
                if score > best_score:
                    best_score = score
                    best_name = name
            except Exception as e:
                logger.debug(f"[FaceRecognizer] Match comparison error: {e}")

        if best_score >= self.cosine_threshold:
            clean_name = self.normalize_name(best_name) if (best_name and best_name != "Unknown") else best_name
            if not clean_name or clean_name.strip().lower() == "unknown":
                return ("Unknown", max(0.0, best_score), "Unknown")
            pct = int(round(best_score * 100.0))
            label = f"{clean_name} ({pct}%)"
            return (clean_name, best_score, label)
        else:
            # Telemetry: log closest candidate even if rejected below threshold.
            # Helps operators calibrate dataset / threshold without touching the HUD.
            if best_score > 0.0 and best_name != "Unknown":
                logger.info(
                    f"[DEBUG-FACE] Wajah mirip dengan '{best_name}' "
                    f"({best_score:.3f}) tapi ditolak karena < {self.cosine_threshold:.2f}"
                )
            return ("Unknown", max(0.0, best_score), "Unknown")
