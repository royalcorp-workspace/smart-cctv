"""OpenCV SFace Face Recognition Engine with Local Photo Database."""

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
        cosine_threshold: float = 0.50,
        auto_download: bool = True,
        detector: Optional[Any] = None,
    ) -> None:
        self.cosine_threshold: float = float(cosine_threshold)
        base_dir = Path(__file__).resolve().parent.parent

        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = base_dir / "models" / "face_recognition_sface_2021dec.onnx"

        if known_faces_dir:
            self.known_faces_dir = Path(known_faces_dir)
        else:
            self.known_faces_dir = base_dir / "data" / "known_faces"

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

    def load_database(self) -> int:
        """Scan data/known_faces/, extract 128-D embeddings, and cache in memory.

        Returns total number of successfully indexed photos.
        """
        self.known_embeddings.clear()
        if not self.known_faces_dir.exists():
            self.known_faces_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[FaceRecognizer] Created known faces directory at: {self.known_faces_dir}")
            return 0

        valid_extensions = {".jpg", ".jpeg", ".png"}
        image_files = [
            p for p in self.known_faces_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_extensions
        ]

        if not image_files:
            logger.info(
                f"[FaceRecognizer] Known faces directory '{self.known_faces_dir}' is empty. "
                f"System will operate normally with 'Unknown' labels."
            )
            return 0

        logger.info(f"[FaceRecognizer] Indexing {len(image_files)} reference photo(s) from {self.known_faces_dir}...")
        detector = self._get_detector()
        loaded_count = 0

        for img_path in sorted(image_files):
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    logger.warning(f"[FaceRecognizer] Failed to read image: {img_path.name}")
                    continue

                h, w = img.shape[:2]
                detector.set_input_size((w, h))
                retval, faces = detector.detector.detect(img)

                if retval is None or faces is None or len(faces) == 0:
                    logger.warning(
                        f"[FaceRecognizer] No face detected in reference photo '{img_path.name}'. Skipping."
                    )
                    continue

                # Take highest scoring face
                best_face = max(faces, key=lambda f: float(f[-1]))
                aligned_face = self.recognizer.alignCrop(img, best_face)
                feature = self.recognizer.feature(aligned_face)

                # Format identity name from filename (e.g. "rian.jpeg" -> "Rian", "andi_wijaya_2.png" -> "Andi Wijaya")
                raw_name = img_path.stem
                clean_name = re.sub(r"_\d+$", "", raw_name).replace("_", " ").strip().title()
                if not clean_name:
                    clean_name = raw_name

                self.known_embeddings.append((clean_name, feature))
                loaded_count += 1
                logger.info(f"[FaceRecognizer] Loaded reference face: '{clean_name}' (from {img_path.name})")

            except Exception as e:
                logger.error(f"[FaceRecognizer] Error processing reference photo {img_path.name}: {e}")

        logger.info(
            f"[FaceRecognizer] Database indexing complete: {loaded_count}/{len(image_files)} faces loaded."
        )
        return loaded_count

    def extract_feature(
        self,
        frame: np.ndarray,
        face_data: Union[np.ndarray, Tuple[int, int, int, int], List],
        min_size: int = 28,
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

            # Minimum size gating: reject distant micro-faces
            if len(face_arr) >= 4:
                fw = float(face_arr[2])
                fh = float(face_arr[3])
                if fw < float(min_size) or fh < float(min_size):
                    return None

            aligned_face = self.recognizer.alignCrop(frame, face_arr)
            feature = self.recognizer.feature(aligned_face)
            return feature
        except Exception as e:
            logger.debug(f"[FaceRecognizer] Feature extraction failed: {e}")
            return None

    def recognize(
        self,
        frame: np.ndarray,
        face_data: Union[np.ndarray, Tuple[int, int, int, int], List],
        min_size: int = 28,
    ) -> Tuple[str, float, str]:
        """Match detected face against database embeddings with minimum size gating.

        Returns:
            (name: str, confidence_score: float, display_label: str)
            e.g. ("Rian", 0.78, "Rian (78%)") or ("Unknown", 0.32, "Unknown")
        """
        if not self.known_embeddings:
            return ("Unknown", 0.0, "Unknown")

        # Fast dimension gate before array conversion
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
            pct = int(round(best_score * 100.0))
            label = f"{best_name} ({pct}%)"
            return (best_name, best_score, label)
        else:
            return ("Unknown", max(0.0, best_score), "Unknown")
