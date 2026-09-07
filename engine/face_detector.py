"""OpenCV YuNet Lightweight Face Detector with Outside-Zone Filtering & Temporal Stabilization."""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import urllib.request
import urllib.error

import cv2
import numpy as np

logger = logging.getLogger("smart_cctv")

# Official OpenCV Zoo YuNet weights URL
YUNET_DEFAULT_URL: str = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
EXPECTED_MIN_BYTES: int = 200_000  # Official file is ~232 KB


def _compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes (x, y, w, h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = float(inter_w * inter_h)
    if inter_area <= 0.0:
        return 0.0
    union_area = float(w1 * h1 + w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


class YuNetFaceDetector:
    """Lightweight CPU-based face detector with anti-parallax boundary filtering and temporal smoothing."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        score_threshold: float = 0.48,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
        input_size: Tuple[int, int] = (640, 480),
        auto_download: bool = True,
        max_missed_frames: int = 6,
        ema_alpha: float = 0.45,
        alternate_inference: bool = True,
        resize_interpolation: int = cv2.INTER_LINEAR,
    ) -> None:
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self._current_input_size: Tuple[int, int] = input_size

        # Temporal Smoothing & Retention Buffer
        self._face_buffer: Dict[int, Dict[str, Any]] = {}
        self._next_face_id: int = 1
        self._max_missed_frames: int = max_missed_frames  # ~1.8s grace period to eliminate flickering
        self._ema_alpha: float = ema_alpha                # Smooth bounding box transitions

        # Alternating Dual-YuNet Inference State (Cuts CPU inference load in half)
        self.alternate_inference: bool = bool(alternate_inference)
        self._eval_cycle: int = 0
        self.resize_interpolation: int = resize_interpolation

        if model_path:
            self.model_path = Path(model_path)
        else:
            base_dir = Path(__file__).resolve().parent.parent
            self.model_path = base_dir / "models" / "face_detection_yunet_2023mar.onnx"

        # Auto-download if missing
        if not self.model_path.exists():
            if auto_download:
                self._download_weights(self.model_path)
            else:
                raise FileNotFoundError(f"YuNet model weights not found at: {self.model_path}")

        # Initialize OpenCV FaceDetectorYN
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=self._current_input_size,
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
            )
            logger.info(
                f"[YuNetFaceDetector] Loaded model from {self.model_path.name} "
                f"(score_thresh={self.score_threshold}, input={self._current_input_size})"
            )
        except Exception as e:
            logger.error(f"[YuNetFaceDetector] Failed to initialize FaceDetectorYN: {e}")
            raise RuntimeError(f"FaceDetectorYN initialization failed: {e}") from e

    def _download_weights(self, target_path: Path, max_retries: int = 3, timeout_sec: float = 15.0) -> None:
        """Download YuNet weights with retry, timeout, and file integrity validation."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_suffix(".tmp")

        logger.info(f"[YuNetFaceDetector] Weights not found. Downloading YuNet model to {target_path}...")
        headers = {"User-Agent": "Smart-CCTV/2.0 FaceDetectorYN"}

        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(YUNET_DEFAULT_URL, headers=headers)
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

                logger.info(f"[YuNetFaceDetector] Download complete ({len(data):,} bytes) on attempt {attempt}.")
                return

            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, IOError) as e:
                logger.warning(
                    f"[YuNetFaceDetector] Download attempt {attempt}/{max_retries} failed: {e}. "
                    + ("Retrying..." if attempt < max_retries else "No more retries.")
                )
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except Exception:
                        pass
                time.sleep(1.5)

        raise RuntimeError(
            f"Failed to download YuNet model weights from {YUNET_DEFAULT_URL} after {max_retries} attempts. "
            f"Please download 'face_detection_yunet_2023mar.onnx' manually and place it in {target_path.parent}."
        )

    def set_input_size(self, size: Tuple[int, int]) -> None:
        """Dynamically update input size if frame resolution changes."""
        if self._current_input_size != size:
            self._current_input_size = size
            self.detector.setInputSize(size)

    def detect(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float, np.ndarray]]:
        """Run raw face detection on given BGR frame.

        Returns list of ((x, y, w, h), score, raw_face).
        """
        fh, fw = frame.shape[:2]
        self.set_input_size((fw, fh))

        retval, faces = self.detector.detect(frame)
        if retval is None or faces is None or len(faces) == 0:
            return []

        results: List[Tuple[Tuple[int, int, int, int], float, np.ndarray]] = []
        for face in faces:
            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            score = float(face[-1])
            # Clamp bbox to image boundaries
            x = max(0, min(x, fw - 1))
            y = max(0, min(y, fh - 1))
            w = max(1, min(w, fw - x))
            h = max(1, min(h, fh - y))
            results.append(((x, y, w, h), score, face))

        return results

    def detect_crop(
        self,
        display_frame: np.ndarray,
        crop_roi: Union[Tuple[int, int, int, int], List[int]],
        target_crop_w: int = 640,
    ) -> List[Dict[str, Any]]:
        """Run YuNet face detection on a focused sub-frame crop directly from display_frame (1080p).

        The crop is proportionally resized to target_crop_w keeping the aspect ratio intact.
        Detected bounding boxes and landmarks are mapped back to:
        - Canvas display_frame (1080p) coordinates: bbox_1080 & raw_face_1080
        - Canonical 640x480 space: bbox_640 & raw_face_640 (for HUD rendering & tracking)
        """
        disp_h, disp_w = display_frame.shape[:2]
        if disp_w <= 0 or disp_h <= 0 or len(crop_roi) < 4:
            return []

        # Adapt 1080p reference coordinates [x1, y1, x2, y2] to display_frame actual size
        scale_x_ref = disp_w / 1920.0
        scale_y_ref = disp_h / 1080.0

        x1 = int(round(crop_roi[0] * scale_x_ref))
        y1 = int(round(crop_roi[1] * scale_y_ref))
        x2 = int(round(crop_roi[2] * scale_x_ref))
        y2 = int(round(crop_roi[3] * scale_y_ref))

        x1 = max(0, min(x1, disp_w - 20))
        y1 = max(0, min(y1, disp_h - 20))
        x2 = max(x1 + 20, min(x2, disp_w))
        y2 = max(y1 + 20, min(y2, disp_h))

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 20 or crop_h < 20:
            return []

        desk_crop = display_frame[y1:y2, x1:x2]

        # Resize proportionally to preserve aspect ratio
        aspect = float(crop_h) / float(crop_w)
        target_crop_h = max(16, int(round(target_crop_w * aspect)))
        if target_crop_h % 2 != 0:
            target_crop_h += 1

        resized_crop = cv2.resize(
            desk_crop, (target_crop_w, target_crop_h), interpolation=self.resize_interpolation
        )

        # Detect faces on high-resolution crop
        crop_raw_faces = self.detect(resized_crop)
        if not crop_raw_faces:
            return []

        scale_crop_to_disp_x = float(crop_w) / float(target_crop_w)
        scale_crop_to_disp_y = float(crop_h) / float(target_crop_h)

        scale_disp_to_640_x = 640.0 / float(disp_w)
        scale_disp_to_640_y = 480.0 / float(disp_h)

        results: List[Dict[str, Any]] = []
        for (lx, ly, lw, lh), score, raw_crop_face in crop_raw_faces:
            # 1. Map to display_frame (1080p) coordinates
            gx = int(round(lx * scale_crop_to_disp_x)) + x1
            gy = int(round(ly * scale_crop_to_disp_y)) + y1
            gw = int(round(lw * scale_crop_to_disp_x))
            gh = int(round(lh * scale_crop_to_disp_y))

            gx = max(0, min(gx, disp_w - 1))
            gy = max(0, min(gy, disp_h - 1))
            gw = max(1, min(gw, disp_w - gx))
            gh = max(1, min(gh, disp_h - gy))

            raw_face_1080 = raw_crop_face.copy().astype(np.float32)
            raw_face_1080[0] = gx
            raw_face_1080[1] = gy
            raw_face_1080[2] = gw
            raw_face_1080[3] = gh
            if len(raw_face_1080) >= 14:
                for k in range(4, 14, 2):
                    raw_face_1080[k] = raw_crop_face[k] * scale_crop_to_disp_x + x1
                    raw_face_1080[k + 1] = raw_crop_face[k + 1] * scale_crop_to_disp_y + y1

            # 2. Map to canonical 640x480 space (for HUD & tracking)
            bx_640 = int(round(gx * scale_disp_to_640_x))
            by_640 = int(round(gy * scale_disp_to_640_y))
            bw_640 = int(round(gw * scale_disp_to_640_x))
            bh_640 = int(round(gh * scale_disp_to_640_y))

            raw_face_640 = raw_crop_face.copy().astype(np.float32)
            raw_face_640[0] = bx_640
            raw_face_640[1] = by_640
            raw_face_640[2] = bw_640
            raw_face_640[3] = bh_640
            if len(raw_face_640) >= 14:
                for k in range(4, 14, 2):
                    raw_face_640[k] = raw_face_1080[k] * scale_disp_to_640_x
                    raw_face_640[k + 1] = raw_face_1080[k + 1] * scale_disp_to_640_y

            results.append({
                "bbox_640": (bx_640, by_640, bw_640, bh_640),
                "bbox_1080": (gx, gy, gw, gh),
                "score": score,
                "raw_face_640": raw_face_640,
                "raw_face_1080": raw_face_1080,
                "is_crop": True,
            })

        return results

    def detect_faces(
        self,
        frame: np.ndarray,
        display_frame: Optional[np.ndarray] = None,
        crop_roi: Optional[Union[Tuple[int, int, int, int], List[int]]] = None,
        zone_filter: Any = None,
        margin_px: float = 20.0,
        in_zone_person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
        target_crop_w: int = 640,
        alternate: Optional[bool] = None,
    ) -> List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray], Optional[np.ndarray], int]]:
        """Universal face detection merging full frame & focused desk crop with deduplication & temporal buffer.

        Supports alternating dual-YuNet inference (1 YuNet call per evaluation) to cut CPU latency in half.
        Returns list of (bbox_640, score, raw_face_640, raw_face_1080, face_id).
        """
        has_crop = display_frame is not None and crop_roi is not None and len(crop_roi) == 4
        use_alternate = self.alternate_inference if alternate is None else bool(alternate)

        run_crop = False
        run_full = False

        if not has_crop:
            run_full = True
        elif not use_alternate:
            run_crop = True
            run_full = True
        else:
            # Alternating Dual-YuNet: 1 inference per evaluation cycle
            if (self._eval_cycle % 2) == 0:
                run_crop = True
            else:
                run_full = True
            self._eval_cycle += 1

        candidates: List[Dict[str, Any]] = []

        # 1. Run Focused Desk Crop Face Detection (High Resolution)
        crop_detections: List[Dict[str, Any]] = []
        if run_crop and has_crop:
            crop_detections = self.detect_crop(
                display_frame=display_frame,
                crop_roi=crop_roi,
                target_crop_w=target_crop_w,
            )
            candidates.extend(crop_detections)

        # 2. Run Full-Frame Face Detection (Transit / Koridor / Outside Walkway)
        if run_full:
            all_full_faces = self.detect(frame)
            disp_h, disp_w = display_frame.shape[:2] if display_frame is not None else (480, 640)
            scale_640_to_disp_x = float(disp_w) / 640.0
            scale_640_to_disp_y = float(disp_h) / 480.0

            # Collect known crop boxes (from current detections + buffered crop faces)
            known_crop_boxes = [cd["bbox_640"] for cd in crop_detections]
            for bface in self._face_buffer.values():
                if bface.get("is_crop", False):
                    known_crop_boxes.append(bface["bbox"])

            for item in all_full_faces:
                if len(item) >= 3:
                    (fx, fy, fw, fh), score, raw_face = item[0], item[1], item[2]
                else:
                    (fx, fy, fw, fh), score = item[0], item[1]
                    raw_face = None

                fcx = fx + (fw // 2)
                fcy = fy + (fh // 2)

                # Deduplicate against high-resolution crop detections (current & buffered)
                is_duplicate = False
                for cbx, cby, cbw, cbh in known_crop_boxes:
                    ccx = cbx + (cbw // 2)
                    ccy = cby + (cbh // 2)
                    dist = ((fcx - ccx) ** 2 + (fcy - ccy) ** 2) ** 0.5
                    iou = _compute_iou((fx, fy, fw, fh), (cbx, cby, cbw, cbh))
                    if dist < 40.0 or iou > 0.20:
                        is_duplicate = True
                        break

                if is_duplicate:
                    # Prefer high-resolution crop detection
                    continue

                # Scale full-frame landmarks/bbox to 1080p display_frame space
                raw_face_1080 = None
                if raw_face is not None and display_frame is not None:
                    raw_face_1080 = raw_face.copy().astype(np.float32)
                    raw_face_1080[0] = int(round(raw_face[0] * scale_640_to_disp_x))
                    raw_face_1080[1] = int(round(raw_face[1] * scale_640_to_disp_y))
                    raw_face_1080[2] = int(round(raw_face[2] * scale_640_to_disp_x))
                    raw_face_1080[3] = int(round(raw_face[3] * scale_640_to_disp_y))
                    if len(raw_face_1080) >= 14:
                        for k in range(4, 14, 2):
                            raw_face_1080[k] = raw_face[k] * scale_640_to_disp_x
                            raw_face_1080[k + 1] = raw_face[k + 1] * scale_640_to_disp_y

                candidates.append({
                    "bbox_640": (fx, fy, fw, fh),
                    "score": score,
                    "raw_face_640": raw_face,
                    "raw_face_1080": raw_face_1080,
                    "is_crop": False,
                })

        # 3. Temporal Smoothing & Grace Period Tracking across all valid faces
        matched_buffer_ids = set()
        for cand in candidates:
            fx, fy, fw, fh = cand["bbox_640"]
            score = cand["score"]
            raw_face_640 = cand.get("raw_face_640")
            raw_face_1080 = cand.get("raw_face_1080")
            is_crop = cand.get("is_crop", False)

            fcx = fx + (fw // 2)
            fcy = fy + (fh // 2)

            # Match with closest buffered face
            best_fid = None
            best_dist = 60.0  # max 60 px matching distance

            for fid, bface in self._face_buffer.items():
                if fid in matched_buffer_ids:
                    continue
                bx, by, bw, bh = bface["bbox"]
                bcx = bx + (bw // 2)
                bcy = by + (bh // 2)
                dist = ((fcx - bcx) ** 2 + (fcy - bcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_fid = fid

            if best_fid is not None:
                # Update matched face with EMA coordinate smoothing
                old_x, old_y, old_w, old_h = self._face_buffer[best_fid]["bbox"]
                sm_x = int(self._ema_alpha * fx + (1.0 - self._ema_alpha) * old_x)
                sm_y = int(self._ema_alpha * fy + (1.0 - self._ema_alpha) * old_y)
                sm_w = int(self._ema_alpha * fw + (1.0 - self._ema_alpha) * old_w)
                sm_h = int(self._ema_alpha * fh + (1.0 - self._ema_alpha) * old_h)
                self._face_buffer[best_fid]["bbox"] = (sm_x, sm_y, sm_w, sm_h)
                self._face_buffer[best_fid]["score"] = score
                self._face_buffer[best_fid]["missed_frames"] = 0
                if raw_face_640 is not None:
                    self._face_buffer[best_fid]["raw_face"] = raw_face_640
                if raw_face_1080 is not None:
                    self._face_buffer[best_fid]["raw_face_1080"] = raw_face_1080
                self._face_buffer[best_fid]["is_crop"] = is_crop
                matched_buffer_ids.add(best_fid)
            else:
                # Register new face in buffer
                self._face_buffer[self._next_face_id] = {
                    "bbox": (fx, fy, fw, fh),
                    "score": score,
                    "missed_frames": 0,
                    "raw_face": raw_face_640,
                    "raw_face_1080": raw_face_1080,
                    "is_crop": is_crop,
                }
                matched_buffer_ids.add(self._next_face_id)
                self._next_face_id += 1

        # Increment missed frames only for unobserved buffered faces whose domain was active
        stale_fids = []
        for fid, bface in self._face_buffer.items():
            if fid not in matched_buffer_ids:
                is_crop_face = bface.get("is_crop", False)
                domain_evaluated = (is_crop_face and run_crop) or (not is_crop_face and run_full)
                if domain_evaluated:
                    bface["missed_frames"] += 1
                    if bface["missed_frames"] > self._max_missed_frames:
                        stale_fids.append(fid)

        for fid in stale_fids:
            del self._face_buffer[fid]

        # Return all active/retained faces from buffer
        return [
            (
                bface["bbox"],
                bface["score"],
                bface.get("raw_face"),
                bface.get("raw_face_1080"),
                fid,
            )
            for fid, bface in self._face_buffer.items()
        ]

    def detect_outside_faces(
        self,
        frame: np.ndarray,
        zone_filter: Any = None,
        margin_px: float = 20.0,
        in_zone_person_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Tuple[Tuple[int, int, int, int], float]]:
        """Backward-compatibility alias for detect_faces (Universal Face Detection)."""
        res = self.detect_faces(
            frame=frame,
            zone_filter=zone_filter,
            margin_px=margin_px,
            in_zone_person_boxes=in_zone_person_boxes,
        )
        return [(item[0], item[1]) for item in res]

