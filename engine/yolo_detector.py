"""YOLO11 Nano Object Detection Engine accelerated with Intel OpenVINO."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from engine.logger import logger


class YOLOOpenVINODetector:
    """YOLO11 Nano detector optimized for Intel CPU via OpenVINO."""

    TARGET_CLASSES: Dict[int, str] = {
        0: "person",
        24: "backpack",
        26: "handbag",
        28: "suitcase",
    }

    CLASS_CONFIDENCE_THRESHOLDS: Dict[int, float] = {
        0: 0.38,   # person: calibrated to filter static door frame artifacts while detecting walking persons
        24: 0.18,  # backpack: calibrated for floor/lying bags
        26: 0.18,  # handbag: calibrated for floor/lying bags
        28: 0.18,  # suitcase: calibrated for floor/lying bags
    }

    def __init__(
        self,
        model_name: str = "yolo11n",
        device: str = "cpu",
        confidence_threshold: float = 0.20,
        bag_confidence_threshold: Optional[float] = None,
        imgsz: int = 640,
        models_dir: Optional[Path] = None,
    ) -> None:
        self.device: str = device
        self.conf: float = confidence_threshold
        self.imgsz: int = imgsz
        self.model_name: str = model_name

        bag_target_conf = bag_confidence_threshold if bag_confidence_threshold is not None else min(0.18, self.conf)
        # Align bag thresholds with confidence_threshold
        for bag_cid in (24, 26, 28):
            self.CLASS_CONFIDENCE_THRESHOLDS[bag_cid] = bag_target_conf

        if models_dir is None:
            models_dir = Path(__file__).resolve().parent.parent
        self.models_dir: Path = models_dir

        self.pt_path: Path = self.models_dir / f"{model_name}.pt"
        self.openvino_dir: Path = self.models_dir / f"{model_name}_openvino_model"

        self.model: YOLO = self._load_or_export_model()

        # Warm up engine
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = self.detect(dummy)
        logger.info(
            f"YOLOOpenVINODetector ({self.model_name}) initialized successfully on device='{device}' "
            f"(imgsz={imgsz}, base_conf={self.conf:.2f}, person_conf={self.CLASS_CONFIDENCE_THRESHOLDS.get(0, 0.38):.2f}, "
            f"bag_conf={self.CLASS_CONFIDENCE_THRESHOLDS.get(24, 0.25):.2f})"
        )

    def _load_or_export_model(self) -> YOLO:
        """Export to OpenVINO if not present, and load optimized model."""
        if not self.openvino_dir.exists():
            logger.info(f"OpenVINO model not found at {self.openvino_dir}. Exporting from {self.pt_path}...")
            source = str(self.pt_path) if self.pt_path.exists() else f"{self.model_name}.pt"
            base_model = YOLO(source)
            base_model.export(format="openvino", half=False)

        if self.openvino_dir.exists():
            logger.info(f"Loading OpenVINO optimized model from: {self.openvino_dir}")
            return YOLO(str(self.openvino_dir), task="detect")
        else:
            logger.warning(f"OpenVINO export folder not found. Falling back to PyTorch model.")
            source = str(self.pt_path) if self.pt_path.exists() else f"{self.model_name}.pt"
            return YOLO(source, task="detect")

    def detect(
        self, frame: np.ndarray
    ) -> List[Tuple[Tuple[int, int, int, int], Tuple[int, int], Tuple[int, int], float, int, str]]:
        """Run inference on frame.

        Returns list of tuples:
            ( (x, y, w, h), (cx, cy), (ref_x, ref_y), confidence, class_id, class_name )
        """
        # Run base inference with low threshold (0.25) and class-agnostic NMS
        results = self.model(
            frame,
            device=self.device,
            imgsz=self.imgsz,
            conf=0.25,
            classes=list(self.TARGET_CLASSES.keys()),
            agnostic_nms=True,
            verbose=False,
        )

        detections: List[Tuple[Tuple[int, int, int, int], Tuple[int, int], Tuple[int, int], float, int, str]] = []
        if not results:
            return detections

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        orig_h, orig_w = frame.shape[:2]

        for box, conf_val, cls_val in zip(xyxy, confs, classes):
            cid = int(cls_val)
            cconf = float(conf_val)

            # Stratified per-class confidence filter
            req_conf = self.CLASS_CONFIDENCE_THRESHOLDS.get(cid, self.conf)
            if cconf < req_conf:
                continue

            x1 = max(0, min(orig_w - 1, int(box[0])))
            y1 = max(0, min(orig_h - 1, int(box[1])))
            x2 = max(0, min(orig_w - 1, int(box[2])))
            y2 = max(0, min(orig_h - 1, int(box[3])))
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            cx = x1 + w // 2
            cy = y1 + h // 2
            cname = self.TARGET_CLASSES.get(cid, "object")

            # Accurate reference point:
            # Person: center-bottom / feet (contact point with floor)
            # Bag: center of mass
            if cid == 0:  # person
                ref_point = (cx, min(orig_h - 1, y2 - 2))
            else:
                ref_point = (cx, cy)

            detections.append(((x1, y1, w, h), (cx, cy), ref_point, cconf, cid, cname))

        # Enforce Class-Agnostic NMS to eliminate overlapping duplicate boxes (e.g. backpack vs handbag)
        if len(detections) > 1:
            boxes_xywh = [[d[0][0], d[0][1], d[0][2], d[0][3]] for d in detections]
            scores = [float(d[3]) for d in detections]
            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_xywh,
                scores=scores,
                score_threshold=0.20,
                nms_threshold=0.45,
            )
            if len(indices) > 0:
                indices_flat = indices.flatten()
                detections = [detections[i] for i in indices_flat]

        return detections
