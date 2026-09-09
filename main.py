"""Smart CCTV 2.0 - Multi-Camera Pipeline Orchestrator."""

import argparse
import concurrent.futures
import datetime
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from engine.config_loader import load_camera_config
from engine.dual_subtractor import DualSubtractor
from engine.logger import logger
from engine.retention import cleanup_old_records, DiskGuardWorker
from engine.rtsp_stream import ThreadedCapture
from engine.face_detector import YuNetFaceDetector
from engine.face_recognizer import FaceRecognizer
from engine.tracker import CentroidTracker, TrackedObject, compute_bbox_iou
from engine.yolo_detector import YOLOOpenVINODetector
from engine.zone_filter import ZoneFilter
from notification.local_alert import GlobalAudioWorker, VisualHUD
from notification.telegram_alert import TelegramNotifier
from storage.db import init_db, log_event, resolve_event, update_event_clip
from web.buffer import MultiCameraBuffer
from web.server import DashboardServer


class RingBufferRecorder:
    """Maintains an in-memory circular buffer of JPEG-compressed frames (~15-25 MB RAM).
    Exports browser-compatible H.264 MP4 video clips asynchronously upon incident breach."""

    def __init__(self, maxlen: int = 250, target_size: Tuple[int, int] = (1280, 720), fps: float = 13.0) -> None:
        self.maxlen = maxlen
        self.target_size = target_size
        self.fps = fps
        self._buffer: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def push_frame(self, frame: np.ndarray) -> None:
        """Downscale frame to 720p and compress to JPEG bytes to maintain minimal RAM usage."""
        if frame is None or frame.size == 0:
            return
        try:
            if frame.shape[1] != self.target_size[0] or frame.shape[0] != self.target_size[1]:
                scaled = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
            else:
                scaled = frame
            success, encoded = cv2.imencode(".jpg", scaled, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                with self._lock:
                    self._buffer.append(encoded.tobytes())
        except Exception as e:
            logger.debug(f"[RingBuffer] Error encoding frame: {e}")

    def export_clip(self, output_path: Path) -> bool:
        """Decompress JPEG frames and export to an H.264 MP4 video clip."""
        with self._lock:
            frames_to_write = list(self._buffer)

        if not frames_to_write:
            logger.warning(f"[RingBuffer] No frames to export to {output_path}")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        w, h = self.target_size

        # Codec selection: try avc1 first (standard HTML5 H.264), fallback to mp4v if needed
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (w, h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (w, h))

        if not writer.isOpened():
            logger.error(f"[RingBuffer] Failed to initialize VideoWriter for {output_path}")
            return False

        try:
            for fb in frames_to_write:
                dec = cv2.imdecode(np.frombuffer(fb, dtype=np.uint8), cv2.IMREAD_COLOR)
                if dec is not None:
                    if dec.shape[1] != w or dec.shape[0] != h:
                        dec = cv2.resize(dec, (w, h))
                    writer.write(dec)
            writer.release()
            logger.info(f"[RingBuffer] Incident clip exported ({len(frames_to_write)} frames) -> {output_path}")
            return True
        except Exception as e:
            logger.error(f"[RingBuffer] Failed writing video clip: {e}")
            if writer:
                writer.release()
            return False


class CameraPipeline:
    """Encapsulates autonomous capture, processing, tracking, and alerting for a single camera workspace."""

    def __init__(self, camera_dir: Path) -> None:
        self.camera_dir: Path = camera_dir
        self.config_path: Path = camera_dir / "config.json"
        self.roi_path: Path = camera_dir / "roi_zones.json"
        self.snapshots_dir: Path = camera_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._load_configurations()

        # Register in shared multi-camera buffer for Web Dashboard
        MultiCameraBuffer.get_instance().register_camera(
            self.camera_id,
            self.config.get("name", self.camera_id),
        )

        # Target resolution (Display / Capture frame)
        self.target_resolution: Optional[Tuple[int, int]] = None
        cfg_res = self.config.get("target_resolution")
        if cfg_res and len(cfg_res) == 2:
            self.target_resolution = (int(cfg_res[0]), int(cfg_res[1]))

        # Decoupled AI inference resolution (native 16:9 aspect 640x360 for zero-distortion CPU inference)
        self.infer_resolution: Tuple[int, int] = (640, 360)

        # Initialize Subcomponents
        self.capture = ThreadedCapture(source=self.config.get("source", 0))
        
        # YOLO11 Nano with Intel OpenVINO CPU Acceleration
        detector_cfg = self.config.get("detector", {})
        conf_thresh = float(detector_cfg.get("confidence_threshold", detector_cfg.get("base_conf", 0.20)))
        bag_conf = float(detector_cfg.get("bag_conf", conf_thresh))
        model_name = detector_cfg.get("model_name", "yolo11n")
        self.detector = YOLOOpenVINODetector(
            model_name=model_name,
            device="cpu",
            confidence_threshold=conf_thresh,
            bag_confidence_threshold=bag_conf,
            imgsz=640,
        )
        # YOLO 2-frame inference stride state (cuts object detector CPU load in half)
        self._yolo_frame_index: int = 0
        self._cached_yolo_results: List[Any] = []

        self.subtractor = DualSubtractor(
            slow_history=5000,
            slow_learning_rate=0.0001,
            fast_history=40,
            fast_learning_rate=0.05,
        )
        self.zone_filter = ZoneFilter(
            zones=self.roi_zones,
            zone_configs=self.config.get("zones", {}),
            frame_shape=(self.infer_resolution[1], self.infer_resolution[0]),
            base_resolution=self.roi_base_resolution,
        )
        self.tracker = CentroidTracker(
            max_distance_px=80.0,
            movement_threshold_px=15.0,
            anchor_radius_px=40.0,
            flicker_tolerance_sec=2.0,
            max_disappeared_sec=12.0,
            max_age_frames=150,
            ema_alpha=0.3,
            spatial_memory_ttl_sec=600.0,          # 10 minutes: give YOLO enough time to re-detect
            spatial_match_distance_px=60.0,
            stationary_max_age_frames=900,          # ~67s latching at 13fps before purge
            stationary_max_disappeared_sec=75.0,    # sync with 900 frames at ~13fps
        )
        self._kernel_close_large = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        self._kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.audio_worker = GlobalAudioWorker()
        self.telegram_notifier: TelegramNotifier = TelegramNotifier.get_instance()
        self.io_executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._last_dwell_log_time: float = 0.0

        # Threading state
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._display_lock = threading.Lock()
        self._latest_display_frame: Optional[np.ndarray] = None
        
        # Universal Face Detector (OpenCV YuNet across all areas)
        face_cfg = self.config.get("face_detector", {})
        self.face_detection_enabled: bool = bool(face_cfg.get("enabled", True))
        self.face_detector: Optional[YuNetFaceDetector] = None
        self._face_detect_interval: int = int(face_cfg.get("detect_interval_frames", 3))
        self._cached_faces: List[Tuple[Tuple[int, int, int, int], float, str]] = []
        self._face_frame_index: int = 0
        self._face_burst_until_time: float = 0.0
        self._last_person_seen_time: float = 0.0
        self._prev_stationary_bag_ids: set = set()

        # Focused Desk Sub-Frame Crop ROI on 1080p canvas with dynamic fallback
        face_crop_cfg = face_cfg.get("face_roi_crop", [100, 40, 1250, 520])
        if isinstance(face_crop_cfg, (list, tuple)) and len(face_crop_cfg) == 4:
            self.face_roi_crop: Tuple[int, int, int, int] = (
                int(face_crop_cfg[0]),
                int(face_crop_cfg[1]),
                int(face_crop_cfg[2]),
                int(face_crop_cfg[3]),
            )
        else:
            self.face_roi_crop = (100, 40, 1250, 520)

        # Recognition caching for stationary faces (eliminates redundant SFace calls & preserves 8-10 FPS)
        self._face_recog_cache: Dict[int, Dict[str, Any]] = {}
        self._person_face_recog: Dict[int, Dict[str, Any]] = {}

        # Anti-ghost tile suppression tracking for retrieved bags
        self._recently_retrieved_bags: Dict[int, Dict[str, Any]] = {}

        if self.face_detection_enabled:
            try:
                face_model = face_cfg.get("model_path", None)
                score_th = float(face_cfg.get("score_threshold", 0.62))
                nms_th = float(face_cfg.get("nms_threshold", 0.30))
                alternate_infer = bool(face_cfg.get("alternate_inference", True))
                min_face_sz = int(face_cfg.get("min_face_size", 24))
                aspect_range = face_cfg.get("aspect_ratio_range", [0.6, 1.4])
                self.face_detector = YuNetFaceDetector(
                    model_path=face_model,
                    score_threshold=score_th,
                    nms_threshold=nms_th,
                    input_size=(640, 360),
                    auto_download=True,
                    max_missed_frames=6,
                    alternate_inference=alternate_infer,
                    min_face_size=min_face_sz,
                    aspect_ratio_range=aspect_range,
                )
                logger.info(
                    f"[{self.camera_id}] YuNet Face Detector initialized "
                    f"(interval={self._face_detect_interval} frames, alternate={alternate_infer}, input_size=(640, 360), "
                    f"score_thresh={score_th}, min_size={min_face_sz}px)."
                )
            except Exception as e:
                logger.warning(f"[{self.camera_id}] Failed to initialize YuNet Face Detector: {e}. Running without face detection.")
                self.face_detector = None
                self.face_detection_enabled = False

        # Universal Face Recognizer (OpenCV SFace with Local Photo Database)
        face_rec_cfg = self.config.get("face_recognizer", {})
        self.face_recognition_enabled: bool = bool(face_rec_cfg.get("enabled", True))
        self.face_recognizer: Optional[FaceRecognizer] = None

        if self.face_recognition_enabled and self.face_detector is not None:
            try:
                rec_model = face_rec_cfg.get("model_path", None)
                rec_dir = face_rec_cfg.get("known_faces_dir", None)
                cos_th = float(face_rec_cfg.get("cosine_threshold", 0.60))
                if cos_th > 1.0:
                    cos_th = cos_th / 100.0
                self.face_recognizer = FaceRecognizer(
                    model_path=rec_model,
                    known_faces_dir=rec_dir,
                    cosine_threshold=cos_th,
                    auto_download=True,
                    detector=self.face_detector,
                )
                logger.info(
                    f"[{self.camera_id}] SFace Face Recognizer initialized "
                    f"({len(self.face_recognizer.known_embeddings)} identities indexed, threshold={cos_th})."
                )
            except Exception as e:
                logger.warning(f"[{self.camera_id}] Failed to initialize Face Recognizer: {e}. Running without recognition.")
                self.face_recognizer = None
                self.face_recognition_enabled = False

        # FPS Calculation
        self._frame_count: int = 0
        self._fps_start_time: float = time.time()
        self.current_fps: float = 0.0

        self.active_db_events: Dict[int, int] = {}
        self._zone_lock = threading.Lock()
        self.ring_buffer = RingBufferRecorder(maxlen=250, target_size=(1280, 720), fps=13.0)

        # Register pipeline in MultiCameraBuffer for web control and atomic reload
        MultiCameraBuffer.get_instance().register_pipeline(self.camera_id, self)

    def reload_zones(self) -> Dict[str, Any]:
        """Thread-safely reload ROI zones from roi_zones.json and re-initialize ZoneFilter."""
        with self._zone_lock:
            self._load_configurations()
            self._unattended_zones_mask = None
            self.zone_filter = ZoneFilter(
                zones=self.roi_zones,
                zone_configs=self.config.get("zones", {}),
                frame_shape=(self.infer_resolution[1], self.infer_resolution[0]),
                base_resolution=self.roi_base_resolution,
            )
            logger.info(f"[{self.camera_id}] ZoneFilter atomically reloaded ({len(self.roi_zones)} active zones).")
            return {
                "camera_id": self.camera_id,
                "zones": self.roi_zones,
                "base_resolution": self.roi_base_resolution,
            }

    def _load_configurations(self) -> None:
        """Parse configuration and ROI zone files."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Missing camera configuration: {self.config_path}")

        self.config = load_camera_config(self.config_path)
        self.camera_id: str = self.config.get("camera_id", self.camera_dir.name)

        self.roi_zones: Dict[str, List[List[int]]] = {}
        self.roi_base_resolution: Tuple[int, int] = (1920, 1080)
        if self.roi_path.exists():
            with open(self.roi_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "base_resolution" in data and isinstance(data["base_resolution"], (list, tuple)):
                        self.roi_base_resolution = (int(data["base_resolution"][0]), int(data["base_resolution"][1]))
                    else:
                        # Auto-detect: if any coordinate x > 640 or y > 480 -> 1080p, else 640p
                        max_x = 0
                        max_y = 0
                        for k, v in data.items():
                            if isinstance(v, list) and not k.startswith("_") and k != "base_resolution":
                                for pt in v:
                                    if len(pt) >= 2:
                                        max_x = max(max_x, pt[0])
                                        max_y = max(max_y, pt[1])
                        if max_x > 640 or max_y > 480:
                            self.roi_base_resolution = (1920, 1080)
                        else:
                            self.roi_base_resolution = (640, 480)

                    self.roi_zones = {
                        k: v for k, v in data.items()
                        if isinstance(v, list) and not k.startswith("_") and k != "base_resolution"
                    }

    def start(self) -> "CameraPipeline":
        """Start capture and processing pipeline in background thread."""
        self._stopped.clear()
        self.capture.start()
        self._thread = threading.Thread(target=self._run_pipeline, name=f"Pipeline-{self.camera_id}", daemon=True)
        self._thread.start()
        return self

    def _resolve_event_async(
        self,
        event_id: int,
        zone_name: Optional[str] = None,
        track_id: Optional[int] = None,
        dwell_duration: Optional[float] = None,
        owner_name: Optional[str] = None,
    ) -> None:
        """I/O worker task: mark event resolved in SQLite and dispatch Telegram resolution."""
        try:
            resolve_event(event_id=event_id)
            logger.info(f"[{self.camera_id}] Event ID {event_id} marked as resolved.")
            if zone_name and track_id is not None and dwell_duration is not None:
                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.telegram_notifier.dispatch_resolution(
                    camera_id=self.camera_id,
                    zone_name=zone_name,
                    track_id=track_id,
                    dwell_duration=dwell_duration,
                    timestamp_str=timestamp_str,
                    owner_name=owner_name,
                )
        except Exception as e:
            logger.error(f"[{self.camera_id}] Async resolve failed for event {event_id}: {e}")

    def _associate_object_actors(
        self,
        active_tracks: List[TrackedObject],
        infer_frame: np.ndarray,
        clean_frame: np.ndarray,
    ) -> None:
        """Associate tracked objects (tas, ransel, koper, boks, paket, dll.) with nearest/contacting actor."""
        monitored_objects = [
            t for t in active_tracks
            if t.class_label != "person"
            and (getattr(t, "is_stationary", False) or getattr(t, "frame_count", 1) <= 15)
        ]
        if not monitored_objects:
            return

        persons = [
            t for t in active_tracks
            if t.class_label == "person" and getattr(t, "is_active_this_frame", False)
        ]

        scale_x = clean_frame.shape[1] / float(infer_frame.shape[1])
        scale_y = clean_frame.shape[0] / float(infer_frame.shape[0])

        for obj in monitored_objects:
            bx, by, bw, bh = obj.bbox
            bcx, bcy = obj.centroid
            obj_base = (bx + bw / 2.0, by + bh)
            adaptive_radius = max(60.0, min(120.0, float(bh * 2.5)))

            # Find nearest contacting person
            nearest_person = None
            min_dist = float("inf")
            for person in persons:
                px, py, pw, ph = person.bbox
                person_foot = (px + pw / 2.0, py + ph)
                dist = float(
                    ((obj_base[0] - person_foot[0]) ** 2 + (obj_base[1] - person_foot[1]) ** 2) ** 0.5
                )

                # Overlap or proximity check (radius <= 120 px or bbox overlap)
                is_contact = (
                    dist <= adaptive_radius
                    or (px <= bcx <= (px + pw) and py <= bcy <= (py + ph))
                    or (bx < px + pw and bx + bw > px and by < py + ph and by + bh > py)
                )
                if is_contact and dist < min_dist:
                    min_dist = dist
                    nearest_person = person

            if nearest_person is not None:
                px, py, pw, ph = nearest_person.bbox
                head_h = int(ph * 0.35)
                head_top = py
                head_bottom = py + head_h

                # 1. Extract Person Body Crop from Full Resolution Frame
                pad_px = int(pw * 0.10)
                pad_py = int(ph * 0.05)
                pcx1 = max(0, int((px - pad_px) * scale_x))
                pcy1 = max(0, int((py - pad_py) * scale_y))
                pcx2 = min(clean_frame.shape[1], int((px + pw + pad_px) * scale_x))
                pcy2 = min(clean_frame.shape[0], int((py + ph + pad_py) * scale_y))
                person_crop = clean_frame[pcy1:pcy2, pcx1:pcx2].copy() if (pcx2 > pcx1 and pcy2 > pcy1) else None

                # 2. Match with cached faces from YuNet
                matched_face = None
                for face_item in (self._cached_faces or []):
                    if len(face_item) >= 3:
                        f_bbox, f_score, f_label = face_item[0], face_item[1], face_item[2]
                    elif len(face_item) == 2:
                        f_bbox, f_score_or_lbl = face_item[0], face_item[1]
                        f_score = float(f_score_or_lbl) if isinstance(f_score_or_lbl, (int, float)) else 0.5
                        f_label = str(f_score_or_lbl) if isinstance(f_score_or_lbl, str) else "Unknown"
                    else:
                        continue

                    fx, fy, fw, fh = f_bbox
                    fcx = fx + fw // 2
                    fcy = fy + fh // 2

                    # Check if face center falls inside head region (with 10px margin)
                    if (px - 10) <= fcx <= (px + pw + 10) and (head_top - 10) <= fcy <= (head_bottom + 10):
                        matched_face = (f_bbox, f_score, f_label)
                        break

                if matched_face is not None:
                    f_bbox, f_score, f_label = matched_face
                    name = str(f_label) if f_label else "Unknown"
                    conf = float(f_score)
                    fx, fy, fw, fh = f_bbox
                    pad_w = int(fw * 0.25)
                    pad_h = int(fh * 0.25)
                    cx1 = max(0, int((fx - pad_w) * scale_x))
                    cy1 = max(0, int((fy - pad_h) * scale_y))
                    cx2 = min(clean_frame.shape[1], int((fx + fw + pad_w) * scale_x))
                    cy2 = min(clean_frame.shape[0], int((fy + fh + pad_h) * scale_y))
                    face_crop = clean_frame[cy1:cy2, cx1:cx2].copy() if (cx2 > cx1 and cy2 > cy1) else None
                else:
                    name = "Unknown"
                    conf = 0.5
                    cx1 = max(0, int((px - 5) * scale_x))
                    cy1 = max(0, int((head_top - 5) * scale_y))
                    cx2 = min(clean_frame.shape[1], int((px + pw + 5) * scale_x))
                    cy2 = min(clean_frame.shape[0], int(head_bottom * scale_y))
                    face_crop = clean_frame[cy1:cy2, cx1:cx2].copy() if (cx2 > cx1 and cy2 > cy1) else None

                # Update actor info if none exists or if upgraded from Unknown to a known name
                existing_actor = getattr(obj, "associated_face_meta", None) or getattr(obj, "last_owner_info", None)
                if existing_actor is None or (existing_actor.get("name") == "Unknown" and name != "Unknown"):
                    actor_meta = {
                        "name": name,
                        "confidence": conf,
                        "face_crop": face_crop,
                        "person_crop": person_crop,
                        "person_bbox": (px, py, pw, ph),
                        "timestamp": time.time(),
                    }
                    obj.associated_face_meta = actor_meta
                    obj.associated_face_crop = face_crop
                    obj.associated_person_crop = person_crop
                    obj.last_owner_info = actor_meta

    def _associate_bag_owners(
        self,
        active_tracks: List[TrackedObject],
        infer_frame: np.ndarray,
        clean_frame: np.ndarray,
    ) -> None:
        """Backward compatibility alias for _associate_object_actors."""
        self._associate_object_actors(active_tracks, infer_frame, clean_frame)


    @staticmethod
    def _is_bag_overlapping_person(
        bag_box: Tuple[int, int, int, int],
        bag_centroid: Tuple[int, int],
        person_boxes: List[Tuple[int, int, int, int]],
        iof_threshold: float = 0.60,
    ) -> bool:
        """Check if bag centroid is inside any person box or has IoF > 0.3 (carried bag)."""
        bx, by, bw, bh = bag_box
        bcx, bcy = bag_centroid
        bag_area = float(bw * bh)
        if bag_area <= 0:
            return True

        for (px, py, pw, ph) in person_boxes:
            # Check 1: Bag centroid inside person bounding box (with 5px margin)
            if (px - 5) <= bcx <= (px + pw + 5) and (py - 5) <= bcy <= (py + ph + 5):
                return True

            # Check 2: Intersection over Foreground (bag area) > 0.3
            ix1 = max(bx, px)
            iy1 = max(by, py)
            ix2 = min(bx + bw, px + pw)
            iy2 = min(by + bh, py + ph)

            if ix2 > ix1 and iy2 > iy1:
                intersection = float((ix2 - ix1) * (iy2 - iy1))
                if (intersection / bag_area) > iof_threshold:
                    return True

        return False

    def _run_pipeline(self) -> None:
        """Continuous camera frame processing loop."""
        last_valid_frame_time = time.time()
        while not self._stopped.is_set():
            ret, frame = self.capture.read(timeout=0.15)
            now = time.time()

            if not ret or frame is None:
                # Only enter offline state if actually disconnected or timeout exceeded (>2.5s)
                if not self.capture.is_connected or (now - last_valid_frame_time > 2.5):
                    # Evaluate tracker timeout during stream idle/disconnection
                    _, purged_tracks = self.tracker.update([], timestamp=now)
                    for purged in purged_tracks:
                        eid = self.active_db_events.pop(purged.track_id, purged.db_event_id)
                        if purged.is_triggered and eid is not None:
                            z_name = self.config.get("zones", {}).get(purged.zone_id, {}).get("name", purged.zone_id)
                            o_name = purged.last_owner_info.get("name") if getattr(purged, "last_owner_info", None) else None
                            self._resolve_event_async(eid, zone_name=z_name, track_id=purged.track_id, dwell_duration=purged.dwell_duration, owner_name=o_name)

                    # Render offline canvas using target resolution dimensions
                    target_w = self.target_resolution[0] if self.target_resolution else 1920
                    target_h = self.target_resolution[1] if self.target_resolution else 1080
                    placeholder = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    VisualHUD.render(
                        canvas=placeholder,
                        zones=self.roi_zones,
                        tracked_objects=[],
                        camera_id=self.camera_id,
                        fps=self.current_fps,
                        is_connected=False,
                        faces=[],
                    )
                    with self._display_lock:
                        self._latest_display_frame = placeholder

                    # Push offline placeholder frame to MultiCameraBuffer
                    MultiCameraBuffer.get_instance().update_frame(
                        camera_id=self.camera_id,
                        frame=placeholder,
                        telemetry={
                            "fps": 0.0,
                            "online": False,
                            "is_connected": False,
                            "rtsp_status": "Reconnecting" if (now - last_valid_frame_time > 2.5) else "Connecting",
                            "violations": 0,
                            "clear_area_count": 0,
                            "active_tracks": 0,
                            "identified_faces": [],
                        },
                        camera_name=self.config.get("name", self.camera_id),
                    )
                    time.sleep(0.05)
                else:
                    # Transient inter-frame delay on low FPS cameras
                    time.sleep(0.01)
                continue

            last_valid_frame_time = now

            # Expire recently retrieved bags older than 120.0s
            if self._recently_retrieved_bags:
                self._recently_retrieved_bags = {
                    tid: info for tid, info in self._recently_retrieved_bags.items()
                    if (now - info["time"]) <= 120.0
                }

            # Target resolution normalization (Display frame)
            if self.target_resolution is not None:
                tw, th = self.target_resolution
                if frame.shape[1] != tw or frame.shape[0] != th:
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

            # Decoupled resolution architecture:
            # 1. High-resolution clean copy for display and snapshots (Full HD 1080p)
            display_frame = frame.copy()
            raw_clean_frame = frame.copy()

            # Push clean frame into memory ring buffer (compressed to 720p JPEG in memory)
            self.ring_buffer.push_frame(raw_clean_frame)

            # 2. Scaled lightweight inference frame (640x480) for AI models (YOLO11n, MOG2, YuNet)
            infer_frame = cv2.resize(frame, self.infer_resolution, interpolation=cv2.INTER_LINEAR)

            # 1. YOLO11 Nano Object Detection on 640x360 inference frame with adaptive zero-delay stride
            # Zero delay (stride = 1) when persons are present or recently seen (< 2.0s); 2-frame stride when empty room
            has_recent_person = (now - self._last_person_seen_time) <= 2.0
            yolo_interval = 1 if has_recent_person else 2
            if (self._yolo_frame_index % yolo_interval) == 0:
                self._cached_yolo_results = self.detector.detect(infer_frame)
            self._yolo_frame_index += 1
            yolo_results = self._cached_yolo_results
            person_boxes = [bbox for bbox, _, _, _, cid, _ in yolo_results if cid == 0]
            # Also include any active person tracks to bridge single-frame detection drops
            for trk in self.tracker.objects.values():
                if trk.class_label == "person" and getattr(trk, "is_active_this_frame", False):
                    if trk.bbox not in person_boxes:
                        person_boxes.append(trk.bbox)

            # 2. DualSubtractor for stationary object extraction on 640x360 inference frame
            _, _, static_mask = self.subtractor.apply(infer_frame)
            if getattr(self, "_unattended_zones_mask", None) is None or self._unattended_zones_mask.shape != infer_frame.shape[:2]:
                self._unattended_zones_mask = self.zone_filter.get_unattended_zones_mask(shape=infer_frame.shape[:2])

            # Spatial masking strictly by unattended zones (walkway / corridor is 100% blacked out)
            static_mask_zoned = cv2.bitwise_and(static_mask, static_mask, mask=self._unattended_zones_mask)

            # Masking area person pada subtractor:
            # Paint solid black (0) over person bounding boxes with expanded +15px padding
            # to eliminate person clothing, torso, shadow, and feet from static mask
            fh, fw = infer_frame.shape[:2]
            for (px, py, pw, ph) in person_boxes:
                px1 = max(0, px - 15)
                py1 = max(0, py - 15)
                px2 = min(fw, px + pw + 15)
                py2 = min(fh, py + ph + 15)
                cv2.rectangle(static_mask_zoned, (px1, py1), (px2, py2), 0, -1)
                cv2.rectangle(static_mask, (px1, py1), (px2, py2), 0, -1)

            # Universal contour merging: MORPH_CLOSE (11x11) + light dilation to bridge split straps, shadows, fragmented blobs
            static_merged = cv2.morphologyEx(static_mask_zoned, cv2.MORPH_CLOSE, self._kernel_close_large)
            static_merged = cv2.dilate(static_merged, self._kernel_dilate, iterations=1)

            static_detections = self.zone_filter.filter_contours(static_merged, min_area_default=250)

            formatted_detections = []

            # 2a. Stationary bags extracted by DualSubtractor across all active ROI zones
            for cnt, (bx, by, bw, bh), (bcx, bcy), bzone in static_detections:
                # 3. Filter batas aspek rasio & ketinggian tas:
                # Ambang tinggi maksimal: h <= 200 px, area minimal: w * h >= 250 px
                if bh > 200 or (bw * bh) < 250:
                    continue

                # 2. Filter overlap / carried bag suppression:
                # Discard if centroid is inside person box or IoF > 0.60
                if self._is_bag_overlapping_person((bx, by, bw, bh), (bcx, bcy), person_boxes, iof_threshold=0.60):
                    continue

                # Ghost Artifact Elimination (uncovered flat floor verification)
                crop = infer_frame[by:by+bh, bx:bx+bw]
                crop_var = 0.0
                crop_edges = 0
                edge_density = 0.0
                if crop.size > 0:
                    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    crop_var = float(np.var(crop_gray))
                    crop_edges = int(np.count_nonzero(cv2.Canny(crop_gray, 40, 120)))
                    edge_density = crop_edges / float(max(1, bw * bh))
                    # Flat bare floor revealed upon object removal has near-zero variance and edges.
                    # Discard bare floor ghost blob only if ALL criteria confirm flat bare floor
                    if crop_var < 40.0 and crop_edges < 20 and edge_density < 0.015:
                        continue  # Discard bare floor ghost blob

                # Anti-Ghost Tile Line Filter for Recently Retrieved / Taken Bags
                is_near_retrieved = False
                for r_info in self._recently_retrieved_bags.values():
                    rcx, rcy = r_info["centroid"]
                    r_anch = r_info["anchor_centroid"]
                    r_box = r_info["bbox"]
                    d_r = float(np.hypot(bcx - rcx, bcy - rcy))
                    d_ra = float(np.hypot(bcx - r_anch[0], bcy - r_anch[1]))
                    iou_r = compute_bbox_iou((bx, by, bw, bh), r_box)
                    if min(d_r, d_ra) < 75.0 or iou_r > 0.15:
                        is_near_retrieved = True
                        break

                # Also check active stationary bags where a person is nearby / interacting (< 80px)
                is_near_active_bag_with_person = False
                for trk in self.tracker.objects.values():
                    if trk.class_label in ("tas", "backpack", "handbag", "suitcase") and trk.is_stationary:
                        d_c = float(np.hypot(bcx - trk.centroid[0], bcy - trk.centroid[1]))
                        d_a = float(np.hypot(bcx - trk.anchor_centroid[0], bcy - trk.anchor_centroid[1]))
                        iou_b = compute_bbox_iou((bx, by, bw, bh), getattr(trk, "anchor_bbox", trk.bbox))
                        if min(d_c, d_a) < 75.0 or iou_b > 0.15:
                            if (now - getattr(trk, "last_person_near_time", 0.0)) <= 2.5 or (now - getattr(trk, "last_occluded_time", 0.0)) <= 2.5:
                                is_near_active_bag_with_person = True
                                break

                # YOLO Verification Gate for DualSubtractor Blobs:
                # 1. Current frame YOLO confirmation (confidence > 0.15)
                has_current_yolo = False
                for y_box, y_cent, _, y_conf, y_cid, _ in yolo_results:
                    if y_cid in (24, 26, 28) and y_conf > 0.15:
                        d_yb = float(np.hypot(bcx - y_cent[0], bcy - y_cent[1]))
                        iou_yb = compute_bbox_iou((bx, by, bw, bh), y_box)
                        if d_yb < 75.0 or iou_yb > 0.15:
                            has_current_yolo = True
                            break

                # 2. Active track confirmed by YOLO within <= 5.0 seconds
                has_recent_yolo_track = False
                for trk in self.tracker.objects.values():
                    if trk.class_label in ("tas", "backpack", "handbag", "suitcase"):
                        d_tc = float(np.hypot(bcx - trk.centroid[0], bcy - trk.centroid[1]))
                        d_ta = float(np.hypot(bcx - trk.anchor_centroid[0], bcy - trk.anchor_centroid[1]))
                        iou_trk = compute_bbox_iou((bx, by, bw, bh), getattr(trk, "anchor_bbox", trk.bbox))
                        if min(d_tc, d_ta) < 75.0 or iou_trk > 0.15:
                            if (now - getattr(trk, "last_yolo_seen_time", 0.0)) <= 5.0:
                                has_recent_yolo_track = True
                                break

                # Safeguard 2: Small contour verification (>= 250 px, < 400 px)
                # Must pass strict edge density/variance or require confirmed YOLO to prevent tile reflection false alarms
                is_small_contour = (bw * bh) < 400
                if is_small_contour and not (has_current_yolo or has_recent_yolo_track):
                    if crop_var < 55.0 or edge_density < 0.020:
                        logger.debug(
                            f"[{self.camera_id}] Discarded small low-texture contour at ({bcx}, {bcy}) "
                            f"(var={crop_var:.1f}, edge_density={edge_density:.4f})."
                        )
                        continue

                if (is_near_retrieved or is_near_active_bag_with_person) and not has_current_yolo:
                    logger.debug(
                        f"[{self.camera_id}] Discarded ghost tile blob at ({bcx}, {bcy}) "
                        f"(retrieved/interaction location without YOLO bag confirmation > 0.15)."
                    )
                    continue  # DISCARD ghost tile blob!

                # DualSubtractor Gate: Forbid bare floor/tile from creating or sustaining a bag
                # without current YOLO bag detection OR active track seen by YOLO within 5.0s
                if not has_current_yolo and not has_recent_yolo_track:
                    logger.debug(
                        f"[{self.camera_id}] Discarded unconfirmed DualSubtractor blob at ({bcx}, {bcy}) "
                        f"(No YOLO bag on current frame and no active track confirmed by YOLO within 5.0s)."
                    )
                    continue

                area = float(bw * bh)
                formatted_detections.append(
                    ((bx, by, bw, bh), (bcx, bcy), bzone, area, "tas", 20.0, 0.95, "subtractor")
                )

            # 2b. Add YOLO detections (Persons and valid non-carried bags)
            for bbox, centroid, ref_point, conf, cid, cname in yolo_results:
                is_bag = cid in (24, 26, 28)

                if cid == 0:  # person
                    # Step tolerance margin: allow dynamic step tolerance capped strictly at max 10.0 px
                    matched_zone, edge_dist = self.zone_filter.find_zone_and_distance(
                        ref_point, margin_px=10.0
                    )
                    area = float(bbox[2] * bbox[3])
                    if matched_zone is not None:
                        formatted_detections.append(
                            (bbox, centroid, matched_zone, area, "person", edge_dist, float(conf), "yolo")
                        )
                    else:
                        # Also track person near any monitored zone (within 150 px) for universal owner proximity
                        dist_to_nearest = self.zone_filter.get_distance_to_nearest_zone(ref_point)
                        if dist_to_nearest >= -150.0:
                            formatted_detections.append(
                                (bbox, centroid, "outside_zone", area, "person", dist_to_nearest, float(conf), "yolo")
                            )
                elif is_bag:
                    # 3. Filter batas aspek rasio & dimensi tas: h <= 200 px, area >= 250 px
                    if bbox[3] > 200 or (bbox[2] * bbox[3]) < 250:
                        continue

                    # 2. Filter overlap / carried bag suppression
                    if self._is_bag_overlapping_person(bbox, centroid, person_boxes, iof_threshold=0.60):
                        continue

                    # Bags detected by YOLO (margin 8.0 px for overhead angle tolerance)
                    matched_zone, edge_dist = self.zone_filter.find_zone_and_distance(
                        ref_point, margin_px=8.0
                    )
                    if matched_zone is not None:
                        # Zone Role Separation: Only allow unattended bag monitoring in unattended zones!
                        z_cfg = self.config.get("zones", {}).get(matched_zone, {})
                        if not z_cfg and matched_zone.replace("__", "_") in self.config.get("zones", {}):
                            z_cfg = self.config.get("zones", {})[matched_zone.replace("__", "_")]
                        is_unattended = z_cfg.get("detect_unattended", None)
                        if is_unattended is None:
                            is_unattended = ("transit" in matched_zone.lower()) and ("koridor" not in matched_zone.lower())
                        if not is_unattended:
                            continue

                        area = float(bbox[2] * bbox[3])
                        # Deduplicate with DualSubtractor detection (within 50 px) in the same zone
                        is_dup = False
                        for idx_d, det_item in enumerate(formatted_detections):
                            if det_item[4] == "tas" and det_item[2] == matched_zone:
                                d_c = det_item[1]
                                if ((d_c[0] - centroid[0]) ** 2 + (d_c[1] - centroid[1]) ** 2) ** 0.5 < 50.0:
                                    # Refine with YOLO bounding box and confidence
                                    formatted_detections[idx_d] = (
                                        bbox, centroid, matched_zone, area, "tas", edge_dist, float(conf), "yolo"
                                    )
                                    is_dup = True
                                    break
                        if not is_dup:
                            formatted_detections.append(
                                (bbox, centroid, matched_zone, area, "tas", edge_dist, float(conf), "yolo")
                            )

            # 3. Tracking & Dwell Classification
            active_tracks, purged_tracks = self.tracker.update(formatted_detections, timestamp=now)
            self._associate_bag_owners(active_tracks, infer_frame, raw_clean_frame)

            # Record retrieved / departing stationary bags for ghost tile suppression
            for purged in purged_tracks:
                is_purged_bag = getattr(purged, "class_label", "") in ("tas", "backpack", "handbag", "suitcase")
                if getattr(purged, "is_retrieved", False) or (is_purged_bag and getattr(purged, "is_stationary", False)):
                    self._recently_retrieved_bags[purged.track_id] = {
                        "centroid": purged.centroid,
                        "anchor_centroid": purged.anchor_centroid,
                        "bbox": getattr(purged, "anchor_bbox", purged.bbox),
                        "zone_id": purged.zone_id,
                        "time": now,
                    }

            zones_cfg = self.config.get("zones", {})

            # Dynamic Burst Capture Triggers (3.0s burst at interval=1 on bag interaction in sterile zone)
            trigger_burst = False
            current_stationary_bag_ids = set()
            for trk in active_tracks:
                is_bag = getattr(trk, "class_label", "") in ("tas", "backpack", "handbag", "suitcase")
                if is_bag:
                    z_info = zones_cfg.get(trk.zone_id, {})
                    if not z_info and trk.zone_id.replace("__", "_") in zones_cfg:
                        z_info = zones_cfg[trk.zone_id.replace("__", "_")]
                    is_unattended_z = z_info.get("detect_unattended", None)
                    if is_unattended_z is None:
                        is_unattended_z = ("transit" in trk.zone_id.lower()) and ("koridor" not in trk.zone_id.lower())

                    if is_unattended_z:
                        # Condition 1: New bag in sterile zone
                        if trk.frame_count <= 2:
                            trigger_burst = True

                        # Condition 2: Bag newly stationary
                        if getattr(trk, "is_stationary", False):
                            current_stationary_bag_ids.add(trk.track_id)
                            if trk.track_id not in self._prev_stationary_bag_ids:
                                trigger_burst = True

            # Condition 3: Previously stationary bag starts moving (>50px for >=30 frames) or is picked up/removed by person
            for prev_sid in self._prev_stationary_bag_ids:
                if prev_sid not in current_stationary_bag_ids:
                    if prev_sid in self.tracker.objects:
                        trk = self.tracker.objects[prev_sid]
                        # Only trigger burst if genuine movement was confirmed!
                        if getattr(trk, "moved_confirmation_frames", 0) >= 30 or (now - getattr(trk, "last_moved_time", 0.0)) <= 3.0:
                            trigger_burst = True
                    else:
                        # Bag removed from active tracker: only trigger burst if contact with person occurred!
                        purged_match = next((p for p in purged_tracks if p.track_id == prev_sid), None)
                        if purged_match is not None:
                            was_attended = getattr(purged_match, "is_attended", False)
                            was_occluded = getattr(purged_match, "is_occluded", False)
                            had_owner = getattr(purged_match, "last_owner_info", None) is not None
                            was_retrieved = getattr(purged_match, "is_retrieved", False)
                            if was_attended or was_occluded or had_owner or was_retrieved:
                                trigger_burst = True

            self._prev_stationary_bag_ids = current_stationary_bag_ids

            if trigger_burst:
                self._face_burst_until_time = max(self._face_burst_until_time, now + 3.0)
                logger.info(f"[{self.camera_id}] Dynamic Face Burst activated for 3.0s (interval=1) due to bag interaction.")

            # 2a. Periodic 5-second terminal dwell time logging for stationary bags
            if (now - self._last_dwell_log_time) >= 5.0:
                for track in active_tracks:
                    is_bag = getattr(track, "class_label", "") in ("tas", "backpack", "handbag", "suitcase")
                    if is_bag and getattr(track, "is_stationary", False):
                        z_info = zones_cfg.get(track.zone_id, {})
                        if not z_info and track.zone_id.replace("__", "_") in zones_cfg:
                            z_info = zones_cfg[track.zone_id.replace("__", "_")]
                        is_unattended_z = z_info.get("detect_unattended", None)
                        if is_unattended_z is None:
                            is_unattended_z = ("transit" in track.zone_id.lower()) and ("koridor" not in track.zone_id.lower())
                        if not is_unattended_z:
                            continue
                        dwell_max = float(z_info.get("dwell_threshold_sec", z_info.get("dwell_time_threshold", z_info.get("unattended_threshold", 3600.0))))
                        max_str = f"{max(1, int(round(dwell_max / 60.0)))}m"
                        cur_str = f"{track.dwell_duration / 60.0:.1f}m"
                        if getattr(track, "is_occluded", False):
                            status_str = "OCCLUDED (PAUSED)"
                        elif getattr(track, "is_attended", False):
                            status_str = "ATTENDED"
                        else:
                            status_str = "UNATTENDED"
                        log_msg = f"[TRACKER] ID: {track.track_id} | Dwell: {cur_str} / {max_str} | Status: {status_str}"
                        print(log_msg)
                        logger.info(log_msg)
                self._last_dwell_log_time = now

            # 3. Handle Violation Triggers & Exit Zone Auto-Reset
            pending_telegram_alerts = []
            for track in active_tracks:
                # 1. PERSON LOITERING DISABLED: Person never triggers violations or dwell alarms
                if getattr(track, "class_label", "") == "person":
                    if track.is_triggered:
                        track.is_triggered = False
                        track.alert_sent = False
                        track.dwell_duration = 0.0
                        eid = self.active_db_events.pop(track.track_id, track.db_event_id)
                        if eid is not None:
                            self._resolve_event_async(eid)
                    continue

                # 2. EXIT ZONE / NON-UNATTENDED ZONE RESET:
                # If track is not in a valid unattended monitoring zone (e.g. corridor or outside), immediately disarm and clear
                zone_info = zones_cfg.get(track.zone_id, {})
                if not zone_info and track.zone_id.replace("__", "_") in zones_cfg:
                    zone_info = zones_cfg[track.zone_id.replace("__", "_")]

                is_unattended_zone = zone_info.get("detect_unattended", None)
                if is_unattended_zone is None:
                    is_unattended_zone = ("transit" in track.zone_id.lower()) and ("koridor" not in track.zone_id.lower())

                if not is_unattended_zone or not track.zone_id or "unassigned" in track.zone_id or "outside" in track.zone_id:
                    if track.is_triggered:
                        track.is_triggered = False
                        track.alert_sent = False
                        track.dwell_duration = 0.0
                        eid = self.active_db_events.pop(track.track_id, track.db_event_id)
                        if eid is not None:
                            z_name = zone_info.get("name", track.zone_id)
                            o_name = track.last_owner_info.get("name") if getattr(track, "last_owner_info", None) else None
                            self._resolve_event_async(eid, zone_name=z_name, track_id=track.track_id, dwell_duration=track.dwell_duration, owner_name=o_name)
                    continue

                track.dwell_threshold = float(zone_info.get("dwell_threshold_sec", zone_info.get("dwell_time_threshold", zone_info.get("unattended_threshold", 3600.0))))

                # 3. OWNER PROXIMITY CHECK: If owner is nearby (is_attended = True), disarm trigger
                if getattr(track, "is_attended", False):
                    if track.is_triggered:
                        track.is_triggered = False
                        track.alert_sent = False
                        track.warning_alerted = False
                        track.pre_alarm_alerted = False
                        track.is_warning = False
                        track.is_pre_alarm = False
                        eid = self.active_db_events.pop(track.track_id, track.db_event_id)
                        if eid is not None:
                            z_name = zone_info.get("name", track.zone_id)
                            o_name = track.last_owner_info.get("name") if getattr(track, "last_owner_info", None) else None
                            self._resolve_event_async(eid, zone_name=z_name, track_id=track.track_id, dwell_duration=track.dwell_duration, owner_name=o_name)
                    continue

                # 4. VIOLATION ONLY FOR UNATTENDED OBJECTS:
                if track.is_stationary:
                    dwell_thresh = float(zone_info.get("dwell_threshold_sec", zone_info.get("dwell_time_threshold", zone_info.get("unattended_threshold", 3600.0))))

                    # Multi-Stage Alert Escalation:
                    # Stage 1 (Warning 50% Dwell)
                    if track.dwell_duration >= (dwell_thresh * 0.50) and not getattr(track, "warning_alerted", False):
                        track.warning_alerted = True
                        track.is_warning = True
                        logger.warning(
                            f"[{self.camera_id}] STAGE 1 WARNING: {track.class_label.upper()} #{track.track_id} "
                            f"mencapai 50% dwell ({track.dwell_duration / 60.0:.1f}m/{dwell_thresh / 60.0:.1f}m) di '{zone_info.get('name', track.zone_id)}'"
                        )
                        try:
                            self.telegram_notifier.dispatch_composite_alert(
                                camera_id=self.camera_id,
                                zone_id=track.zone_id,
                                track=track,
                                stage="WARNING",
                                frame=raw_clean_frame,
                                zone_name=zone_info.get("name", track.zone_id),
                            )
                        except Exception as e:
                            logger.warning(f"[Telegram] Error sending warning composite alert: {e}")

                    # Stage 2 (Pre-Alarm 85% Dwell) Local Chime & Telegram Notification
                    if track.dwell_duration >= (dwell_thresh * 0.85) and not getattr(track, "pre_alarm_alerted", False):
                        self.audio_worker.request_pre_alarm_chime()
                        track.pre_alarm_alerted = True
                        track.is_pre_alarm = True
                        logger.warning(
                            f"[{self.camera_id}] STAGE 2 PRE-ALARM: {track.class_label.upper()} #{track.track_id} "
                            f"mencapai 85% dwell ({track.dwell_duration / 60.0:.1f}m/{dwell_thresh / 60.0:.1f}m) di '{zone_info.get('name', track.zone_id)}'"
                        )
                        try:
                            self.telegram_notifier.dispatch_composite_alert(
                                camera_id=self.camera_id,
                                zone_id=track.zone_id,
                                track=track,
                                stage="PRE_ALARM",
                                frame=raw_clean_frame,
                                zone_name=zone_info.get("name", track.zone_id),
                            )
                        except Exception as e:
                            logger.warning(f"[Telegram] Error sending pre-alarm composite alert: {e}")

                    if track.dwell_duration >= dwell_thresh and not track.is_triggered:
                        track.is_triggered = True
                        self.audio_worker.request_alarm()
                        trigger_iso = datetime.datetime.now().isoformat()
                        start_iso = datetime.datetime.fromtimestamp(track.stationary_start).isoformat()
                        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Dedicated abandoned bag snapshot naming
                        bag_filename = f"ALERT_ABANDONED_BAG_cam01_{timestamp_str}.jpg"
                        storage_path = Path(__file__).resolve().parent / "storage" / bag_filename
                        reports_path = Path(__file__).resolve().parent / "reports" / bag_filename
                        
                        zone_name = zone_info.get("name", track.zone_id)
                        dwell_minutes = track.dwell_duration / 60.0
                        alert_msg = f"[CLEAR AREA VIOLATION] Objek terlarang terdeteksi di area steril ({zone_name}), durasi: {dwell_minutes:.1f} menit"
                        print(f"\n{alert_msg}\n")
                        logger.warning(alert_msg)

                        # Camera workspace snapshot path
                        filename = f"{timestamp_str}_{track.zone_id}_{track.track_id}.jpg"
                        filepath = self.snapshots_dir / filename

                        # Extract owner metadata for breach registration and Telegram album
                        owner_info = getattr(track, "last_owner_info", None)
                        owner_name = owner_info.get("name") if owner_info else "Tidak Teridentifikasi"
                        owner_conf = float(owner_info.get("confidence", 0.0)) if owner_info else 0.0
                        face_crop = owner_info.get("face_crop") if owner_info else None

                        face_filepath_str = None
                        if face_crop is not None and face_crop.size > 0:
                            face_filename = f"{timestamp_str}_{track.zone_id}_{track.track_id}_face.jpg"
                            face_filepath = self.snapshots_dir / face_filename
                            self.io_executor.submit(cv2.imwrite, str(face_filepath), face_crop)
                            face_filepath_str = str(face_filepath)

                        # Immediate synchronous DB event registration
                        try:
                            event_type = "CLEAR_AREA_VIOLATION"
                            event_id = log_event(
                                camera_id=self.camera_id,
                                zone_id=track.zone_id,
                                track_id=track.track_id,
                                event_type=event_type,
                                dwell_duration=track.dwell_duration,
                                start_time=start_iso,
                                trigger_time=trigger_iso,
                                snapshot_path=str(filepath),
                                owner_name=owner_name,
                                owner_confidence=owner_conf,
                                face_snapshot_path=face_filepath_str,
                            )
                            track.db_event_id = event_id
                            self.active_db_events[track.track_id] = event_id
                            logger.warning(
                                f"[{self.camera_id}] Pelanggaran Clear Area terdaftar di '{zone_name}' "
                                f"oleh Track ID {track.track_id} (Dwell: {dwell_minutes:.1f}m, Pemilik: {owner_name}). Event ID: {event_id}"
                            )

                            # Asynchronously export 15-20s incident video clip from Ring Buffer
                            clip_filename = f"INCIDENT_{self.camera_id}_{timestamp_str}_{track.track_id}.mp4"
                            clip_storage_path = Path(__file__).resolve().parent / "storage" / "clips" / clip_filename

                            def _export_incident_clip_task(eid: int, out_path: Path, cname: str):
                                try:
                                    success = self.ring_buffer.export_clip(out_path)
                                    if success and eid:
                                        update_event_clip(eid, cname)
                                        logger.info(f"[{self.camera_id}] Event ID {eid} linked with video clip: {cname}")
                                except Exception as err:
                                    logger.error(f"[{self.camera_id}] Failed exporting incident video clip: {err}")

                            self.io_executor.submit(_export_incident_clip_task, event_id, clip_storage_path, clip_filename)
                        except Exception as e:
                            logger.error(f"[{self.camera_id}] DB event log failed: {e}")

                        # Stage alert for Telegram notification and snapshot saving after VisualHUD renders display_frame
                        if not getattr(track, "alert_sent", False):
                            track.alert_sent = True
                            pending_telegram_alerts.append({
                                "track": track,
                                "zone_id": track.zone_id,
                                "zone_name": zone_name,
                                "dwell_duration": track.dwell_duration,
                                "timestamp_str": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "bbox": list(track.bbox),
                                "storage_path": storage_path,
                                "reports_path": reports_path,
                                "filepath": filepath,
                                "owner_name": owner_name,
                                "owner_face_crop": face_crop,
                            })
                    elif track.is_triggered:
                        # Continue requesting alarm pulses within cooldown
                        self.audio_worker.request_alarm()

            # 5. Handle Deregistered / Purged Objects (Resolution hook)
            for purged in purged_tracks:
                eid = self.active_db_events.pop(purged.track_id, purged.db_event_id)
                if purged.is_triggered and eid is not None:
                    z_name = zones_cfg.get(purged.zone_id, {}).get("name", purged.zone_id)
                    o_name = purged.last_owner_info.get("name") if getattr(purged, "last_owner_info", None) else None
                    self._resolve_event_async(eid, zone_name=z_name, track_id=purged.track_id, dwell_duration=purged.dwell_duration, owner_name=o_name)

            # 6. FPS Calculation
            self._frame_count += 1
            elapsed = now - self._fps_start_time
            if elapsed >= 1.0:
                self.current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = now

            # Universal Face Detection (OpenCV YuNet) and Recognition (OpenCV SFace)
            if self.face_detection_enabled and self.face_detector is not None:
                active_person_tracks = [
                    trk for trk in active_tracks
                    if getattr(trk, "class_label", "") == "person"
                ]
                has_persons = len(active_person_tracks) > 0 or len(person_boxes) > 0

                if has_persons:
                    self._last_person_seen_time = now
                else:
                    # Grace period: if no person seen for >= 1.5s, purge cached faces
                    if (now - self._last_person_seen_time) >= 1.5:
                        self._cached_faces = []
                        self._face_recog_cache.clear()
                        self._person_face_recog.clear()

                if has_persons:
                    try:
                        disp_h, disp_w = display_frame.shape[:2]
                        scale_disp_x = float(disp_w) / 640.0
                        scale_disp_y = float(disp_h) / 360.0

                        # 1. TRACK-ID BASED EVALUATION GATING (Eliminates FPS drop & CPU spikes)
                        # Re-evaluation criteria: Track not yet evaluated, OR (elapsed >= 1.0s AND frames >= 15)
                        tracks_needing_eval = []
                        for trk in active_person_tracks:
                            p_cache = self._person_face_recog.get(trk.track_id)
                            if p_cache is None:
                                tracks_needing_eval.append((trk, 999.0))
                            else:
                                elapsed_p = now - p_cache.get("last_eval_time", 0.0)
                                frames_p = self._face_frame_index - p_cache.get("last_eval_frame", 0)
                                if elapsed_p >= 1.0 and frames_p >= 15:
                                    tracks_needing_eval.append((trk, elapsed_p))

                        # Sort candidates: longest un-evaluated first, then by closest person (bbox area)
                        tracks_needing_eval.sort(
                            key=lambda item: (item[1], item[0].bbox[2] * item[0].bbox[3]),
                            reverse=True,
                        )

                        # Per-frame budget: evaluate at most 1 person track per frame (preserves 12-15+ FPS)
                        candidate_tracks_to_eval = [t for t, _ in tracks_needing_eval[:1]]

                        eval_head_rois = []
                        eval_track = None
                        if candidate_tracks_to_eval:
                            eval_track = candidate_tracks_to_eval[0]
                            px, py, pw, ph = eval_track.bbox
                            pad_x = int(pw * 0.15)
                            h_x1 = max(0, px - pad_x)
                            h_y1 = max(0, py - int(ph * 0.10))
                            h_x2 = min(640, px + pw + pad_x)
                            h_y2 = min(360, py + int(ph * 0.50))
                            disp_x1 = int(round(h_x1 * scale_disp_x))
                            disp_y1 = int(round(h_y1 * scale_disp_y))
                            disp_x2 = int(round(h_x2 * scale_disp_x))
                            disp_y2 = int(round(h_y2 * scale_disp_y))
                            if (disp_x2 - disp_x1) >= 20 and (disp_y2 - disp_y1) >= 20:
                                eval_head_rois.append((disp_x1, disp_y1, disp_x2, disp_y2))

                        # Execute YuNet detection only for the single budgeted head crop
                        if eval_head_rois and eval_track is not None:
                            detected_faces = self.face_detector.detect_faces(
                                frame=infer_frame,
                                display_frame=display_frame,
                                crop_roi=eval_head_rois,
                            )
                            if detected_faces:
                                best_item = max(detected_faces, key=lambda it: it[1])
                                bbox_f, f_det_score = best_item[0], best_item[1]
                                raw_f_640 = best_item[2] if len(best_item) >= 3 else None
                                raw_f_1080 = best_item[3] if len(best_item) >= 4 else None

                                # 2. FACE QUALITY GATE BERBASIS LANDMARK (Bypass SFace on side-profile/extreme yaw)
                                face_landmark_data = raw_f_1080 if raw_f_1080 is not None else raw_f_640
                                is_frontal, quality_reason = FaceRecognizer.is_frontal_face(
                                    face_landmark_data,
                                    min_eye_dist_ratio=0.22,
                                    min_symmetry_ratio=0.20,
                                    min_score=0.60,
                                )

                                prev_cache = self._person_face_recog.get(eval_track.track_id)
                                prev_known = prev_cache and prev_cache.get("is_known", False)

                                if not is_frontal:
                                    logger.debug(
                                        f"[{self.camera_id}] Track #{eval_track.track_id} face quality gate bypass: {quality_reason}"
                                    )
                                    if prev_cache:
                                        prev_cache["last_eval_time"] = now
                                        prev_cache["last_eval_frame"] = self._face_frame_index
                                    else:
                                        # First time seen but non-frontal: throttle retry without creating fake face
                                        self._person_face_recog[eval_track.track_id] = {
                                            "has_seen_face": False,
                                            "last_eval_time": now,
                                            "last_eval_frame": self._face_frame_index,
                                        }
                                else:
                                    # Frontal face: execute SFace feature extraction & identity matching
                                    best_n = "Unknown"
                                    best_s = 0.0
                                    face_lbl = "Unknown"
                                    if self.face_recognition_enabled and self.face_recognizer is not None:
                                        if raw_f_1080 is not None and display_frame is not None:
                                            best_n, best_s, face_lbl = self.face_recognizer.recognize(
                                                frame=display_frame,
                                                face_data=raw_f_1080,
                                                min_size=24,
                                                check_frontal=False,
                                            )
                                        else:
                                            face_input = raw_f_640 if raw_f_640 is not None else bbox_f
                                            best_n, best_s, face_lbl = self.face_recognizer.recognize(
                                                frame=infer_frame,
                                                face_data=face_input,
                                                min_size=10,
                                                check_frontal=False,
                                            )

                                    if best_n != "Unknown" and face_lbl != "Unknown":
                                        self._person_face_recog[eval_track.track_id] = {
                                            "label": face_lbl,
                                            "name": best_n,
                                            "score": best_s,
                                            "last_eval_time": now,
                                            "last_eval_frame": self._face_frame_index,
                                            "face_bbox_640": bbox_f,
                                            "face_score": f_det_score,
                                            "is_known": True,
                                            "has_seen_face": True,
                                        }
                                    else:
                                        # Unknown / non-match: retain previous known identity if established
                                        if prev_known:
                                            prev_cache["last_eval_time"] = now
                                            prev_cache["last_eval_frame"] = self._face_frame_index
                                            prev_cache["face_bbox_640"] = bbox_f
                                        else:
                                            self._person_face_recog[eval_track.track_id] = {
                                                "label": "Unknown",
                                                "name": "Unknown",
                                                "score": 0.0,
                                                "last_eval_time": now,
                                                "last_eval_frame": self._face_frame_index,
                                                "face_bbox_640": bbox_f,
                                                "face_score": f_det_score,
                                                "is_known": False,
                                                "has_seen_face": True,
                                            }
                            else:
                                # Head crop yielded no face (e.g. back of head/hair): throttle retry without creating fake face
                                if eval_track.track_id in self._person_face_recog:
                                    self._person_face_recog[eval_track.track_id]["last_eval_time"] = now
                                    self._person_face_recog[eval_track.track_id]["last_eval_frame"] = self._face_frame_index
                                else:
                                    self._person_face_recog[eval_track.track_id] = {
                                        "has_seen_face": False,
                                        "last_eval_time": now,
                                        "last_eval_frame": self._face_frame_index,
                                    }

                        # Synthesize recognized_faces for VisualHUD and telemetry ONLY for tracks with genuine face detections
                        recognized_faces = []
                        active_track_ids = {t.track_id for t in active_person_tracks}

                        for trk in active_person_tracks:
                            p_info = self._person_face_recog.get(trk.track_id)
                            if p_info is not None and p_info.get("has_seen_face", False):
                                px, py, pw, ph = trk.bbox
                                f_box = p_info.get("face_bbox_640")
                                if not f_box or f_box[2] <= 0 or f_box[3] <= 0 or abs(f_box[0] - px) > (pw * 1.5):
                                    f_box = (px + pw // 4, py, pw // 2, ph // 3)
                                f_lbl = p_info.get("label", "Unknown")
                                f_name = p_info.get("name", "Unknown")
                                f_score = p_info.get("score", 0.0)
                                f_det_s = p_info.get("face_score", 0.80)
                                recognized_faces.append((f_box, f_det_s, f_lbl, f_name, f_score))

                        # Purge stale tracks no longer in active_person_tracks
                        if active_track_ids:
                            self._person_face_recog = {
                                tid: info for tid, info in self._person_face_recog.items()
                                if tid in active_track_ids
                            }

                        self._cached_faces = recognized_faces
                    except Exception as e:
                        logger.error(f"[{self.camera_id}] Face detection/recognition error: {e}")
                else:
                    self._cached_faces = []

                self._face_frame_index += 1

            # Refresh bag-to-owner association with latest face detection results
            self._associate_bag_owners(active_tracks, infer_frame, raw_clean_frame)

            # 7. Render Decoupled Visual Overlay onto Full HD display_frame
            display_frame = VisualHUD.render(
                canvas=display_frame,
                zones=self.roi_zones,
                tracked_objects=active_tracks,
                camera_id=self.camera_id,
                fps=self.current_fps,
                is_connected=self.capture.is_connected,
                faces=self._cached_faces,
                zone_base_resolution=self.roi_base_resolution,
            )

            # Process pending alerts with synchronized VisualHUD annotated display_frame
            if pending_telegram_alerts:
                annotated_snapshot = display_frame.copy()
                for alert in pending_telegram_alerts:
                    # Save snapshot evidence with annotated HUD
                    self.io_executor.submit(cv2.imwrite, str(alert["storage_path"]), annotated_snapshot)
                    self.io_executor.submit(cv2.imwrite, str(alert["reports_path"]), annotated_snapshot)
                    self.io_executor.submit(cv2.imwrite, str(alert["filepath"]), annotated_snapshot)

                    # Non-blocking Telegram photo alert dispatch (Single dispatch & anti-spam guard)
                    try:
                        self.telegram_notifier.dispatch_composite_alert(
                            camera_id=self.camera_id,
                            zone_id=alert["zone_id"],
                            track=alert["track"],
                            stage="BREACH",
                            frame=raw_clean_frame,
                            zone_name=alert["zone_name"],
                            timestamp_str=alert["timestamp_str"],
                        )
                        self.telegram_notifier.dispatch_alert(
                            camera_id=self.camera_id,
                            zone_id=alert["zone_id"],
                            track_id=alert["track"].track_id,
                            dwell_duration=alert["dwell_duration"],
                            timestamp_str=alert["timestamp_str"],
                            frame=raw_clean_frame,
                            overview_frame=annotated_snapshot,
                            bbox=alert["bbox"],
                            zone_name=alert["zone_name"],
                            zones=self.roi_zones,
                            owner_name=alert.get("owner_name"),
                            owner_face_crop=alert.get("owner_face_crop"),
                        )
                    except Exception as e:
                        logger.warning(f"[Telegram] Credentials not configured or dispatch error ({e}), skipping alert.")

            with self._display_lock:
                self._latest_display_frame = display_frame

            # 8. Push rendered frame and telemetry to MultiCameraBuffer for Web Dashboard
            rendering_bags = [
                obj for obj in active_tracks
                if getattr(obj, "class_label", "") in ("tas", "backpack", "handbag", "suitcase")
                and (not hasattr(obj, "should_render") or obj.should_render)
            ]
            violations_count = sum(
                1 for obj in rendering_bags
                if getattr(obj, "is_triggered", False)
                or (getattr(obj, "dwell_duration", 0.0) >= getattr(obj, "dwell_threshold", 3600.0) and getattr(obj, "is_stationary", False))
            )
            rendering_tracks = [
                obj for obj in active_tracks
                if not hasattr(obj, "should_render") or obj.should_render
            ]
            face_telemetry = []
            for f in (self._cached_faces or []):
                if isinstance(f, dict):
                    face_telemetry.append({
                        "name": f.get("name", "Unknown"),
                        "confidence": float(f.get("confidence", 0.0)),
                        "zone": f.get("zone", self.camera_id),
                    })
                elif isinstance(f, (list, tuple)) and len(f) >= 3:
                    # Format: (bbox, score, face_label, [best_n, best_s])
                    _, f_score, f_label = f[0], f[1], f[2]
                    f_name = f[3] if len(f) >= 4 and f[3] else (str(f_label).split(" (")[0] if isinstance(f_label, str) and " (" in f_label else str(f_label))
                    f_conf = f[4] if len(f) >= 5 and f[4] > 0 else (float(f_score) if isinstance(f_score, (int, float)) else 0.0)
                    is_recog = bool(f_name and f_name.lower() != "unknown" and str(f_label) != "Unknown")

                    face_telemetry.append({
                        "name": f_name if is_recog else "Unknown",
                        "confidence": float(f_conf) if is_recog else 0.0,
                        "zone": self.camera_id,
                    })
                elif isinstance(f, (list, tuple)) and len(f) == 2:
                    _, f_score_or_label = f[0], f[1]
                    name = str(f_score_or_label) if isinstance(f_score_or_label, str) else "Unknown"
                    conf = float(f_score_or_label) if isinstance(f_score_or_label, (int, float)) else 0.0
                    is_recog = bool(name and name.lower() != "unknown")
                    face_telemetry.append({
                        "name": name if is_recog else "Unknown",
                        "confidence": conf if is_recog else 0.0,
                        "zone": self.camera_id,
                    })
            telemetry = {
                "fps": float(self.current_fps),
                "online": bool(self.capture.is_connected),
                "is_connected": bool(self.capture.is_connected),
                "rtsp_status": "Connected" if self.capture.is_connected else "Reconnecting",
                "violations": violations_count,
                "clear_area_count": len(rendering_bags),
                "active_tracks": len(rendering_tracks),
                "identified_faces": face_telemetry,
            }
            try:
                MultiCameraBuffer.get_instance().update_frame(
                    camera_id=self.camera_id,
                    frame=display_frame,
                    telemetry=telemetry,
                    camera_name=self.config.get("name", self.camera_id),
                )
            except Exception as e:
                logger.error(f"[{self.camera_id}] Buffer update error: {e}")

    def get_display_frame(self) -> Optional[np.ndarray]:
        """Fetch latest rendered display frame safely."""
        with self._display_lock:
            return None if self._latest_display_frame is None else self._latest_display_frame.copy()

    def stop(self) -> None:
        """Signal pipeline shutdown and release all camera resources."""
        self._stopped.set()
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.5)
            except KeyboardInterrupt:
                pass
        self.capture.stop()
        try:
            self.io_executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.telegram_notifier.stop()
        except KeyboardInterrupt:
            pass


def discover_cameras(base_dir: Path) -> List[Path]:
    """Scan cameras directory and locate camera workspaces."""
    cam_root = base_dir / "cameras"
    if not cam_root.exists():
        return []
    return [p for p in cam_root.iterdir() if p.is_dir() and (p / "config.json").exists()]


def main() -> None:
    """Bootstrap multi-camera pipelines, FastAPI web dashboard, and event loop."""
    parser = argparse.ArgumentParser(description="Smart CCTV 2.0 - Core Orchestrator")
    parser.add_argument("--headless", action="store_true", help="Force headless mode without cv2 GUI display windows")
    parser.add_argument("--gui", action="store_true", help="Force GUI desktop display windows (cv2.imshow)")
    parser.add_argument("--port", type=int, default=None, help="Web dashboard port override (default: 8000)")
    parser.add_argument("--host", type=str, default=None, help="Web dashboard host override (default: 127.0.0.1)")
    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("       Smart CCTV 2.0 - Core Orchestrator         ")
    logger.info("==================================================")

    # Initialize SQLite database
    init_db()
    logger.info("Database tables and indexes verified.")

    # Execute auto-purge retention maintenance
    cleanup_old_records(retention_days=30)

    # Start background Disk Guard (6-hour retention & low-disk emergency failsafe)
    DiskGuardWorker.reset_instance()
    disk_guard = DiskGuardWorker(
        retention_interval_sec=21600.0,
        check_disk_interval_sec=300.0,
        retention_days=30,
        min_free_gb=5.0,
        max_usage_percent=90.0,
    )
    disk_guard.start()

    workspace_dir = Path(__file__).resolve().parent
    camera_dirs = discover_cameras(workspace_dir)

    if not camera_dirs:
        logger.warning("No camera workspaces discovered in cameras/. Exiting.")
        disk_guard.stop()
        sys.exit(0)

    logger.info(f"Discovered {len(camera_dirs)} camera workspace(s): {[c.name for c in camera_dirs]}")

    # Determine Web Dashboard & GUI Configuration
    dashboard_enabled = True
    dashboard_host = args.host or os.getenv("WEB_DASHBOARD_HOST", "127.0.0.1")
    dashboard_port = args.port or int(os.getenv("WEB_DASHBOARD_PORT", "8000"))

    # Read config.json defaults first (headless by default: enable_gui = false)
    cfg_enable_gui = False
    first_cfg_path = camera_dirs[0] / "config.json"
    if first_cfg_path.exists():
        try:
            first_cfg = load_camera_config(first_cfg_path)
            web_cfg = first_cfg.get("web_dashboard", {})
            if isinstance(web_cfg, dict):
                if "enabled" in web_cfg:
                    dashboard_enabled = bool(web_cfg["enabled"])
                if args.host is None and "host" in web_cfg:
                    dashboard_host = str(web_cfg["host"])
                if args.port is None and "port" in web_cfg:
                    dashboard_port = int(web_cfg["port"])
                if "enable_gui" in web_cfg:
                    cfg_enable_gui = bool(web_cfg["enable_gui"])
        except Exception as e:
            logger.debug(f"Note loading web_dashboard config: {e}")

    # Prioritize:
    # 1. CLI flag --headless -> False
    # 2. CLI flag --gui -> True
    # 3. config.json "enable_gui" value (default: False)
    if args.headless:
        enable_gui = False
    elif args.gui:
        enable_gui = True
    else:
        enable_gui = cfg_enable_gui

    # Instantiate and start all camera pipelines
    pipelines: List[CameraPipeline] = []
    for c_dir in camera_dirs:
        try:
            pipe = CameraPipeline(camera_dir=c_dir)
            pipe.start()
            pipelines.append(pipe)
            logger.info(f"Pipeline started for camera: {pipe.camera_id}")
        except Exception as e:
            logger.error(f"Failed to start pipeline for {c_dir.name}: {e}")

    # Start FastAPI Web Dashboard in background daemon thread
    if dashboard_enabled:
        DashboardServer.start(host=dashboard_host, port=dashboard_port)
        logger.info(f"Web Dashboard live at: http://{dashboard_host}:{dashboard_port}")

    # Run GUI loop or Headless loop
    if enable_gui:
        for pipe in pipelines:
            win_name = f"Smart CCTV - {pipe.camera_id}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        logger.info("Running live display with GUI windows. Press 'q' or 'ESC' on display window to terminate.")
        try:
            while True:
                for pipe in pipelines:
                    frame = pipe.get_display_frame()
                    if frame is not None:
                        win_name = f"Smart CCTV - {pipe.camera_id}"
                        try:
                            rect = cv2.getWindowImageRect(win_name)
                            if rect and rect[2] > 50 and rect[3] > 50:
                                win_w, win_h = rect[2], rect[3]
                                if frame.shape[1] != win_w or frame.shape[0] != win_h:
                                    frame = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
                            cv2.imshow(win_name, frame)
                        except cv2.error:
                            break

                key = cv2.waitKey(15) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("Termination key received. Shutting down camera pipelines...")
                    break
        except KeyboardInterrupt:
            logger.info("Shutdown requested via KeyboardInterrupt.")
        finally:
            for pipe in pipelines:
                pipe.stop()
            DashboardServer.stop()
            disk_guard.stop()
            cv2.destroyAllWindows()
            logger.info("All camera pipelines terminated gracefully.")
    else:
        logger.info("Running in HEADLESS mode (cv2.imshow GUI windows disabled).")
        logger.info("Web Dashboard active for remote monitoring. Press Ctrl+C to terminate.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Shutdown requested via KeyboardInterrupt.")
        finally:
            for pipe in pipelines:
                pipe.stop()
            DashboardServer.stop()
            disk_guard.stop()
            logger.info("All camera pipelines terminated gracefully.")


if __name__ == "__main__":
    main()
