"""Smart CCTV 2.0 - Multi-Camera Pipeline Orchestrator."""

import concurrent.futures
import datetime
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from engine.config_loader import load_camera_config
from engine.dual_subtractor import DualSubtractor
from engine.logger import logger
from engine.retention import cleanup_old_records
from engine.rtsp_stream import ThreadedCapture
from engine.face_detector import YuNetFaceDetector
from engine.face_recognizer import FaceRecognizer
from engine.tracker import CentroidTracker, TrackedObject
from engine.yolo_detector import YOLOOpenVINODetector
from engine.zone_filter import ZoneFilter
from notification.local_alert import GlobalAudioWorker, VisualHUD
from notification.telegram_alert import TelegramNotifier
from storage.db import init_db, log_event, resolve_event


class CameraPipeline:
    """Encapsulates autonomous capture, processing, tracking, and alerting for a single camera workspace."""

    def __init__(self, camera_dir: Path) -> None:
        self.camera_dir: Path = camera_dir
        self.config_path: Path = camera_dir / "config.json"
        self.roi_path: Path = camera_dir / "roi_zones.json"
        self.snapshots_dir: Path = camera_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._load_configurations()

        # Target resolution (Display / Capture frame)
        self.target_resolution: Optional[Tuple[int, int]] = None
        cfg_res = self.config.get("target_resolution")
        if cfg_res and len(cfg_res) == 2:
            self.target_resolution = (int(cfg_res[0]), int(cfg_res[1]))

        # Decoupled AI inference resolution (fixed 640x480 for lightweight CPU performance)
        self.infer_resolution: Tuple[int, int] = (640, 480)

        # Initialize Subcomponents
        self.capture = ThreadedCapture(source=self.config.get("source", 0))
        
        # YOLO11 Nano with Intel OpenVINO CPU Acceleration
        detector_cfg = self.config.get("detector", {})
        conf_thresh = float(detector_cfg.get("confidence_threshold", 0.40))
        model_name = detector_cfg.get("model_name", "yolo11n")
        self.detector = YOLOOpenVINODetector(
            model_name=model_name,
            device="cpu",
            confidence_threshold=conf_thresh,
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
            max_distance_px=50.0,
            movement_threshold_px=15.0,
            anchor_radius_px=15.0,
            flicker_tolerance_sec=2.0,
            max_disappeared_sec=4.5,
            max_age_frames=45,
            ema_alpha=0.3,
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

        if self.face_detection_enabled:
            try:
                face_model = face_cfg.get("model_path", None)
                score_th = float(face_cfg.get("score_threshold", 0.48))
                nms_th = float(face_cfg.get("nms_threshold", 0.30))
                alternate_infer = bool(face_cfg.get("alternate_inference", True))
                self.face_detector = YuNetFaceDetector(
                    model_path=face_model,
                    score_threshold=score_th,
                    nms_threshold=nms_th,
                    input_size=self.infer_resolution,
                    auto_download=True,
                    max_missed_frames=6,
                    alternate_inference=alternate_infer,
                )
                logger.info(
                    f"[{self.camera_id}] YuNet Face Detector initialized "
                    f"(interval={self._face_detect_interval} frames, alternate={alternate_infer}, input_size={self.infer_resolution})."
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
                cos_th = float(face_rec_cfg.get("cosine_threshold", 0.50))
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

    def _resolve_event_async(self, event_id: int) -> None:
        """I/O worker task: mark event resolved in SQLite."""
        try:
            resolve_event(event_id=event_id)
            logger.info(f"[{self.camera_id}] Event ID {event_id} marked as resolved.")
        except Exception as e:
            logger.error(f"[{self.camera_id}] Async resolve failed for event {event_id}: {e}")

    @staticmethod
    def _is_bag_overlapping_person(
        bag_box: Tuple[int, int, int, int],
        bag_centroid: Tuple[int, int],
        person_boxes: List[Tuple[int, int, int, int]],
        iof_threshold: float = 0.3,
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
                            self._resolve_event_async(eid)

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
                    time.sleep(0.05)
                else:
                    # Transient inter-frame delay on low FPS cameras
                    time.sleep(0.01)
                continue

            last_valid_frame_time = now

            # Target resolution normalization (Display frame)
            if self.target_resolution is not None:
                tw, th = self.target_resolution
                if frame.shape[1] != tw or frame.shape[0] != th:
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

            # Decoupled resolution architecture:
            # 1. High-resolution clean copy for display and snapshots (Full HD 1080p)
            display_frame = frame.copy()
            raw_clean_frame = frame.copy()

            # 2. Scaled lightweight inference frame (640x480) for AI models (YOLO11n, MOG2, YuNet)
            infer_frame = cv2.resize(frame, self.infer_resolution, interpolation=cv2.INTER_LINEAR)

            # 1. YOLO11 Nano Object Detection on 640x480 inference frame with 2-frame stride (50% CPU savings)
            if (self._yolo_frame_index % 2) == 0:
                self._cached_yolo_results = self.detector.detect(infer_frame)
            self._yolo_frame_index += 1
            yolo_results = self._cached_yolo_results
            person_boxes = [bbox for bbox, _, _, _, cid, _ in yolo_results if cid == 0]
            # Also include any active person tracks to bridge single-frame detection drops
            for trk in self.tracker.objects.values():
                if trk.class_label == "person" and getattr(trk, "is_active_this_frame", False):
                    if trk.bbox not in person_boxes:
                        person_boxes.append(trk.bbox)

            # 2. DualSubtractor for stationary object extraction on 640x480 inference frame
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

            static_detections = self.zone_filter.filter_contours(static_merged, min_area_default=400)

            formatted_detections = []

            # 2a. Stationary bags extracted by DualSubtractor across all active ROI zones
            for cnt, (bx, by, bw, bh), (bcx, bcy), bzone in static_detections:
                # 3. Filter batas aspek rasio & ketinggian tas:
                # Ambang tinggi maksimal: h <= 120 px, area minimal: w * h >= 400 px
                if bh > 120 or (bw * bh) < 400:
                    continue

                # 2. Filter overlap / carried bag suppression:
                # Discard if centroid is inside person box or IoF > 0.3
                if self._is_bag_overlapping_person((bx, by, bw, bh), (bcx, bcy), person_boxes):
                    continue

                # Ghost Artifact Elimination (uncovered flat floor verification)
                crop = infer_frame[by:by+bh, bx:bx+bw]
                if crop.size > 0:
                    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    crop_var = float(np.var(crop_gray))
                    crop_edges = int(np.count_nonzero(cv2.Canny(crop_gray, 40, 120)))
                    edge_density = crop_edges / float(max(1, bw * bh))
                    # Flat bare floor revealed upon object removal has near-zero variance and edges.
                    # Real bags have rich textures, zippers, straps (variance > 80, edges > 40, density > 0.035)
                    if crop_var < 80.0 or crop_edges < 40 or edge_density < 0.035:
                        continue  # Discard bare floor ghost blob

                area = float(bw * bh)
                formatted_detections.append(
                    ((bx, by, bw, bh), (bcx, bcy), bzone, area, "tas", 20.0, 0.95)
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
                            (bbox, centroid, matched_zone, area, "person", edge_dist, float(conf))
                        )
                    else:
                        # Also track person near any monitored zone (within 150 px) for universal owner proximity
                        dist_to_nearest = self.zone_filter.get_distance_to_nearest_zone(ref_point)
                        if dist_to_nearest >= -150.0:
                            formatted_detections.append(
                                (bbox, centroid, "outside_zone", area, "person", dist_to_nearest, float(conf))
                            )
                elif is_bag:
                    # 3. Filter batas aspek rasio & dimensi tas
                    if bbox[3] > 120 or (bbox[2] * bbox[3]) < 400:
                        continue

                    # 2. Filter overlap / carried bag suppression
                    if self._is_bag_overlapping_person(bbox, centroid, person_boxes):
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
                                        bbox, centroid, matched_zone, area, "tas", edge_dist, float(conf)
                                    )
                                    is_dup = True
                                    break
                        if not is_dup:
                            formatted_detections.append(
                                (bbox, centroid, matched_zone, area, "tas", edge_dist, float(conf))
                            )

            # 3. Tracking & Dwell Classification
            active_tracks, purged_tracks = self.tracker.update(formatted_detections, timestamp=now)

            zones_cfg = self.config.get("zones", {})

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
                        status_str = "ATTENDED" if getattr(track, "is_attended", False) else "UNATTENDED"
                        log_msg = f"[TRACKER] ID: {track.track_id} | Dwell: {cur_str} / {max_str} | Status: {status_str}"
                        print(log_msg)
                        logger.info(log_msg)
                self._last_dwell_log_time = now

            # 3. Handle Violation Triggers & Exit Zone Auto-Reset
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
                            self._resolve_event_async(eid)
                    continue

                track.dwell_threshold = float(zone_info.get("dwell_threshold_sec", zone_info.get("dwell_time_threshold", zone_info.get("unattended_threshold", 3600.0))))

                # 3. OWNER PROXIMITY CHECK: If owner is nearby (is_attended = True), disarm trigger
                if getattr(track, "is_attended", False):
                    if track.is_triggered:
                        track.is_triggered = False
                        track.alert_sent = False
                        eid = self.active_db_events.pop(track.track_id, track.db_event_id)
                        if eid is not None:
                            self._resolve_event_async(eid)
                    continue

                # 4. VIOLATION ONLY FOR UNATTENDED BAGS:
                if track.is_stationary:
                    dwell_thresh = float(zone_info.get("dwell_threshold_sec", zone_info.get("dwell_time_threshold", zone_info.get("unattended_threshold", 3600.0))))

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
                        self.io_executor.submit(cv2.imwrite, str(storage_path), raw_clean_frame)
                        self.io_executor.submit(cv2.imwrite, str(reports_path), raw_clean_frame)
                        
                        zone_name = zone_info.get("name", track.zone_id)
                        dwell_minutes = track.dwell_duration / 60.0
                        alert_msg = f"[CLEAR AREA VIOLATION] Objek terlarang terdeteksi di area steril ({zone_name}), durasi: {dwell_minutes:.1f} menit"
                        print(f"\n{alert_msg}\n")
                        logger.warning(alert_msg)

                        # Camera workspace snapshot path
                        filename = f"{timestamp_str}_{track.zone_id}_{track.track_id}.jpg"
                        filepath = self.snapshots_dir / filename
                        self.io_executor.submit(cv2.imwrite, str(filepath), raw_clean_frame)

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
                            )
                            track.db_event_id = event_id
                            self.active_db_events[track.track_id] = event_id
                            logger.warning(
                                f"[{self.camera_id}] Pelanggaran Clear Area terdaftar di '{zone_name}' "
                                f"oleh Track ID {track.track_id} (Dwell: {dwell_minutes:.1f}m). Event ID: {event_id}"
                            )
                        except Exception as e:
                            logger.error(f"[{self.camera_id}] DB event log failed: {e}")

                        # Non-blocking Telegram photo alert dispatch (Single dispatch & anti-spam guard)
                        if not getattr(track, "alert_sent", False):
                            track.alert_sent = True
                            formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            zone_name = zone_info.get("name", track.zone_id)
                            try:
                                self.telegram_notifier.dispatch_alert(
                                    camera_id=self.camera_id,
                                    zone_id=track.zone_id,
                                    track_id=track.track_id,
                                    dwell_duration=track.dwell_duration,
                                    timestamp_str=formatted_time,
                                    frame=raw_clean_frame,
                                    bbox=track.bbox,
                                    zone_name=zone_name,
                                    zones=self.roi_zones,
                                )
                            except Exception as e:
                                logger.warning(f"[Telegram] Credentials not configured or dispatch error ({e}), skipping alert.")
                    elif track.is_triggered:
                        # Continue requesting alarm pulses within cooldown
                        self.audio_worker.request_alarm()

            # 5. Handle Deregistered / Purged Objects (Resolution hook)
            for purged in purged_tracks:
                eid = self.active_db_events.pop(purged.track_id, purged.db_event_id)
                if purged.is_triggered and eid is not None:
                    self._resolve_event_async(eid)

            # 6. FPS Calculation
            self._frame_count += 1
            elapsed = now - self._fps_start_time
            if elapsed >= 1.0:
                self.current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = now

            # Universal Face Detection (OpenCV YuNet) and Recognition (OpenCV SFace)
            if self.face_detection_enabled and self.face_detector is not None:
                if (self._face_frame_index % self._face_detect_interval) == 0:
                    try:
                        detected_faces = self.face_detector.detect_faces(
                            frame=infer_frame,
                            display_frame=display_frame,
                            crop_roi=self.face_roi_crop,
                        )
                        recognized_faces = []
                        active_fids = set()
                        now_recog = time.time()

                        for item in detected_faces:
                            bbox = item[0]
                            score = item[1]
                            raw_face_640 = item[2] if len(item) >= 3 else None
                            raw_face_1080 = item[3] if len(item) >= 4 else None
                            face_id = item[4] if len(item) >= 5 else None

                            if face_id is not None:
                                active_fids.add(face_id)

                            face_label = "Unknown"
                            if self.face_recognition_enabled and self.face_recognizer is not None:
                                # Determine 1080p face dimensions for minimum size gating (min 28x28 px)
                                if raw_face_1080 is not None:
                                    face_w_1080 = float(raw_face_1080[2])
                                    face_h_1080 = float(raw_face_1080[3])
                                else:
                                    face_w_1080 = float(bbox[2]) * (1920.0 / 640.0)
                                    face_h_1080 = float(bbox[3]) * (1080.0 / 480.0)

                                if face_w_1080 < 28.0 or face_h_1080 < 28.0:
                                    # Distant micro-face (< 28x28 px): Immediately classify as Unknown without SFace inference
                                    face_label = "Unknown"
                                    if face_id is not None:
                                        self._face_recog_cache[face_id] = {
                                            "label": "Unknown",
                                            "time": now_recog,
                                            "pos": (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2),
                                        }
                                else:
                                    # Prominent face (>= 28x28 px): Check stationary cache
                                    cached = self._face_recog_cache.get(face_id, {})
                                    cached_label = cached.get("label")
                                    last_time = cached.get("time", 0.0)
                                    last_pos = cached.get("pos", (0, 0))

                                    cur_pos = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                                    dist_moved = ((cur_pos[0] - last_pos[0]) ** 2 + (cur_pos[1] - last_pos[1]) ** 2) ** 0.5
                                    elapsed = now_recog - last_time

                                    # Reuse cache if stationary (< 15 px movement) within validity window (2.5s for known, 1.0s for unknown)
                                    cache_ttl = 2.5 if (cached_label and cached_label != "Unknown") else 1.0
                                    if cached_label and dist_moved < 15.0 and elapsed < cache_ttl:
                                        face_label = cached_label
                                    else:
                                        # Recognize using 1080p display_frame with raw_face_1080 if available
                                        if raw_face_1080 is not None and display_frame is not None:
                                            _, _, face_label = self.face_recognizer.recognize(
                                                frame=display_frame, face_data=raw_face_1080, min_size=28
                                            )
                                        else:
                                            face_input = raw_face_640 if raw_face_640 is not None else bbox
                                            _, _, face_label = self.face_recognizer.recognize(
                                                frame=infer_frame, face_data=face_input, min_size=10
                                            )

                                        if face_id is not None:
                                            self._face_recog_cache[face_id] = {
                                                "label": face_label,
                                                "time": now_recog,
                                                "pos": cur_pos,
                                            }

                            recognized_faces.append((bbox, score, face_label))

                        # Purge stale face recognition caches
                        if active_fids:
                            self._face_recog_cache = {
                                fid: v for fid, v in self._face_recog_cache.items() if fid in active_fids
                            }

                        self._cached_faces = recognized_faces
                    except Exception as e:
                        logger.error(f"[{self.camera_id}] Face detection/recognition error: {e}")
                self._face_frame_index += 1

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

            with self._display_lock:
                self._latest_display_frame = display_frame

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
    """Bootstrap multi-camera pipelines and run primary window event loop."""
    logger.info("==================================================")
    logger.info("       Smart CCTV 2.0 - Core Orchestrator         ")
    logger.info("==================================================")

    # Initialize SQLite database
    init_db()
    logger.info("Database tables and indexes verified.")

    # Execute auto-purge retention maintenance
    cleanup_old_records(retention_days=30)

    workspace_dir = Path(__file__).resolve().parent
    camera_dirs = discover_cameras(workspace_dir)

    if not camera_dirs:
        logger.warning("No camera workspaces discovered in cameras/. Exiting.")
        sys.exit(0)

    logger.info(f"Discovered {len(camera_dirs)} camera workspace(s): {[c.name for c in camera_dirs]}")

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

    # Initialize display windows with WINDOW_NORMAL for dynamic resizing
    for pipe in pipelines:
        win_name = f"Smart CCTV - {pipe.camera_id}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    logger.info("Running live display. Press 'q' or 'ESC' on display window to terminate.")

    try:
        while True:
            for pipe in pipelines:
                frame = pipe.get_display_frame()
                if frame is not None:
                    win_name = f"Smart CCTV - {pipe.camera_id}"
                    
                    try:
                        # Auto-resize frame to fit maximized or resized window without gray bars
                        rect = cv2.getWindowImageRect(win_name)
                        if rect and rect[2] > 50 and rect[3] > 50:
                            win_w, win_h = rect[2], rect[3]
                            if frame.shape[1] != win_w or frame.shape[0] != win_h:
                                frame = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)

                        cv2.imshow(win_name, frame)
                    except cv2.error:
                        # Window closed by user
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
        cv2.destroyAllWindows()
        logger.info("All camera pipelines terminated gracefully.")


if __name__ == "__main__":
    main()
