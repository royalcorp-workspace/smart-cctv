"""Smart CCTV 2.0 - Multi-Camera Pipeline Orchestrator."""

import concurrent.futures
import datetime
import json
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from engine.config_loader import load_camera_config
from engine.dual_subtractor import DualSubtractor
from engine.logger import logger
from engine.retention import cleanup_old_records
from engine.rtsp_stream import ThreadedCapture
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

        # Target resolution
        self.target_resolution: Optional[Tuple[int, int]] = None
        cfg_res = self.config.get("target_resolution")
        if cfg_res and len(cfg_res) == 2:
            self.target_resolution = (int(cfg_res[0]), int(cfg_res[1]))

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

        self.subtractor = DualSubtractor(
            slow_history=5000,
            slow_learning_rate=0.0001,
            fast_history=40,
            fast_learning_rate=0.05,
        )
        self.zone_filter = ZoneFilter(
            zones=self.roi_zones,
            zone_configs=self.config.get("zones", {}),
            frame_shape=self.target_resolution[::-1] if self.target_resolution else (720, 1280),
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
        if self.roi_path.exists():
            with open(self.roi_path, "r", encoding="utf-8") as f:
                self.roi_zones = json.load(f)

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
            # Check 1: Bag centroid inside person bounding box
            if px <= bcx <= (px + pw) and py <= bcy <= (py + ph):
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
                    target_w = self.target_resolution[0] if self.target_resolution else 640
                    target_h = self.target_resolution[1] if self.target_resolution else 480
                    placeholder = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    VisualHUD.render(
                        canvas=placeholder,
                        zones=self.roi_zones,
                        tracked_objects=[],
                        camera_id=self.camera_id,
                        fps=self.current_fps,
                        is_connected=False,
                    )
                    with self._display_lock:
                        self._latest_display_frame = placeholder
                    time.sleep(0.05)
                else:
                    # Transient inter-frame delay on low FPS cameras
                    time.sleep(0.01)
                continue

            last_valid_frame_time = now

            # Target resolution normalization
            if self.target_resolution is not None:
                tw, th = self.target_resolution
                if frame.shape[1] != tw or frame.shape[0] != th:
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)

            # Keep clean raw copy for snapshot storage
            raw_clean_frame = frame.copy()

            # 1. YOLO11 Nano Object Detection (Person & Optional Bag proposal refinement)
            yolo_results = self.detector.detect(frame)
            person_boxes = [bbox for bbox, _, _, _, cid, _ in yolo_results if cid == 0]

            # 2. DualSubtractor for stationary object extraction across all active ROI zones
            _, _, static_mask = self.subtractor.apply(frame)
            if getattr(self, "_all_zones_mask", None) is None or self._all_zones_mask.shape != frame.shape[:2]:
                self._all_zones_mask = self.zone_filter.get_all_zones_mask(shape=frame.shape[:2])

            # Spatial masking by union of all active ROI zones
            static_mask_zoned = cv2.bitwise_and(static_mask, static_mask, mask=self._all_zones_mask)

            # Masking area person pada subtractor:
            # Paint solid black (0) over person bounding boxes with +10px padding
            # to eliminate person clothing, torso, and feet from static mask
            fh, fw = frame.shape[:2]
            for (px, py, pw, ph) in person_boxes:
                px1 = max(0, px - 10)
                py1 = max(0, py - 10)
                px2 = min(fw, px + pw + 10)
                py2 = min(fh, py + ph + 10)
                cv2.rectangle(static_mask_zoned, (px1, py1), (px2, py2), 0, -1)

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
                crop = frame[by:by+bh, bx:bx+bw]
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
                        dwell_max = int(z_info.get("dwell_threshold_sec", 60))
                        status_str = "ATTENDED" if getattr(track, "is_attended", False) else "UNATTENDED"
                        log_msg = f"[TRACKER] ID: {track.track_id} | Dwell: {track.dwell_duration:.1f}s / {dwell_max}s | Status: {status_str}"
                        print(log_msg)
                        logger.info(log_msg)
                self._last_dwell_log_time = now

            # 3. Handle Violation Triggers & Exit Zone Auto-Reset
            for track in active_tracks:
                # 1. PERSON LOITERING DISABLED: Person never triggers violations or dwell alarms
                if getattr(track, "class_label", "") == "person":
                    if track.is_triggered:
                        track.is_triggered = False
                        track.dwell_duration = 0.0
                        eid = self.active_db_events.pop(track.track_id, track.db_event_id)
                        if eid is not None:
                            self._resolve_event_async(eid)
                    continue

                # 2. EXIT ZONE RESET: If track is not in a valid monitored zone, immediately disarm and clear
                zone_info = zones_cfg.get(track.zone_id, {})
                if not zone_info and track.zone_id.replace("__", "_") in zones_cfg:
                    zone_info = zones_cfg[track.zone_id.replace("__", "_")]

                if not zone_info or not track.zone_id or "unassigned" in track.zone_id or "outside" in track.zone_id:
                    if track.is_triggered:
                        track.is_triggered = False
                        track.dwell_duration = 0.0
                        eid = self.active_db_events.pop(track.track_id, track.db_event_id)
                        if eid is not None:
                            self._resolve_event_async(eid)
                    continue

                track.dwell_threshold = float(zone_info.get("dwell_threshold_sec", 60.0))

                # 3. OWNER PROXIMITY CHECK: If owner is nearby (is_attended = True), disarm trigger
                if getattr(track, "is_attended", False):
                    if track.is_triggered:
                        track.is_triggered = False
                        eid = self.active_db_events.pop(track.track_id, track.db_event_id)
                        if eid is not None:
                            self._resolve_event_async(eid)
                    continue

                # 4. VIOLATION ONLY FOR UNATTENDED BAGS:
                if track.is_stationary:
                    dwell_thresh = float(zone_info.get("dwell_threshold_sec", 60.0))

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
                        
                        alert_msg = f"[ALERT] Unattended Bag detected in {track.zone_id} for over {int(dwell_thresh)} seconds!"
                        print(f"\n{alert_msg}\n")
                        logger.warning(alert_msg)

                        # Camera workspace snapshot path
                        filename = f"{timestamp_str}_{track.zone_id}_{track.track_id}.jpg"
                        filepath = self.snapshots_dir / filename
                        self.io_executor.submit(cv2.imwrite, str(filepath), raw_clean_frame)

                        # Immediate synchronous DB event registration
                        try:
                            event_type = "UNATTENDED_BAG"
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
                                f"[{self.camera_id}] Unattended bag violation registered in '{track.zone_id}' "
                                f"by Track ID {track.track_id} (Dwell: {track.dwell_duration:.1f}s). Event ID: {event_id}"
                            )
                        except Exception as e:
                            logger.error(f"[{self.camera_id}] DB event log failed: {e}")

                        # Non-blocking Telegram photo alert dispatch (Zero FPS drop)
                        formatted_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.telegram_notifier.dispatch_alert(
                            camera_id=self.camera_id,
                            zone_id=track.zone_id,
                            track_id=track.track_id,
                            dwell_duration=track.dwell_duration,
                            timestamp_str=formatted_time,
                            frame=raw_clean_frame,
                            bbox=track.bbox,
                        )
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

            # 7. Render Decoupled Visual Overlay
            display_frame = VisualHUD.render(
                canvas=frame,
                zones=self.roi_zones,
                tracked_objects=active_tracks,
                camera_id=self.camera_id,
                fps=self.current_fps,
                is_connected=self.capture.is_connected,
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
