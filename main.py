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
from engine.zone_filter import ZoneFilter
from notification.local_alert import GlobalAudioWorker, VisualHUD
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
            movement_threshold_px=8.0,
            max_disappeared_sec=5.0,
        )
        self.audio_worker = GlobalAudioWorker()
        self.io_executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

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

            # 1. Dual Subtraction
            _, _, static_mask = self.subtractor.apply(frame)

            # 2. Zone Filtering
            raw_detections = self.zone_filter.filter_contours(static_mask)
            formatted_detections = [
                (det[1], det[2], det[3], cv2.contourArea(det[0]))
                for det in raw_detections
            ]

            # 3. Tracking & Dwell Classification
            active_tracks, purged_tracks = self.tracker.update(formatted_detections, timestamp=now)

            # 4. Handle Violation Triggers
            zones_cfg = self.config.get("zones", {})
            for track in active_tracks:
                if getattr(track, "is_attended", False):
                    # If previously triggered but owner returned nearby, disarm trigger
                    if track.is_triggered:
                        track.is_triggered = False
                        eid = self.active_db_events.pop(track.track_id, track.db_event_id)
                        if eid is not None:
                            self._resolve_event_async(eid)
                    continue

                if track.is_stationary:
                    zone_info = zones_cfg.get(track.zone_id, {})
                    if not zone_info and track.zone_id.replace("__", "_") in zones_cfg:
                        zone_info = zones_cfg[track.zone_id.replace("__", "_")]
                    
                    is_zone_2 = "zone_2" in track.zone_id
                    default_thresh = 60.0 if is_zone_2 else 15.0
                    dwell_thresh = float(zone_info.get("dwell_threshold_sec", default_thresh))

                    if track.dwell_duration >= dwell_thresh and not track.is_triggered:
                        track.is_triggered = True
                        self.audio_worker.request_alarm()
                        trigger_iso = datetime.datetime.now().isoformat()
                        start_iso = datetime.datetime.fromtimestamp(track.stationary_start).isoformat()
                        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # Dedicated abandoned bag snapshot naming
                        if is_zone_2:
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
                            event_type = "UNATTENDED_BAG" if is_zone_2 else "DWELL_VIOLATION"
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
                                f"[{self.camera_id}] Violation registered in '{track.zone_id}' "
                                f"by Track ID {track.track_id} (Dwell: {track.dwell_duration:.1f}s). Event ID: {event_id}"
                            )
                        except Exception as e:
                            logger.error(f"[{self.camera_id}] DB event log failed: {e}")
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
            self._thread.join(timeout=2.0)
        self.capture.stop()
        self.io_executor.shutdown(wait=True)


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
                    
                    # Auto-resize frame to fit maximized or resized window without gray bars
                    rect = cv2.getWindowImageRect(win_name)
                    if rect and rect[2] > 50 and rect[3] > 50:
                        win_w, win_h = rect[2], rect[3]
                        if frame.shape[1] != win_w or frame.shape[0] != win_h:
                            frame = cv2.resize(frame, (win_w, win_h), interpolation=cv2.INTER_LINEAR)

                    cv2.imshow(win_name, frame)

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
