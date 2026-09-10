"""Threaded frame grabber with single-frame buffer anti-latency architecture and TCP transport."""

import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from engine.logger import logger

# Force OpenCV FFmpeg RTSP to use TCP transport with zero-delay socket buffering
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000"


class ThreadedCapture:
    """Non-blocking threaded video capture with auto-reconnect and zero-latency atomic frame slot."""

    def __init__(
        self,
        source: Union[int, str],
        reconnect_interval_sec: float = 3.0,
        frame_timeout_sec: float = 8.0,
        loop_playback: bool = True,
    ) -> None:
        self.source: Union[int, str] = source
        self.reconnect_interval_sec: float = reconnect_interval_sec
        self.frame_timeout_sec: float = frame_timeout_sec
        self.loop_playback: bool = loop_playback

        self._is_file: bool = False
        if isinstance(self.source, str):
            src_path = Path(self.source)
            if src_path.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov"):
                self._is_file = True

        self._lock: threading.Lock = threading.Lock()
        self._frame_cond: threading.Condition = threading.Condition(self._lock)
        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_available: bool = False
        self._last_valid_frame: Optional[np.ndarray] = None
        self._last_frame_timestamp: float = 0.0
        self._stopped: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self.is_connected: bool = False

    def start(self) -> "ThreadedCapture":
        """Start the background frame consumer daemon."""
        self._stopped.clear()
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()
        return self

    def _open_capture(self) -> bool:
        """Initialize OpenCV VideoCapture with TCP transport and minimal buffer."""
        if self._cap is not None:
            self._cap.release()

        backend = cv2.CAP_FFMPEG if isinstance(self.source, str) and self.source.startswith("rtsp://") else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(self.source, backend)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.is_connected = True
            logger.info("Video capture session established successfully via TCP transport.")
            return True

        self.is_connected = False
        logger.warning(f"Failed to open video capture source. Retrying in {self.reconnect_interval_sec}s...")
        return False

    def _capture_worker(self) -> None:
        """Continuous background capture loop with auto-reconnect and frame drop."""
        last_frame_time = time.time()
        while not self._stopped.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._open_capture():
                    time.sleep(self.reconnect_interval_sec)
                    continue
                last_frame_time = time.time()

            ret, frame = self._cap.read()
            now = time.time()
            if not ret or frame is None:
                if self._is_file and self.loop_playback:
                    # Rewind file to start for continuous test playback
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                
                # Check if frame drop timeout exceeded (tolerates 10 FPS camera ~100ms intervals)
                if now - last_frame_time >= self.frame_timeout_sec:
                    self.is_connected = False
                    if self._cap is not None:
                        self._cap.release()
                    logger.warning(
                        f"No video frames received for {self.frame_timeout_sec:.1f}s. "
                        f"Reconnecting in {self.reconnect_interval_sec}s..."
                    )
                    time.sleep(self.reconnect_interval_sec)
                else:
                    # Transient gap between frames at low native FPS
                    time.sleep(0.01)
                continue

            # Frame read succeeded
            last_frame_time = now
            self.is_connected = True
            self._last_valid_frame = frame
            self._last_frame_timestamp = now

            # Atomic single-frame overwrite slot (O(1) drop, zero lag accumulation)
            with self._frame_cond:
                self._latest_frame = frame
                self._new_frame_available = True
                self._frame_cond.notify_all()

            # Pace file playback to native FPS
            if self._is_file:
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                delay = 1.0 / fps if fps > 0 else 0.03
                time.sleep(delay)

        if self._cap is not None:
            self._cap.release()
            self.is_connected = False

    def read(self, timeout: float = 0.15) -> Tuple[bool, Optional[np.ndarray]]:
        """Fetch the latest frame from the atomic slot, waiting up to timeout seconds."""
        with self._frame_cond:
            if not self._new_frame_available and not self._stopped.is_set():
                self._frame_cond.wait(timeout=timeout)
            if self._new_frame_available and self._latest_frame is not None:
                frame = self._latest_frame
                self._new_frame_available = False
                return True, frame

            now = time.time()
            # Graceful read: if transient queue empty but last frame was within 1.5s and still connected
            if self.is_connected and self._last_valid_frame is not None and (now - self._last_frame_timestamp) < 1.5:
                return True, self._last_valid_frame.copy()
            return False, None

    def stop(self) -> None:
        """Signal thread shutdown and release resources."""
        self._stopped.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self.is_connected = False
