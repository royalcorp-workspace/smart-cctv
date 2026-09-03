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

# Force OpenCV FFmpeg RTSP to use TCP transport (prevents packet loss and frame drops)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


class ThreadedCapture:
    """Non-blocking threaded video capture with auto-reconnect and frame dropping."""

    def __init__(
        self,
        source: Union[int, str],
        reconnect_interval_sec: float = 2.0,
        frame_timeout_sec: float = 3.0,
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

        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
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

            # Anti-lag buffer: drop stale frame if queue is full
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass

            # Pace file playback to native FPS
            if self._is_file:
                fps = self._cap.get(cv2.CAP_PROP_FPS)
                delay = 1.0 / fps if fps > 0 else 0.03
                time.sleep(delay)

        if self._cap is not None:
            self._cap.release()
            self.is_connected = False

    def read(self, timeout: float = 0.15) -> Tuple[bool, Optional[np.ndarray]]:
        """Fetch the latest frame from the buffer, waiting up to timeout seconds."""
        try:
            frame = self._frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        """Signal thread shutdown and release resources."""
        self._stopped.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self.is_connected = False
