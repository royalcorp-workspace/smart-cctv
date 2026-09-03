"""Threaded frame grabber with single-frame buffer anti-latency architecture."""

import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from engine.logger import logger


class ThreadedCapture:
    """Non-blocking threaded video capture with auto-reconnect and frame dropping."""

    def __init__(
        self,
        source: Union[int, str],
        reconnect_interval_sec: float = 2.0,
        loop_playback: bool = True,
    ) -> None:
        self.source: Union[int, str] = source
        self.reconnect_interval_sec: float = reconnect_interval_sec
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
        """Initialize OpenCV VideoCapture."""
        if self._cap is not None:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.source)
        if self._cap.isOpened():
            self.is_connected = True
            logger.info("Video capture session established successfully.")
            return True

        self.is_connected = False
        logger.warning(f"Failed to open video capture source. Retrying in {self.reconnect_interval_sec}s...")
        return False

    def _capture_worker(self) -> None:
        """Continuous background capture loop with auto-reconnect and frame drop."""
        while not self._stopped.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._open_capture():
                    time.sleep(self.reconnect_interval_sec)
                    continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                if self._is_file and self.loop_playback:
                    # Rewind file to start for continuous test playback
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    self.is_connected = False
                    self._cap.release()
                    logger.warning(
                        f"Stream frame read failed or dropped. Reconnecting in {self.reconnect_interval_sec}s..."
                    )
                    time.sleep(self.reconnect_interval_sec)
                    continue

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

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Fetch the latest frame non-blockingly without accumulating latency."""
        try:
            frame = self._frame_queue.get_nowait()
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
