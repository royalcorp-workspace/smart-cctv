"""Non-blocking Telegram Alert Notification Module for Smart CCTV.

Dispatches instant photo snapshots and structured alerts to Telegram bots and channels
using an asynchronous background worker queue (Zero FPS drop).
"""

import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import requests

from engine.logger import logger


class TelegramNotifier:
    """Thread-safe, non-blocking Telegram notification dispatcher."""

    _instance: Optional["TelegramNotifier"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, config_path: Optional[Path] = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "telegram.json"
        self.config_path: Path = config_path

        self.enabled: bool = False
        self.bot_token: str = ""
        self.request_timeout: int = 10
        self.max_retries: int = 2
        self.global_admins: List[str] = []
        self.camera_routing: Dict[str, List[str]] = {}

        self._load_config()

        # Background Queue & Daemon Worker
        self._queue: queue.Queue = queue.Queue(maxsize=30)
        self._stop_event: threading.Event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self.enabled and self.bot_token:
            self._start_worker()

    @classmethod
    def get_instance(cls, config_path: Optional[Path] = None) -> "TelegramNotifier":
        """Singleton accessor for shared notification pipeline."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config_path=config_path)
            return cls._instance

    def _load_config(self) -> None:
        """Parse configuration file or fall back to environment variables."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.enabled = bool(data.get("enabled", True))
                self.bot_token = str(data.get("bot_token", "")).strip()
                self.request_timeout = int(data.get("request_timeout_sec", 10))
                self.max_retries = int(data.get("max_retries", 2))
                recipients = data.get("recipients", {})
                self.global_admins = [str(x).strip() for x in recipients.get("global_admins", []) if str(x).strip()]
                self.camera_routing = {
                    cam: [str(x).strip() for x in chat_ids if str(x).strip()]
                    for cam, chat_ids in recipients.get("cameras", {}).items()
                }
            except Exception as e:
                logger.error(f"[TelegramNotifier] Failed to load config from {self.config_path}: {e}")

        # Environment variable overrides if present
        env_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if env_token:
            self.bot_token = env_token.strip()
            self.enabled = True

    def reload_config(self) -> None:
        """Reload configuration dynamically from disk."""
        self._load_config()
        if self.enabled and self.bot_token and (self._worker_thread is None or not self._worker_thread.is_alive()):
            self._start_worker()

    def _start_worker(self) -> None:
        """Launch background worker daemon thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="TelegramAlertWorker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("[TelegramNotifier] Background worker thread started successfully.")

    def get_recipients_for_camera(self, camera_id: str) -> List[str]:
        """Aggregate and deduplicate Chat IDs for a specific camera."""
        recipients_set: Set[str] = set(self.global_admins)
        cam_recipients = self.camera_routing.get(camera_id, [])
        recipients_set.update(cam_recipients)
        return list(recipients_set)

    def dispatch_alert(
        self,
        camera_id: str,
        zone_id: str,
        track_id: int,
        dwell_duration: float,
        timestamp_str: str,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """Enqueue alert job asynchronously (non-blocking, < 0.1ms).

        Returns True if enqueued, False if dropped or disabled.
        """
        if not self.enabled or not self.bot_token:
            return False

        recipients = self.get_recipients_for_camera(camera_id)
        if not recipients:
            logger.debug(f"[TelegramNotifier] No Chat IDs configured for camera '{camera_id}'. Alert skipped.")
            return False

        job = {
            "camera_id": camera_id,
            "zone_id": zone_id,
            "track_id": track_id,
            "dwell_duration": dwell_duration,
            "timestamp_str": timestamp_str,
            "frame": frame.copy(),
            "bbox": bbox,
            "recipients": recipients,
        }

        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            logger.warning("[TelegramNotifier] Alert queue is full! Dropping alert to prevent memory overflow.")
            return False

    def _worker_loop(self) -> None:
        """Daemon worker loop: consumes queue and sends HTTP requests via Telegram Bot API."""
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._send_job(job)
            except Exception as e:
                logger.error(f"[TelegramNotifier] Unhandled exception in worker: {e}")
            finally:
                self._queue.task_done()

    def _send_job(self, job: dict) -> None:
        """Render annotated snapshot and dispatch via sendPhoto to all recipients."""
        camera_id = job["camera_id"]
        zone_id = job["zone_id"]
        track_id = job["track_id"]
        dwell = job["dwell_duration"]
        timestamp_str = job["timestamp_str"]
        frame = job["frame"]
        bbox = job["bbox"]
        recipients = job["recipients"]

        # 1. Annotate frame with high-visibility alert indicator
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if bbox is not None:
            bx, by, bw, bh = bbox
            # Crisp red bounding box
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            # Alert header badge
            badge_text = f"[ALERT] BARANG TERTINGGAL (ID #{track_id})"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (bx, max(0, by - th - 6)), (bx + tw + 6, by), (0, 0, 255), -1)
            cv2.putText(
                annotated,
                badge_text,
                (bx + 3, by - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Bottom watermark banner
        watermark = f"SMART CCTV | CAM: {camera_id.upper()} | ZONA: {zone_id} | {timestamp_str}"
        cv2.putText(
            annotated,
            watermark,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # 2. Encode to high-quality JPEG in memory
        success, encoded_img = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            logger.error("[TelegramNotifier] Failed to encode snapshot image to JPEG.")
            return

        img_bytes = encoded_img.tobytes()

        # 3. Format structured caption (HTML mode)
        caption = (
            "⚠️ <b>[PERINGATAN CCTV] BARANG TERTINGGAL</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📹 <b>Kamera</b>: <code>{camera_id}</code>\n"
            f"📍 <b>Zona</b>: <code>{zone_id}</code>\n"
            f"🏷️ <b>Objek</b>: Tas / Koper (Track ID #{track_id})\n"
            f"⏱️ <b>Durasi Ditinggal</b>: <b>{int(dwell)} detik</b> (Ambang: 60s)\n"
            f"🕒 <b>Waktu Deteksi</b>: <code>{timestamp_str}</code>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "🚨 <i>Terdeteksi barang tanpa pendamping di area transit. Harap petugas sekuriti memeriksa lokasi fisik.</i>"
        )

        # 4. Dispatch sendPhoto per recipient
        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"

        for chat_id in recipients:
            self._send_photo_with_retry(url, chat_id, img_bytes, caption)

    def _send_photo_with_retry(self, url: str, chat_id: str, img_bytes: bytes, caption: str) -> bool:
        """Send photo with automatic retry mechanism and timeout protection."""
        for attempt in range(1, self.max_retries + 1):
            try:
                files = {"photo": ("unattended_bag.jpg", img_bytes, "image/jpeg")}
                data = {
                    "chat_id": chat_id,
                    "caption": caption,
                    "parse_mode": "HTML",
                }
                response = requests.post(url, data=data, files=files, timeout=self.request_timeout)
                if response.status_code == 200:
                    logger.info(f"[TelegramNotifier] Snapshot alert dispatched successfully to Chat ID: {chat_id}")
                    return True
                else:
                    logger.warning(
                        f"[TelegramNotifier] Telegram API returned HTTP {response.status_code} "
                        f"(Attempt {attempt}/{self.max_retries}): {response.text}"
                    )
            except requests.exceptions.Timeout:
                logger.warning(
                    f"[TelegramNotifier] Request timed out sending alert to Chat ID: {chat_id} "
                    f"(Attempt {attempt}/{self.max_retries})"
                )
            except Exception as e:
                logger.warning(
                    f"[TelegramNotifier] Network error sending alert to Chat ID: {chat_id} "
                    f"(Attempt {attempt}/{self.max_retries}): {e}"
                )

            if attempt < self.max_retries:
                time.sleep(1.0)

        logger.error(f"[TelegramNotifier] Failed to send Telegram alert to Chat ID: {chat_id} after {self.max_retries} attempts.")
        return False

    def stop(self) -> None:
        """Gracefully signal background worker shutdown."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            logger.info("[TelegramNotifier] Worker thread terminated gracefully.")
