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
from typing import Any, Dict, List, Optional, Set, Tuple

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
        self.config_path: Path = Path(config_path)

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

        # Scan cameras/*/config.json for camera-specific telegram configuration (only in default runtime mode)
        default_config = Path(__file__).resolve().parent.parent / "configs" / "telegram.json"
        if self.config_path == default_config:
            try:
                cameras_dir = Path(__file__).resolve().parent.parent / "cameras"
                if cameras_dir.is_dir():
                    for cam_dir in cameras_dir.iterdir():
                        if cam_dir.is_dir():
                            cam_cfg_path = cam_dir / "config.json"
                            if cam_cfg_path.exists():
                                with open(cam_cfg_path, "r", encoding="utf-8") as f:
                                    cam_data = json.load(f)
                                cam_tg = cam_data.get("telegram", {})
                                if isinstance(cam_tg, dict):
                                    if not self.bot_token and cam_tg.get("bot_token"):
                                        self.bot_token = str(cam_tg["bot_token"]).strip()
                                    cam_chat_id = str(cam_tg.get("chat_id", "")).strip()
                                    if cam_chat_id:
                                        cam_id = cam_data.get("camera_id", cam_dir.name)
                                        cur_list = self.camera_routing.setdefault(cam_id, [])
                                        if cam_chat_id not in cur_list:
                                            cur_list.append(cam_chat_id)
                                    if cam_tg.get("enabled"):
                                        self.enabled = True
            except Exception as e:
                logger.debug(f"[TelegramNotifier] Camera config scan note: {e}")

        # Environment variable overrides if present
        env_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if env_token:
            self.bot_token = env_token.strip()
            self.enabled = True

    def register_camera_telegram(self, camera_id: str, telegram_cfg: Dict[str, Any]) -> None:
        """Dynamically register or update camera-specific Telegram configuration."""
        if not isinstance(telegram_cfg, dict):
            return
        if not self.bot_token and telegram_cfg.get("bot_token"):
            self.bot_token = str(telegram_cfg["bot_token"]).strip()
        chat_id = str(telegram_cfg.get("chat_id", "")).strip()
        if chat_id:
            cam_list = self.camera_routing.setdefault(camera_id, [])
            if chat_id not in cam_list:
                cam_list.append(chat_id)
        if telegram_cfg.get("enabled"):
            self.enabled = True
        if self.enabled and self.bot_token and (self._worker_thread is None or not self._worker_thread.is_alive()):
            self._start_worker()

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

    @staticmethod
    def create_zoom_crop(
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        min_width: int = 480,
        padding_ratio: float = 0.35,
    ) -> Optional[np.ndarray]:
        """Extract contextual close-up zoom crop around object bbox with padding and cubic resize.

        - frame: 1080p frame
        - bbox: (x, y, w, h) in 640x480 space or 1080p space
        - padding_ratio: default 35% margin around the object
        - min_width: minimum width (default 480px) scaled with cv2.INTER_CUBIC
        """
        if bbox is None or frame is None or frame.size == 0:
            return None

        fh, fw = frame.shape[:2]
        bx, by, bw, bh = bbox

        # Auto-scale bbox if bbox is in 640x480 inference space and frame is not 640x480
        if (fw, fh) != (640, 480) and (bx + bw <= 640 and by + bh <= 480):
            scale_x = fw / 640.0
            scale_y = fh / 480.0
        else:
            scale_x = 1.0
            scale_y = 1.0

        sbx = int(round(bx * scale_x))
        sby = int(round(by * scale_y))
        sbw = int(round(bw * scale_x))
        sbh = int(round(bh * scale_y))

        # Dynamic contextual padding (35% default)
        pad_w = int(round(sbw * padding_ratio))
        pad_h = int(round(sbh * padding_ratio))

        # Safe clamping within image boundaries
        x1 = max(0, sbx - pad_w)
        y1 = max(0, sby - pad_h)
        x2 = min(fw, sbx + sbw + pad_w)
        y2 = min(fh, sby + sbh + pad_h)

        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return None

        crop = frame[y1:y2, x1:x2].copy()
        ch, cw = crop.shape[:2]
        if ch <= 0 or cw <= 0:
            return None

        # Draw a crisp bounding box on crop highlighting the target item
        rel_bx = sbx - x1
        rel_by = sby - y1
        cv2.rectangle(crop, (rel_bx, rel_by), (rel_bx + sbw, rel_by + sbh), (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Proportional resize if crop width is smaller than min_width (e.g. 480px)
        if cw < min_width:
            target_w = min_width
            target_h = max(1, int(round(ch * (target_w / float(cw)))))
            crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Stamp a clean compact badge "[ZOOM DETAIL]" in top-left
        badge_text = "[ZOOM DETAIL]"
        font_scale = 0.42
        font_thick = 1
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        cv2.rectangle(crop, (4, 4), (4 + tw + 8, 4 + th + 8), (0, 0, 0), -1)
        cv2.rectangle(crop, (4, 4), (4 + tw + 8, 4 + th + 8), (0, 0, 255), 1)
        cv2.putText(crop, badge_text, (8, 4 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thick, cv2.LINE_AA)

        return crop

    def dispatch_alert(
        self,
        camera_id: str,
        zone_id: str,
        track_id: int,
        dwell_duration: float,
        timestamp_str: str,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        zone_name: Optional[str] = None,
        overview_frame: Optional[np.ndarray] = None,
        zones: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Enqueue alert job asynchronously (non-blocking, < 0.1ms).

        Returns True if enqueued, False if dropped or disabled.
        """
        # Strict 60-minute threshold: alerts NEVER fire below 3600 seconds
        if dwell_duration < 3600.0:
            logger.debug(f"[TelegramNotifier] Dwell duration {dwell_duration:.1f}s < 3600.0s threshold. Alert skipped.")
            return False

        if not self.enabled or not self.bot_token or not self.bot_token.strip():
            logger.warning("[Telegram] Credentials not configured, skipping alert.")
            return False

        try:
            recipients = self.get_recipients_for_camera(camera_id)
            if not recipients:
                logger.warning("[Telegram] Credentials not configured, skipping alert.")
                return False

            job = {
                "camera_id": camera_id,
                "zone_id": zone_id,
                "zone_name": zone_name or zone_id,
                "track_id": track_id,
                "dwell_duration": dwell_duration,
                "timestamp_str": timestamp_str,
                "frame": frame.copy(),
                "overview_frame": overview_frame.copy() if overview_frame is not None else None,
                "bbox": bbox,
                "zones": zones,
                "recipients": recipients,
            }

            self._queue.put_nowait(job)
            return True
        except queue.Full:
            logger.warning("[TelegramNotifier] Alert queue is full! Dropping alert to prevent memory overflow.")
            return False
        except Exception as e:
            logger.warning(f"[Telegram] Credentials not configured or dispatch error ({e}), skipping alert.")
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
        """Render annotated overview + contextual zoom crop and dispatch via sendMediaGroup."""
        camera_id = job["camera_id"]
        zone_id = job["zone_id"]
        zone_name = job.get("zone_name", zone_id)
        track_id = job["track_id"]
        dwell = job["dwell_duration"]
        timestamp_str = job["timestamp_str"]
        clean_frame = job["frame"]
        overview_frame = job.get("overview_frame")
        bbox = job["bbox"]
        zones = job.get("zones")
        recipients = job["recipients"]

        # 1. Prepare Overview Image
        if overview_frame is not None and overview_frame.size > 0:
            annotated = overview_frame.copy()
        else:
            annotated = clean_frame.copy()

        h, w = annotated.shape[:2]
        scale_x = w / 640.0
        scale_y = h / 480.0
        scale_factor = min(scale_x, scale_y)

        # Draw zone polylines if zones provided and overview was not pre-rendered
        if zones and isinstance(zones, dict) and overview_frame is None:
            for zid, pts in zones.items():
                if not isinstance(pts, list) or zid.startswith("_") or zid == "base_resolution":
                    continue
                if len(pts) < 3:
                    continue
                max_zx = max(pt[0] for pt in pts if len(pt) >= 2) if pts else 0
                max_zy = max(pt[1] for pt in pts if len(pt) >= 2) if pts else 0
                z_scale_x = (w / 1920.0) if max_zx > 640 else (w / 640.0)
                z_scale_y = (h / 1080.0) if max_zy > 480 else (h / 480.0)
                s_pts = np.array([[int(round(pt[0] * z_scale_x)), int(round(pt[1] * z_scale_y))] for pt in pts], dtype=np.int32).reshape((-1, 1, 2))
                zone_color = (0, 0, 255) if (zid == zone_id or zid.replace("__", "_") == zone_id.replace("__", "_")) else (0, 255, 0)
                cv2.polylines(annotated, [s_pts], isClosed=True, color=zone_color, thickness=2, lineType=cv2.LINE_AA)

        if bbox is not None and overview_frame is None:
            bx, by, bw, bh = bbox
            if (w, h) != (640, 480):
                bx = int(round(bx * scale_x))
                by = int(round(by * scale_y))
                bw = int(round(bw * scale_x))
                bh = int(round(bh * scale_y))

            box_thickness = max(2, int(round(2.0 * scale_factor)))
            font_scale = 0.45 * scale_factor
            font_thick = max(1, int(round(1.2 * scale_factor)))
            pad_x = int(round(4.0 * scale_factor))
            pad_y = int(round(3.0 * scale_factor))

            # Crisp red bounding box
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (0, 0, 255), box_thickness, lineType=cv2.LINE_AA)
            
            # Alert header badge
            badge_text = f"[ALERT] PELANGGARAN CLEAR AREA (ID #{track_id})"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            badge_y1 = max(0, by - th - (pad_y * 2) - 2)
            badge_y2 = by
            badge_x2 = min(w, bx + tw + (pad_x * 2) + 2)

            cv2.rectangle(annotated, (bx, badge_y1), (badge_x2, badge_y2), (0, 0, 255), -1)
            cv2.putText(
                annotated,
                badge_text,
                (bx + pad_x, badge_y2 - pad_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thick,
                cv2.LINE_AA,
            )

        # Bottom watermark banner on overview
        display_zone = zone_name if zone_name else zone_id
        watermark = f"SMART CCTV | CAM: {camera_id.upper()} | ZONA: {display_zone} | {timestamp_str}"
        wm_font_scale = 0.40 * scale_factor
        wm_font_thick = max(1, int(round(1.0 * scale_factor)))
        wm_margin = int(round(10.0 * scale_factor))
        cv2.putText(
            annotated,
            watermark,
            (wm_margin, h - wm_margin),
            cv2.FONT_HERSHEY_SIMPLEX,
            wm_font_scale,
            (0, 255, 255),
            wm_font_thick,
            cv2.LINE_AA,
        )

        # 2. Extract Contextual Close-up Zoom Crop from clean frame (un-occluded, 35% padding)
        zoom_crop = self.create_zoom_crop(clean_frame, bbox, min_width=480, padding_ratio=0.35)

        # 3. Encode images to JPEG
        success_ov, enc_overview = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success_ov:
            logger.error("[TelegramNotifier] Failed to encode overview snapshot image to JPEG.")
            return

        overview_bytes = enc_overview.tobytes()

        zoom_bytes = None
        if zoom_crop is not None:
            success_zm, enc_zoom = cv2.imencode(".jpg", zoom_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if success_zm:
                zoom_bytes = enc_zoom.tobytes()

        # 4. Format structured caption (HTML mode)
        dwell_mins = dwell / 60.0
        caption = (
            "🚨 <b>PERINGATAN: PELANGGARAN CLEAR AREA</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📹 <b>Kamera</b>: <code>{camera_id}</code>\n"
            f"📍 <b>Zona</b>: <code>{display_zone}</code>\n"
            f"🏷️ <b>Objek</b>: Objek Terlarang / Halangan (Track ID #{track_id})\n"
            f"⏱️ <b>Durasi Pelanggaran</b>: <b>{dwell_mins:.1f} Menit</b> (Batas: 60 Menit)\n"
            f"🕒 <b>Waktu</b>: <code>{timestamp_str}</code>\n"
            "📷 <b>Bukti</b>: Foto Overview + Close-up Zoom terlampir.\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ <i>Terdeteksi barang di area steril / clear area yang tidak semestinya. Harap petugas keamanan segera mensterilkan lokasi fisik.</i>"
        )

        # 5. Dispatch dual-image album or single photo per recipient
        if not self.bot_token or not self.bot_token.strip():
            logger.warning("[Telegram] Credentials not configured, skipping alert.")
            return

        for chat_id in recipients:
            if zoom_bytes is not None:
                # Primary: sendMediaGroup (album in 1 bubble)
                sent = self._send_media_group_with_retry(chat_id, overview_bytes, zoom_bytes, caption)
                if not sent:
                    # Fallback: sequential sendPhoto
                    logger.info(f"[TelegramNotifier] sendMediaGroup failed for {chat_id}; falling back to sendPhoto.")
                    send_photo_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
                    self._send_photo_with_retry(send_photo_url, chat_id, overview_bytes, caption)
                    zoom_caption = f"🔍 <b>[DETAIL CROP]</b> Objek Terlarang di Area Steril (Track ID #{track_id})"
                    self._send_photo_with_retry(send_photo_url, chat_id, zoom_bytes, zoom_caption)
            else:
                send_photo_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
                self._send_photo_with_retry(send_photo_url, chat_id, overview_bytes, caption)

    def _send_media_group_with_retry(
        self,
        chat_id: str,
        overview_bytes: bytes,
        zoom_bytes: bytes,
        caption: str,
    ) -> bool:
        """Send dual-image album via Telegram sendMediaGroup with retry protection."""
        if not self.bot_token or not chat_id:
            logger.warning("[Telegram] Credentials not configured, skipping alert.")
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMediaGroup"
        media = [
            {
                "type": "photo",
                "media": "attach://photo_overview.jpg",
                "caption": caption,
                "parse_mode": "HTML",
            },
            {
                "type": "photo",
                "media": "attach://photo_zoom.jpg",
            },
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                files = {
                    "photo_overview.jpg": ("photo_overview.jpg", overview_bytes, "image/jpeg"),
                    "photo_zoom.jpg": ("photo_zoom.jpg", zoom_bytes, "image/jpeg"),
                }
                data = {
                    "chat_id": chat_id,
                    "media": json.dumps(media),
                }
                response = requests.post(url, data=data, files=files, timeout=self.request_timeout)
                if response.status_code == 200:
                    logger.info(f"[TelegramNotifier] Dual-image album alert dispatched successfully to Chat ID: {chat_id}")
                    return True
                elif response.status_code in (400, 401, 404):
                    logger.warning(
                        f"[Telegram] Telegram API Error ({response.status_code}): {response.text}. Skipping alert."
                    )
                    return False
                else:
                    logger.warning(
                        f"[TelegramNotifier] sendMediaGroup returned HTTP {response.status_code} "
                        f"(Attempt {attempt}/{self.max_retries}): {response.text}"
                    )
            except requests.exceptions.Timeout:
                logger.warning(
                    f"[TelegramNotifier] Request timed out sending media group to Chat ID: {chat_id} "
                    f"(Attempt {attempt}/{self.max_retries})"
                )
            except Exception as e:
                logger.warning(
                    f"[TelegramNotifier] Network error sending media group to Chat ID: {chat_id} "
                    f"(Attempt {attempt}/{self.max_retries}): {e}"
                )

            if attempt < self.max_retries:
                time.sleep(1.0)

        return False

    def _send_photo_with_retry(self, url: str, chat_id: str, img_bytes: bytes, caption: str) -> bool:
        """Send photo with automatic retry mechanism and timeout protection."""
        if not self.bot_token or not chat_id:
            logger.warning("[Telegram] Credentials not configured, skipping alert.")
            return False

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
                elif response.status_code in (400, 401, 404):
                    logger.warning(
                        f"[Telegram] Credentials not configured or invalid (HTTP {response.status_code}): {response.text}. Skipping alert."
                    )
                    return False
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
