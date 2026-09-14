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

import datetime
import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from engine.config_loader import ensure_env_loaded, load_camera_config
from engine.logger import logger

# Ensure environment variables are loaded
ensure_env_loaded()


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

        # Scan cameras/*/config.json for camera-specific telegram configuration (expanded via config_loader)
        default_config = Path(__file__).resolve().parent.parent / "configs" / "telegram.json"
        if self.config_path == default_config:
            try:
                cameras_dir = Path(__file__).resolve().parent.parent / "cameras"
                if cameras_dir.is_dir():
                    for cam_dir in cameras_dir.iterdir():
                        if cam_dir.is_dir():
                            cam_cfg_path = cam_dir / "config.json"
                            if cam_cfg_path.exists():
                                cam_data = load_camera_config(cam_cfg_path)
                                cam_tg = cam_data.get("telegram", {})
                                if isinstance(cam_tg, dict):
                                    cam_token = str(cam_tg.get("bot_token", "")).strip()
                                    if not self.bot_token and cam_token and not cam_token.startswith("${"):
                                        self.bot_token = cam_token
                                    cam_chat_id = str(cam_tg.get("chat_id", "")).strip()
                                    if cam_chat_id and not cam_chat_id.startswith("${"):
                                        cam_id = cam_data.get("camera_id", cam_dir.name)
                                        cur_list = self.camera_routing.setdefault(cam_id, [])
                                        if cam_chat_id not in cur_list:
                                            cur_list.append(cam_chat_id)
                                    if cam_tg.get("enabled"):
                                        self.enabled = True
            except Exception as e:
                logger.debug(f"[TelegramNotifier] Camera config scan note: {e}")

        # Environment variable overrides (Highest priority)
        env_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if env_token and env_token.strip() and not env_token.startswith("${"):
            self.bot_token = env_token.strip()

        env_admin = os.getenv("TELEGRAM_GLOBAL_ADMIN_CHAT_ID")
        if env_admin and env_admin.strip() and not env_admin.startswith("${"):
            admin_id = env_admin.strip()
            if admin_id not in self.global_admins:
                self.global_admins.append(admin_id)

        for env_k, env_v in os.environ.items():
            if env_k.startswith("TELEGRAM_CAM") and env_k.endswith("_CHAT_ID"):
                cam_suffix = env_k[len("TELEGRAM_"): -len("_CHAT_ID")].lower()  # e.g. cam01 -> cam_01
                if "_" not in cam_suffix and len(cam_suffix) >= 4:
                    cam_key = f"{cam_suffix[:3]}_{cam_suffix[3:]}"
                else:
                    cam_key = cam_suffix
                if env_v and env_v.strip() and not env_v.startswith("${"):
                    c_id = env_v.strip()
                    c_list = self.camera_routing.setdefault(cam_key, [])
                    if c_id not in c_list:
                        c_list.append(c_id)

        env_enabled = os.getenv("TELEGRAM_ENABLED")
        if env_enabled is not None:
            self.enabled = env_enabled.strip().lower() in ("true", "1", "yes")

        # Sanitize bot_token if placeholder remained unexpanded
        if self.bot_token.startswith("${"):
            self.bot_token = ""

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

        # Auto-scale bbox if bbox is in 640x360 inference space and frame is larger (e.g. 1080p)
        if (fw, fh) != (640, 360):
            if bx < 640 and by < 360 and bw <= 640 and bh <= 360:
                scale_x = fw / 640.0
                scale_y = fh / 360.0
            else:
                scale_x = 1.0
                scale_y = 1.0
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
        class_label: str = "OBJEK",
        owner_name: Optional[str] = None,
        owner_face_crop: Optional[np.ndarray] = None,
    ) -> bool:
        """Enqueue alert job asynchronously (non-blocking, < 0.1ms).

        Returns True if enqueued, False if dropped or disabled.
        """
        # Strict dwell threshold: reads directly from active zone config, default 3600.0s
        dwell_thresh = 3600.0
        if zones and isinstance(zones, dict) and zone_id in zones:
            z_cfg = zones[zone_id]
            if isinstance(z_cfg, dict):
                dwell_thresh = float(z_cfg.get("dwell_threshold_sec", z_cfg.get("dwell_time_threshold", z_cfg.get("unattended_threshold", 3600.0))))

        if dwell_duration < dwell_thresh:
            logger.debug(f"[TelegramNotifier] Dwell duration {dwell_duration:.1f}s < {dwell_thresh:.1f}s threshold. Alert skipped.")
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
                "type": "alert",
                "camera_id": camera_id,
                "zone_id": zone_id,
                "zone_name": zone_name or zone_id,
                "track_id": track_id,
                "class_label": class_label,
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

    def dispatch_resolution(
        self,
        camera_id: str,
        zone_name: str,
        track_id: int,
        dwell_duration: float,
        timestamp_str: str,
        owner_name: Optional[str] = None,
    ) -> bool:
        """Enqueue a resolution notice when stationary object is removed."""
        if not self.enabled or not self.bot_token or not self.bot_token.strip():
            return False

        recipients = self.get_recipients_for_camera(camera_id)
        if not recipients:
            return False

        job = {
            "type": "resolution",
            "camera_id": camera_id,
            "zone_name": zone_name,
            "track_id": track_id,
            "dwell_duration": dwell_duration,
            "timestamp_str": timestamp_str,
            "owner_name": owner_name or "Tidak Teridentifikasi",
            "recipients": recipients,
        }
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def dispatch_composite_alert(self, *args, **kwargs) -> bool:
        """Deprecated: Bypassed in favor of standardized dual-image sendMediaGroup."""
        logger.debug("[TelegramNotifier] dispatch_composite_alert bypassed (standardized dual-image format).")
        return False

    def dispatch_tripwire_alert(
        self,
        camera_id: str,
        line_id: str,
        line_name: str,
        track_id: int,
        class_label: str,
        direction: str,
        timestamp_str: str,
        frame: np.ndarray,
        overview_frame: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """Enqueue tripwire line breach alert asynchronously with dual-image format."""
        if not self.enabled or not self.bot_token or not self.bot_token.strip():
            return False

        try:
            recipients = self.get_recipients_for_camera(camera_id)
            if not recipients:
                return False

            job = {
                "type": "tripwire_alert",
                "camera_id": camera_id,
                "line_id": line_id,
                "line_name": line_name,
                "track_id": track_id,
                "class_label": class_label,
                "direction": direction,
                "timestamp_str": timestamp_str,
                "frame": frame.copy(),
                "overview_frame": overview_frame.copy() if overview_frame is not None else None,
                "bbox": bbox,
                "recipients": recipients,
            }
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            logger.warning("[TelegramNotifier] Alert queue is full! Dropping tripwire alert.")
            return False
        except Exception as e:
            logger.warning(f"[TelegramNotifier] Error dispatching tripwire alert: {e}")
            return False

    def dispatch_vehicle_alert(
        self,
        camera_id: str,
        zone_id: str,
        track: Any,
        alert_label: str,
        frame: np.ndarray,
        zone_name: Optional[str] = None,
        overview_frame: Optional[np.ndarray] = None,
        timestamp_str: Optional[str] = None,
    ) -> bool:
        """Enqueue vehicle parking/obstruction dwell violation alert job asynchronously with dual-image format."""
        if not self.enabled or not self.bot_token or not self.bot_token.strip():
            logger.warning("[Telegram] Credentials not configured, skipping vehicle alert.")
            return False

        recipients = self.get_recipients_for_camera(camera_id)
        if not recipients:
            logger.warning("[Telegram] No recipients configured, skipping vehicle alert.")
            return False

        try:
            if not timestamp_str:
                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            job = {
                "type": "vehicle_alert",
                "camera_id": camera_id,
                "zone_id": zone_id,
                "zone_name": zone_name or zone_id,
                "track_id": getattr(track, "track_id", 0),
                "class_label": getattr(track, "class_label", "VEHICLE"),
                "alert_label": alert_label,
                "dwell_duration": float(getattr(track, "dwell_duration", 0.0)),
                "dwell_threshold": float(getattr(track, "dwell_threshold", 60.0)),
                "timestamp_str": timestamp_str,
                "frame": frame.copy(),
                "overview_frame": overview_frame.copy() if overview_frame is not None else None,
                "bbox": getattr(track, "bbox", None),
                "recipients": recipients,
            }

            self._queue.put_nowait(job)
            return True
        except queue.Full:
            logger.warning("[TelegramNotifier] Alert queue full! Dropping vehicle alert.")
            return False
        except Exception as e:
            logger.warning(f"[Telegram] Error enqueuing vehicle alert: {e}")
            return False

    def _worker_loop(self) -> None:
        """Daemon worker loop: consumes queue and sends HTTP requests via Telegram Bot API."""
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if job.get("type") == "resolution":
                    self._send_resolution_job(job)
                elif job.get("type") == "composite_alert":
                    self._send_composite_job(job)
                elif job.get("type") == "tripwire_alert":
                    self._send_tripwire_job(job)
                elif job.get("type") == "vehicle_alert":
                    self._send_vehicle_job(job)
                else:
                    self._send_job(job)
            except Exception as e:
                logger.error(f"[TelegramNotifier] Unhandled exception in worker: {e}")
            finally:
                self._queue.task_done()

    def _send_tripwire_job(self, job: dict) -> None:
        """Send tripwire line crossing alert with standardized dual-image album (Overview + Zoom Detail)."""
        camera_id = job["camera_id"]
        line_id = job["line_id"]
        line_name = job["line_name"]
        track_id = job["track_id"]
        class_label = str(job.get("class_label", "OBJECT")).upper()
        direction = job.get("direction", "both")
        timestamp_str = job["timestamp_str"]
        clean_frame = job.get("frame")
        overview_frame = job.get("overview_frame")
        bbox = job.get("bbox")
        recipients = job.get("recipients", [])

        # 1. Overview Image (Full scene with HUD line and direction arrow)
        if overview_frame is not None and overview_frame.size > 0:
            annotated = overview_frame.copy()
        elif clean_frame is not None and clean_frame.size > 0:
            annotated = clean_frame.copy()
        else:
            logger.error("[TelegramNotifier] Tripwire job missing valid frames.")
            return

        success_ov, enc_overview = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success_ov:
            logger.error("[TelegramNotifier] Failed to encode tripwire overview image to JPEG.")
            return
        overview_bytes = enc_overview.tobytes()

        # 2. Contextual Close-up Zoom Crop of violator
        zoom_crop = None
        if clean_frame is not None and clean_frame.size > 0 and bbox is not None:
            zoom_crop = self.create_zoom_crop(clean_frame, bbox, min_width=480, padding_ratio=0.30)

        # Fallback if zoom_crop couldn't be generated
        if zoom_crop is None or zoom_crop.size == 0:
            zoom_crop = annotated.copy()

        success_zm, enc_zoom = cv2.imencode(".jpg", zoom_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success_zm:
            logger.error("[TelegramNotifier] Failed to encode tripwire zoom crop image to JPEG.")
            return
        zoom_bytes = enc_zoom.tobytes()

        # 3. Standardized HTML Caption
        caption = (
            "🚨 <b>[ALERT] TEROBOSAN GARIS VIRTUAL (TRIPWIRE)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📹 <b>Kamera</b>: <code>{camera_id.upper()}</code>\n"
            f"📍 <b>Lokasi / Zona</b>: <b>{line_name} ({line_id})</b>\n"
            f"🎯 <b>Target</b>: <b>{class_label} #{track_id}</b>\n"
            f"⏱️ <b>Durasi / Status</b>: <b>Melintas Arah: {direction}</b>\n"
            f"🕒 <b>Waktu</b>: <code>{timestamp_str}</code>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ <i>Bukti visual: [1] Area Kejadian  |  [2] Detail Pelanggar</i>"
        )

        base_url = getattr(self, "api_base_url", "https://api.telegram.org")
        for chat_id in recipients:
            sent = self._send_media_group_with_retry(chat_id, overview_bytes, zoom_bytes, caption)
            if not sent:
                send_photo_url = f"{base_url}/bot{self.bot_token}/sendPhoto"
                self._send_photo_with_retry(send_photo_url, chat_id, overview_bytes, caption)
                zoom_caption = f"🔍 <b>[DETAIL PELANGGAR]</b> {class_label} #{track_id}"
                self._send_photo_with_retry(send_photo_url, chat_id, zoom_bytes, zoom_caption)

    def _send_vehicle_job(self, job: dict) -> None:
        """Send vehicle obstruction/parking violation alert with standardized dual-image album."""
        camera_id = job["camera_id"]
        zone_name = job["zone_name"]
        track_id = job["track_id"]
        class_label = str(job.get("class_label", "VEHICLE")).upper()
        alert_label = job.get("alert_label", "PARKIR MELEBIHI BATAS")
        dwell_sec = job["dwell_duration"]
        thresh_sec = job["dwell_threshold"]
        timestamp_str = job["timestamp_str"]
        clean_frame = job.get("frame")
        overview_frame = job.get("overview_frame")
        bbox = job.get("bbox")
        recipients = job.get("recipients", [])

        dwell_str = f"{dwell_sec:.0f}s" if dwell_sec < 120 else f"{dwell_sec / 60.0:.1f} Menit"
        thresh_str = f"{thresh_sec:.0f}s" if thresh_sec < 120 else f"{thresh_sec / 60.0:.0f} Menit"

        # 1. Overview Image (Full scene with HUD annotations)
        if overview_frame is not None and overview_frame.size > 0:
            annotated = overview_frame.copy()
        elif clean_frame is not None and clean_frame.size > 0:
            annotated = clean_frame.copy()
        else:
            logger.error("[TelegramNotifier] Vehicle job missing valid frames.")
            return

        success_ov, enc_overview = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success_ov:
            logger.error("[TelegramNotifier] Failed to encode vehicle overview image to JPEG.")
            return
        overview_bytes = enc_overview.tobytes()

        # 2. Contextual Close-up Zoom Crop of vehicle
        zoom_crop = None
        if clean_frame is not None and clean_frame.size > 0 and bbox is not None:
            zoom_crop = self.create_zoom_crop(clean_frame, bbox, min_width=480, padding_ratio=0.25)

        # Fallback if zoom_crop couldn't be generated
        if zoom_crop is None or zoom_crop.size == 0:
            zoom_crop = annotated.copy()

        success_zm, enc_zoom = cv2.imencode(".jpg", zoom_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not success_zm:
            logger.error("[TelegramNotifier] Failed to encode vehicle zoom crop image to JPEG.")
            return
        zoom_bytes = enc_zoom.tobytes()

        # 3. Standardized HTML Caption
        caption = (
            f"🚨 <b>[ALERT] {alert_label.upper()}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📹 <b>Kamera</b>: <code>{camera_id.upper()}</code>\n"
            f"📍 <b>Lokasi / Zona</b>: <b>{zone_name}</b>\n"
            f"🎯 <b>Target</b>: <b>{class_label} #{track_id}</b>\n"
            f"⏱️ <b>Durasi / Status</b>: <b>{dwell_str} (Batas: {thresh_str})</b>\n"
            f"🕒 <b>Waktu</b>: <code>{timestamp_str}</code>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ <i>Bukti visual: [1] Area Kejadian  |  [2] Detail Pelanggar</i>"
        )

        base_url = getattr(self, "api_base_url", "https://api.telegram.org")
        for chat_id in recipients:
            sent = self._send_media_group_with_retry(chat_id, overview_bytes, zoom_bytes, caption)
            if not sent:
                send_photo_url = f"{base_url}/bot{self.bot_token}/sendPhoto"
                self._send_photo_with_retry(send_photo_url, chat_id, overview_bytes, caption)
                zoom_caption = f"🔍 <b>[DETAIL PELANGGAR]</b> {class_label} #{track_id}"
                self._send_photo_with_retry(send_photo_url, chat_id, zoom_bytes, zoom_caption)

    def _send_job(self, job: dict) -> None:
        """Render annotated overview + contextual zoom crop and dispatch via sendMediaGroup."""
        camera_id = job["camera_id"]
        zone_id = job["zone_id"]
        zone_name = job.get("zone_name", zone_id)
        display_zone = zone_name if zone_name else zone_id
        track_id = job["track_id"]
        class_label = str(job.get("class_label", "OBJEK")).upper()
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

        success_ov, enc_overview = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success_ov:
            logger.error("[TelegramNotifier] Failed to encode overview snapshot image to JPEG.")
            return

        overview_bytes = enc_overview.tobytes()

        # 2. Extract Contextual Close-up Zoom Crop from clean frame (un-occluded, 35% padding)
        zoom_crop = self.create_zoom_crop(clean_frame, bbox, min_width=480, padding_ratio=0.35)
        if zoom_crop is None or zoom_crop.size == 0:
            zoom_crop = annotated.copy()

        success_zm, enc_zoom = cv2.imencode(".jpg", zoom_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success_zm:
            logger.error("[TelegramNotifier] Failed to encode zoom crop image to JPEG.")
            return

        zoom_bytes = enc_zoom.tobytes()

        # Dwell threshold
        dwell_thresh = 3600.0
        if zones and isinstance(zones, dict) and zone_id in zones:
            z_cfg = zones[zone_id]
            if isinstance(z_cfg, dict):
                dwell_thresh = float(z_cfg.get("dwell_threshold_sec", z_cfg.get("dwell_time_threshold", z_cfg.get("unattended_threshold", 3600.0))))
        
        dwell_str = f"{dwell:.0f}s" if dwell < 120 else f"{dwell / 60.0:.1f} Menit"
        thresh_str = f"{dwell_thresh:.0f}s" if dwell_thresh < 120 else f"{dwell_thresh / 60.0:.0f} Menit"
        dwell_status = f"{dwell_str} (Batas: {thresh_str})"

        # 3. Standardized HTML Caption
        caption = (
            "🚨 <b>[ALERT] PELANGGARAN CLEAR AREA</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📹 <b>Kamera</b>: <code>{camera_id.upper()}</code>\n"
            f"📍 <b>Lokasi / Zona</b>: <b>{display_zone}</b>\n"
            f"🎯 <b>Target</b>: <b>{class_label.upper()} #{track_id}</b>\n"
            f"⏱️ <b>Durasi / Status</b>: <b>{dwell_status}</b>\n"
            f"🕒 <b>Waktu</b>: <code>{timestamp_str}</code>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ <i>Bukti visual: [1] Area Kejadian  |  [2] Detail Pelanggar</i>"
        )

        base_url = getattr(self, "api_base_url", "https://api.telegram.org")
        for chat_id in recipients:
            sent = self._send_media_group_with_retry(chat_id, overview_bytes, zoom_bytes, caption)
            if not sent:
                send_photo_url = f"{base_url}/bot{self.bot_token}/sendPhoto"
                self._send_photo_with_retry(send_photo_url, chat_id, overview_bytes, caption)
                zoom_caption = f"🔍 <b>[DETAIL PELANGGAR]</b> {class_label} #{track_id}"
                self._send_photo_with_retry(send_photo_url, chat_id, zoom_bytes, zoom_caption)

    def _send_resolution_job(self, job: dict) -> None:
        """Send HTML text message to Telegram when incident is resolved."""
        camera_id = job["camera_id"]
        zone_name = job["zone_name"]
        track_id = job["track_id"]
        dwell = job["dwell_duration"]
        timestamp_str = job["timestamp_str"]
        owner_name = job.get("owner_name", "Tidak Teridentifikasi")
        recipients = job.get("recipients", [])

        dwell_mins = dwell / 60.0
        text = (
            "✅ <b>AREA KEMBALI STERIL (RESOLVED)</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📹 <b>Kamera</b>: <code>{camera_id}</code>\n"
            f"📍 <b>Zona</b>: <code>{zone_name}</code>\n"
            f"🏷️ <b>Objek</b>: Objek Terlarang (Track ID #{track_id})\n"
            f"👤 <b>Terkait</b>: <code>{owner_name}</code>\n"
            f"⏱️ <b>Total Durasi Diam</b>: <b>{dwell_mins:.1f} Menit</b>\n"
            f"🕒 <b>Waktu Penyelesaian</b>: <code>{timestamp_str}</code>\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "✨ <i>Objek telah diambil atau dipindahkan dari area steril. Kondisi area telah kembali normal dan steril.</i>"
        )

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        for chat_id in recipients:
            for attempt in range(1, self.max_retries + 1):
                try:
                    payload = {
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                    }
                    resp = requests.post(url, json=payload, timeout=self.request_timeout)
                    if resp.status_code == 200:
                        logger.info(f"[TelegramNotifier] Resolution notice sent to Chat ID {chat_id}")
                        break
                except Exception as e:
                    logger.warning(f"[TelegramNotifier] Failed to send resolution message: {e}")

    def _send_composite_job(self, job: dict) -> None:
        """Deprecated: Pass-through for composite alert."""
        pass

    def _send_media_group_with_retry(
        self,
        chat_id: str,
        overview_bytes: bytes,
        zoom_bytes: bytes,
        caption: str,
    ) -> bool:
        """Send standardized 2-image album via Telegram sendMediaGroup with retry protection."""
        if not self.bot_token or not chat_id:
            logger.warning("[Telegram] Credentials not configured, skipping alert.")
            return False

        base_url = getattr(self, "api_base_url", "https://api.telegram.org")
        url = f"{base_url}/bot{self.bot_token}/sendMediaGroup"
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
