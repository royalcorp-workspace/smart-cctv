"""Non-blocking Hardware Buzzer Webhook Notification Module for Smart CCTV.

Dispatches HTTP POST webhooks to an IoT buzzer / alarm hardware device
using an asynchronous background queue daemon thread (Zero FPS drop).
"""

import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, List, Optional

import requests

from engine.config_loader import ensure_env_loaded
from engine.logger import logger

# Ensure environment variables are loaded
ensure_env_loaded()


class BuzzerNotifier:
    """Thread-safe, non-blocking HTTP buzzer notification dispatcher."""

    _instance: Optional["BuzzerNotifier"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self.ip: str = os.getenv("BUZZER_IP", "").strip()
        port_str = os.getenv("BUZZER_PORT", "8080").strip()
        try:
            self.port: int = int(port_str) if port_str else 8080
        except ValueError:
            self.port = 8080

        endpoint = os.getenv("BUZZER_ENDPOINT", "/trigger").strip()
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        self.endpoint: str = endpoint

        self.cooldown_sec: float = 15.0
        self.timeout_sec: float = 3.0
        self.enabled: bool = True

        self._last_triggered_time: float = 0.0
        self._last_status: str = "IDLE"
        self._last_error: Optional[str] = None
        self._total_triggers: int = 0
        self._successful_triggers: int = 0
        self._failed_triggers: int = 0

        # Background Queue & Worker Thread (Zero FPS drop on main pipeline)
        self._queue: queue.Queue = queue.Queue(maxsize=50)
        self._stop_event: threading.Event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._start_worker()

    @classmethod
    def get_instance(cls) -> "BuzzerNotifier":
        """Singleton accessor for shared buzzer notifier."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @property
    def url(self) -> str:
        """Construct full target URL."""
        if not self.ip:
            return ""
        return f"http://{self.ip}:{self.port}{self.endpoint}"

    def _start_worker(self) -> None:
        """Start background daemon worker thread."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="BuzzerNotifierWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Daemon worker loop processing buzzer trigger requests."""
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._send_http_trigger(task)
            except Exception as e:
                logger.warning(f"[BuzzerNotifier] Unexpected error during buzzer dispatch: {e}")
            finally:
                self._queue.task_done()

    def _send_http_trigger(self, task: Dict[str, Any]) -> bool:
        """Perform HTTP POST request to buzzer hardware with silent failure handling."""
        if not self.ip:
            self._last_status = "SKIPPED_NO_IP"
            logger.debug("[BuzzerNotifier] BUZZER_IP is not configured; skipping dispatch.")
            return False

        target_url = self.url
        payload = {
            "event": task.get("event_type", "tripwire_crossing"),
            "camera_id": task.get("camera_id", "unknown"),
            "timestamp": task.get("timestamp", time.time()),
            "metadata": task.get("metadata", {}),
        }

        try:
            resp = requests.post(
                target_url,
                json=payload,
                timeout=self.timeout_sec,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code in (200, 201, 204):
                self._last_status = f"SUCCESS_{resp.status_code}"
                self._last_error = None
                self._successful_triggers += 1
                logger.info(
                    f"[BuzzerNotifier] Hardware buzzer triggered successfully at {target_url} (HTTP {resp.status_code})"
                )
                return True
            else:
                self._last_status = f"HTTP_{resp.status_code}"
                self._last_error = f"HTTP status {resp.status_code}: {resp.text[:100]}"
                self._failed_triggers += 1
                logger.warning(
                    f"[BuzzerNotifier] Buzzer endpoint returned error HTTP {resp.status_code} at {target_url}"
                )
                return False
        except requests.exceptions.RequestException as e:
            self._last_status = "UNREACHABLE"
            self._last_error = str(e)
            self._failed_triggers += 1
            logger.warning(f"[BuzzerNotifier] Could not reach hardware buzzer at {target_url}: {e}")
            return False

    def trigger(
        self,
        event_type: str = "tripwire_crossing",
        camera_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Enqueue buzzer trigger event if cooldown allows (Zero FPS drop)."""
        if not self.enabled:
            return False

        now = time.time()
        with self._lock:
            if now - self._last_triggered_time < self.cooldown_sec:
                logger.debug(
                    f"[BuzzerNotifier] Cooldown active ({now - self._last_triggered_time:.1f}s < {self.cooldown_sec}s); skipping trigger."
                )
                return False

            self._last_triggered_time = now
            self._total_triggers += 1

        # Check IP configuration
        if not self.ip:
            logger.warning("[BuzzerNotifier] BUZZER_IP is empty or not configured. Buzzer notification skipped.")
            return False

        task = {
            "event_type": event_type,
            "camera_id": camera_id,
            "timestamp": now,
            "metadata": metadata or {},
        }

        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            logger.warning("[BuzzerNotifier] Worker queue full (50 tasks); dropping event.")
            return False

    def test_connection(self) -> Dict[str, Any]:
        """Perform a synchronous test ping/POST to verify buzzer reachability."""
        ensure_env_loaded()
        self.reload_config()

        if not self.ip:
            return {
                "success": False,
                "status": "NO_IP",
                "message": "BUZZER_IP is empty or not configured in .env",
                "url": "",
            }

        target_url = self.url
        try:
            resp = requests.post(
                target_url,
                json={"event": "test_ping", "camera_id": "system_test", "timestamp": time.time()},
                timeout=self.timeout_sec,
                headers={"Content-Type": "application/json"},
            )
            is_ok = resp.status_code in (200, 201, 204)
            return {
                "success": is_ok,
                "status": f"HTTP_{resp.status_code}",
                "message": f"Response HTTP {resp.status_code}" if is_ok else resp.text[:200],
                "url": target_url,
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "status": "UNREACHABLE",
                "message": f"Connection failed: {e}",
                "url": target_url,
            }

    def get_status(self) -> Dict[str, Any]:
        """Retrieve current operational status of the buzzer notifier."""
        now = time.time()
        time_since_last = now - self._last_triggered_time if self._last_triggered_time > 0 else None
        return {
            "enabled": self.enabled,
            "ip": self.ip,
            "port": self.port,
            "endpoint": self.endpoint,
            "url": self.url,
            "is_configured": bool(self.ip),
            "cooldown_sec": self.cooldown_sec,
            "cooldown_remaining_sec": max(0.0, self.cooldown_sec - time_since_last) if time_since_last is not None else 0.0,
            "last_status": self._last_status,
            "last_error": self._last_error,
            "queue_pending": self._queue.qsize(),
            "total_triggers": self._total_triggers,
            "successful_triggers": self._successful_triggers,
            "failed_triggers": self._failed_triggers,
        }

    def reload_config(self) -> None:
        """Reload configuration from environment variables."""
        ensure_env_loaded()
        self.ip = os.getenv("BUZZER_IP", "").strip()
        port_str = os.getenv("BUZZER_PORT", "8080").strip()
        try:
            self.port = int(port_str) if port_str else 8080
        except ValueError:
            self.port = 8080
        endpoint = os.getenv("BUZZER_ENDPOINT", "/trigger").strip()
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        self.endpoint = endpoint

    def stop(self) -> None:
        """Signal background worker to stop gracefully."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
