"""Thread-safe Multi-Camera In-Memory Frame and Telemetry Buffer.

Provides O(1) constant memory usage with in-place frame dropping to decouple
AI video processing pipelines from asynchronous web client streaming.
"""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class MultiCameraBuffer:
    """Thread-safe singleton registry storing the latest frame and telemetry per camera."""

    _instance: Optional["MultiCameraBuffer"] = None
    _singleton_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._conditions: Dict[str, threading.Condition] = {}
        self._frames: Dict[str, bytes] = {}
        self._frame_seqs: Dict[str, int] = {}
        self._telemetries: Dict[str, Dict[str, Any]] = {}
        self._camera_names: Dict[str, str] = {}
        self._last_update_times: Dict[str, float] = {}

    @classmethod
    def get_instance(cls) -> "MultiCameraBuffer":
        """Singleton accessor for shared multi-camera buffer."""
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register_camera(self, camera_id: str, camera_name: str = "") -> None:
        """Register a camera in the buffer registry."""
        with self._lock:
            if camera_id not in self._conditions:
                self._conditions[camera_id] = threading.Condition(self._lock)
            self._camera_names[camera_id] = camera_name or camera_id
            if camera_id not in self._frame_seqs:
                self._frame_seqs[camera_id] = 0
            if camera_id not in self._telemetries:
                self._telemetries[camera_id] = {
                    "camera_id": camera_id,
                    "name": self._camera_names[camera_id],
                    "online": False,
                    "is_connected": False,
                    "fps": 0.0,
                    "rtsp_status": "Connecting",
                    "violations": 0,
                    "clear_area_count": 0,
                    "active_tracks": 0,
                    "identified_faces": [],
                    "updated_at": time.time(),
                }

    def update_frame(
        self,
        camera_id: str,
        frame: Union[np.ndarray, bytes],
        telemetry: Optional[Dict[str, Any]] = None,
        camera_name: str = "",
    ) -> None:
        """Atomically overwrite the latest JPEG frame and telemetry for camera_id."""
        jpeg_bytes: bytes
        if isinstance(frame, np.ndarray):
            # Encode frame to JPEG with optimal performance/quality tradeoff (80%)
            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                return
            jpeg_bytes = encoded.tobytes()
        else:
            jpeg_bytes = frame

        now = time.time()
        with self._lock:
            if camera_id not in self._conditions:
                self._conditions[camera_id] = threading.Condition(self._lock)

            if camera_name:
                self._camera_names[camera_id] = camera_name

            self._frames[camera_id] = jpeg_bytes
            self._frame_seqs[camera_id] = self._frame_seqs.get(camera_id, 0) + 1
            self._last_update_times[camera_id] = now

            # Update telemetry dictionary
            base_telem: Dict[str, Any] = {
                "camera_id": camera_id,
                "name": self._camera_names.get(camera_id, camera_id),
                "online": True,
                "is_connected": True,
                "fps": 0.0,
                "rtsp_status": "Connected",
                "violations": 0,
                "clear_area_count": 0,
                "active_tracks": 0,
                "identified_faces": [],
                "updated_at": now,
            }
            if telemetry:
                base_telem.update(telemetry)
            self._telemetries[camera_id] = base_telem

            # Wake up all waiting streaming consumers for this camera
            self._conditions[camera_id].notify_all()

    def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        """Return the latest JPEG bytes for a camera, or None if not yet available."""
        with self._lock:
            return self._frames.get(camera_id)

    def get_telemetry(self, camera_id: str) -> Dict[str, Any]:
        """Return the current telemetry dictionary for a camera."""
        now = time.time()
        with self._lock:
            telem = dict(self._telemetries.get(camera_id, {}))
            if not telem:
                telem = {
                    "camera_id": camera_id,
                    "name": self._camera_names.get(camera_id, camera_id),
                    "online": False,
                    "is_connected": False,
                    "fps": 0.0,
                    "rtsp_status": "Not Configured",
                    "violations": 0,
                    "clear_area_count": 0,
                    "active_tracks": 0,
                    "identified_faces": [],
                    "updated_at": now,
                }
            else:
                # Mark offline if no update for > 4.0 seconds
                last_up = self._last_update_times.get(camera_id, 0.0)
                if now - last_up > 4.0:
                    telem["online"] = False
                    telem["is_connected"] = False
                    telem["rtsp_status"] = "Signal Lost / Reconnecting"
            return telem

    def get_cameras(self) -> List[Dict[str, Any]]:
        """Return a list of all registered cameras and basic online statuses."""
        now = time.time()
        cameras: List[Dict[str, Any]] = []
        with self._lock:
            for cam_id, name in self._camera_names.items():
                last_up = self._last_update_times.get(cam_id, 0.0)
                is_online = (now - last_up) <= 4.0 if last_up > 0 else False
                cameras.append({
                    "id": cam_id,
                    "name": name,
                    "online": is_online,
                })
        return sorted(cameras, key=lambda c: c["id"])

    def create_placeholder_frame(self, message: str = "Menghubungkan ke Feed Kamera...") -> bytes:
        """Generate a lightweight standby JPEG frame."""
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)

        # Subtle dark grid
        for y in range(0, 720, 40):
            cv2.line(canvas, (0, y), (1280, y), (28, 28, 28), 1)
        for x in range(0, 1280, 40):
            cv2.line(canvas, (x, 0), (x, 720), (28, 28, 28), 1)

        # Center card
        cx, cy = 640, 360
        cv2.rectangle(canvas, (cx - 260, cy - 60), (cx + 260, cy + 60), (32, 32, 32), -1)
        cv2.rectangle(canvas, (cx - 260, cy - 60), (cx + 260, cy + 60), (60, 60, 60), 1)

        cv2.putText(
            canvas,
            "SMART CCTV 2.0 - STANDBY",
            (cx - 210, cy - 15),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 200, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            message,
            (cx - 200, cy + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            lineType=cv2.LINE_AA,
        )

        _, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return enc.tobytes()

    def stream_generator(self, camera_id: str):
        """Yield multipart/x-mixed-replace MJPEG byte stream for HTTP response."""
        last_seq = -1
        standby_bytes = self.create_placeholder_frame(f"Kamera '{camera_id}' belum aktif...")

        while True:
            frame_data: Optional[bytes] = None
            with self._lock:
                cond = self._conditions.setdefault(camera_id, threading.Condition(self._lock))
                cur_seq = self._frame_seqs.get(camera_id, 0)

                # Wait for next frame if current frame was already yielded
                if cur_seq == last_seq or camera_id not in self._frames:
                    # Timeout after 0.5s to permit periodic keep-alive and responsiveness
                    cond.wait(timeout=0.5)

                cur_seq = self._frame_seqs.get(camera_id, 0)
                if cur_seq != last_seq and camera_id in self._frames:
                    frame_data = self._frames[camera_id]
                    last_seq = cur_seq

            if frame_data is None:
                # If no frame received yet, send standby frame occasionally
                frame_data = standby_bytes
                time.sleep(0.1)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
            )
