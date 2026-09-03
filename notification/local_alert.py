"""Local audio alerts and dynamic visual HUD renderer."""

import math
import struct
import threading
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Audio Constants
ALARM_DURATION_SEC: float = 3.0
COOLDOWN_SILENT_SEC: float = 15.0
CYCLE_TOTAL_SEC: float = ALARM_DURATION_SEC + COOLDOWN_SILENT_SEC

# Visual Palette (BGR)
COLOR_SAFE: Tuple[int, int, int] = (0, 255, 0)         # Green
COLOR_WARNING: Tuple[int, int, int] = (0, 215, 255)    # Amber / Yellow
COLOR_VIOLATION: Tuple[int, int, int] = (0, 0, 255)    # Red
COLOR_HUD_BG: Tuple[int, int, int] = (20, 20, 20)      # Dark gray HUD
COLOR_TEXT_MAIN: Tuple[int, int, int] = (240, 240, 240)


def generate_fallback_wav(target_path: Path, frequency_hz: float = 660.0, duration_sec: float = 1.0) -> None:
    """Generate a clean PCM sine beep fallback WAV file."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate: int = 22050
    num_samples: int = int(sample_rate * duration_sec)
    
    with wave.open(str(target_path), "w") as wav_file:
        wav_file.setnchannels(1)        # Mono
        wav_file.setsampwidth(2)       # 16-bit
        wav_file.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            t = float(i) / sample_rate
            # 2 short beeps within the duration
            envelope = 1.0 if (i % (sample_rate // 2) < (sample_rate // 3)) else 0.0
            sample = int(16000 * envelope * math.sin(2.0 * math.pi * frequency_hz * t))
            frames.extend(struct.pack("<h", sample))
        
        wav_file.writeframes(frames)


class GlobalAudioWorker:
    """Singleton audio manager to prevent concurrent sound playback collisions across cameras."""

    _instance: Optional["GlobalAudioWorker"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "GlobalAudioWorker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GlobalAudioWorker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._audio_available: bool = False
        self._sound_obj = None
        self._last_cycle_start: float = 0.0
        self._playback_lock: threading.Lock = threading.Lock()
        
        # Audio file path
        base_dir = Path(__file__).resolve().parent
        self.sound_file: Path = base_dir / "alert.wav"
        if not self.sound_file.exists():
            try:
                generate_fallback_wav(self.sound_file)
            except Exception as e:
                print(f"[WARN] Failed to generate fallback WAV: {e}")

        # Initialize Pygame Mixer safely
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            if self.sound_file.exists():
                self._sound_obj = pygame.mixer.Sound(str(self.sound_file))
            self._audio_available = True
        except Exception as e:
            print(f"[WARN] Audio output unavailable (running in silent mode): {e}")
            self._audio_available = False

        self._initialized = True

    def request_alarm(self) -> None:
        """Trigger alarm cycle if outside silent cooldown and not busy."""
        if not self._audio_available or self._sound_obj is None:
            return

        now = time.time()
        with self._playback_lock:
            import pygame
            # Check 18-second overall cycle (3s play + 15s silent)
            if (now - self._last_cycle_start) >= CYCLE_TOTAL_SEC:
                if not pygame.mixer.get_busy():
                    self._sound_obj.play()
                    self._last_cycle_start = now


class VisualHUD:
    """Decoupled visual overlay renderer for zones, bounding boxes, and system HUD."""

    @staticmethod
    def render(
        canvas: np.ndarray,
        zones: Dict[str, List[List[int]]],
        tracked_objects: list,
        camera_id: str,
        fps: float,
        is_connected: bool,
    ) -> np.ndarray:
        """Render complete CCTV visual indicators onto canvas."""
        h, w = canvas.shape[:2]
        now = time.time()
        blink_state = (int(now * 2) % 2) == 0  # 500ms toggle

        # Identify which zones currently contain triggered violations
        violated_zones = {obj.zone_id for obj in tracked_objects if getattr(obj, "is_triggered", False)}

        # 1. Draw ROI Zones
        for zone_id, pts in zones.items():
            if len(pts) < 3:
                continue
            np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            is_violated = zone_id in violated_zones

            if is_violated:
                zone_color = COLOR_VIOLATION if blink_state else COLOR_WARNING
                thickness = 4
            else:
                zone_color = COLOR_SAFE
                thickness = 2

            cv2.polylines(canvas, [np_pts], isClosed=True, color=zone_color, thickness=thickness, lineType=cv2.LINE_AA)
            label_pos = (pts[0][0] + 5, pts[0][1] + 20)
            cv2.putText(canvas, zone_id, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1)

        # 2. Draw Bounding Boxes and Status Badges
        for obj in tracked_objects:
            x, y, bw, bh = obj.bbox
            track_id = obj.track_id
            dwell = obj.dwell_duration

            if getattr(obj, "is_triggered", False):
                box_color = COLOR_VIOLATION
                badge_text = f"ID {track_id}: VIOLATION! ({dwell:.1f}s)"
            elif getattr(obj, "is_stationary", False):
                box_color = COLOR_WARNING
                badge_text = f"ID {track_id}: Dwell {dwell:.1f}s"
            else:
                box_color = COLOR_SAFE
                badge_text = f"ID {track_id}: Moving"

            # Draw bounding rectangle
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), box_color, 2)
            cv2.circle(canvas, obj.centroid, 4, box_color, -1)

            # Badge background for high legibility
            text_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            badge_y1 = max(0, y - text_size[1] - 6)
            badge_y2 = y
            cv2.rectangle(canvas, (x, badge_y1), (x + text_size[0] + 8, badge_y2), box_color, -1)
            cv2.putText(
                canvas,
                badge_text,
                (x + 4, badge_y2 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0) if box_color != COLOR_VIOLATION else (255, 255, 255),
                1,
            )

        # 3. Top HUD Banner
        hud_height = 42
        hud_bar = np.zeros((hud_height, w, 3), dtype=np.uint8)
        hud_bar[:] = COLOR_HUD_BG
        canvas[0:hud_height, 0:w] = cv2.addWeighted(canvas[0:hud_height, 0:w], 0.25, hud_bar, 0.75, 0)

        conn_text = "ONLINE" if is_connected else "RECONNECTING"
        conn_color = COLOR_SAFE if is_connected else COLOR_VIOLATION

        hud_left = f"CAM: {camera_id} | RES: {w}x{h} | FPS: {fps:.1f}"
        hud_right = f"ACTIVE TARGETS: {len(tracked_objects)} | RTSP: {conn_text}"

        cv2.putText(canvas, hud_left, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        r_size = cv2.getTextSize(hud_right, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(canvas, hud_right, (w - r_size[0] - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, conn_color, 2, cv2.LINE_AA)

        return canvas
