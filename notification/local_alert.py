"""Local audio alerts and dynamic visual HUD renderer."""

import datetime
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

        # Identify which zones currently contain triggered violations (strictly for unattended bags)
        violated_zones = {
            obj.zone_id
            for obj in tracked_objects
            if getattr(obj, "is_triggered", False)
            and getattr(obj, "class_label", "") in ("tas", "backpack", "handbag", "suitcase")
        }

        # 1. Draw ROI Zones
        for zone_id, pts in zones.items():
            if len(pts) < 3:
                continue
            np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            is_violated = zone_id in violated_zones

            if is_violated:
                zone_color = COLOR_VIOLATION if blink_state else COLOR_WARNING
                thickness = 2
            else:
                zone_color = COLOR_SAFE
                thickness = 1

            cv2.polylines(canvas, [np_pts], isClosed=True, color=zone_color, thickness=thickness, lineType=cv2.LINE_AA)

        # 2. Draw Bounding Boxes and Status Badges (Refined for 640x480)
        font_scale = 0.40
        font_thick = 1
        pad_x = 3
        pad_y = 2

        for obj in tracked_objects:
            # EDGE-AWARE VISUAL FILTER: Render active objects and in-zone grace period objects (anti-flicker).
            # Never render ghost boxes near perimeter or exiting boundaries.
            if hasattr(obj, "should_render"):
                if not obj.should_render:
                    continue
            elif not getattr(obj, "is_active_this_frame", False):
                continue

            # STRICT ROI FILTER: Never render objects outside active ROI zones
            if not getattr(obj, "zone_id", None) or obj.zone_id not in zones:
                continue

            x, y, bw, bh = obj.bbox
            track_id = obj.track_id
            dwell = getattr(obj, "dwell_duration", 0.0)
            raw_label = getattr(obj, "class_label", "object")
            is_bag = raw_label in ("tas", "backpack", "handbag", "suitcase")
            is_attended = getattr(obj, "is_attended", False)

            dwell_max = int(getattr(obj, "dwell_threshold", 60))

            if not is_bag:
                # PERSON ALWAYS NORMAL / GREEN (COLOR_SAFE) - Loitering alert disabled
                box_color = COLOR_SAFE
                badge_text = f"ID {track_id} | Person"
            else:
                # BAG OBJECT (ATTENDED / UNATTENDED / ALERT)
                if is_attended:
                    # Attended / Ada Orang (Hijau)
                    box_color = COLOR_SAFE
                    badge_text = f"ID {track_id} | Tas [AMAN / ATTENDED]"
                elif getattr(obj, "is_triggered", False):
                    # Alert (Merah / Oranye Berkedip)
                    box_color = (0, 0, 255) if blink_state else (0, 165, 255)
                    badge_text = f"[ALERT] ID {track_id} | Tas ({dwell:.0f}s)"
                elif getattr(obj, "is_stationary", False):
                    # Unattended (Kuning / Oranye)
                    if dwell <= 30.0:
                        box_color = (0, 255, 255)  # 0 s.d. 30 detik: Kuning
                    else:
                        box_color = (0, 165, 255)  # 31 s.d. threshold: Oranye Warning
                    badge_text = f"ID {track_id} | Tas ({dwell:.0f}s/{dwell_max}s)"
                else:
                    box_color = COLOR_SAFE
                    badge_text = f"ID {track_id} | Tas"

            # Draw crisp thin bounding rectangle (thickness = 1) and small centroid
            cv2.rectangle(canvas, (x, y), (x + bw, y + bh), box_color, 1)
            cv2.circle(canvas, obj.centroid, 2, box_color, -1)

            # Compact badge background with minimal padding
            (text_w, text_h), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            badge_y1 = max(0, y - text_h - (pad_y * 2) - 1)
            badge_y2 = y
            badge_x2 = min(w, x + text_w + (pad_x * 2))

            cv2.rectangle(canvas, (x, badge_y1), (badge_x2, badge_y2), box_color, -1)
            
            # High-contrast text color: white on red, black on green/yellow/orange
            is_dark_bg = box_color == (0, 0, 255)
            text_color = (255, 255, 255) if is_dark_bg else (0, 0, 0)

            cv2.putText(
                canvas,
                badge_text,
                (x + pad_x, badge_y2 - pad_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thick,
                cv2.LINE_AA,
            )

        # 3. Top-Right Status Card (Leaves left & center open for native Hikvision camera OSD)
        conn_text = "ONLINE" if is_connected else "RECONNECTING"
        conn_color = (0, 255, 0) if is_connected else (0, 0, 255)
        text_color_main = (255, 255, 255)

        line_1 = f"CAM: {camera_id} | {w}x{h} | FPS: {fps:.1f}"
        line_2 = f"TARGETS: {len(tracked_objects)} | RTSP: {conn_text}"

        font_scale_hud = 0.38
        font_thick_hud = 1

        size_1 = cv2.getTextSize(line_1, cv2.FONT_HERSHEY_SIMPLEX, font_scale_hud, font_thick_hud)[0]
        size_2 = cv2.getTextSize(line_2, cv2.FONT_HERSHEY_SIMPLEX, font_scale_hud, font_thick_hud)[0]

        card_w = max(size_1[0], size_2[0]) + 16
        card_h = 36
        card_x1 = max(0, w - card_w - 8)
        card_y1 = 6
        card_x2 = min(w, card_x1 + card_w)
        card_y2 = card_y1 + card_h

        # Semi-transparent dark card background only at top-right corner
        sub_img = canvas[card_y1:card_y2, card_x1:card_x2]
        dark_rect = np.zeros_like(sub_img, dtype=np.uint8)
        dark_rect[:] = (15, 15, 15)
        canvas[card_y1:card_y2, card_x1:card_x2] = cv2.addWeighted(sub_img, 0.30, dark_rect, 0.70, 0)
        cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (60, 60, 60), 1)

        # Draw 2 lines of status info
        cv2.putText(
            canvas,
            line_1,
            (card_x1 + 8, card_y1 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_hud,
            text_color_main,
            font_thick_hud,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line_2,
            (card_x1 + 8, card_y1 + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_hud,
            conn_color,
            font_thick_hud,
            cv2.LINE_AA,
        )

        return canvas
