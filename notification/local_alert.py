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
COLOR_FACE_OUTSIDE: Tuple[int, int, int] = (255, 200, 0)  # Cyan / Sky Blue (BGR: 255, 200, 0)
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
        
        # Audio file paths
        base_dir = Path(__file__).resolve().parent
        self.sound_file: Path = base_dir / "alert.wav"
        self.chime_file: Path = base_dir / "chime.wav"
        self._last_chime_time: float = 0.0
        self._chime_obj = None

        if not self.sound_file.exists():
            try:
                generate_fallback_wav(self.sound_file)
            except Exception as e:
                print(f"[WARN] Failed to generate fallback WAV: {e}")

        if not self.chime_file.exists():
            try:
                generate_fallback_wav(self.chime_file, frequency_hz=880.0, duration_sec=0.35)
            except Exception as e:
                print(f"[WARN] Failed to generate fallback chime WAV: {e}")

        # Initialize Pygame Mixer safely
        try:
            import pygame
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            if self.sound_file.exists():
                self._sound_obj = pygame.mixer.Sound(str(self.sound_file))
            if self.chime_file.exists():
                self._chime_obj = pygame.mixer.Sound(str(self.chime_file))
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

    def request_pre_alarm_chime(self) -> None:
        """Trigger gentle pre-alarm chime (85% dwell) with 20s debounce."""
        if not self._audio_available or self._chime_obj is None:
            return

        now = time.time()
        with self._playback_lock:
            import pygame
            if (now - self._last_chime_time) >= 20.0:
                if not pygame.mixer.get_busy():
                    self._chime_obj.play()
                    self._last_chime_time = now


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
        faces: Optional[List[Tuple[Tuple[int, int, int, int], float]]] = None,
        outside_faces: Optional[List[Tuple[Tuple[int, int, int, int], float]]] = None,
        zone_base_resolution: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Render complete CCTV visual indicators onto canvas."""
        h, w = canvas.shape[:2]
        now = time.time()
        blink_state = (int(now * 2) % 2) == 0  # 500ms toggle

        active_faces = faces if faces is not None else (outside_faces or [])

        # Dynamic Resolution Decoupling Scaling Ratios for Inference Objects (base 640x480)
        scale_x = w / 640.0
        scale_y = h / 480.0
        scale_factor = min(scale_x, scale_y)

        # Determine Zone Polygon Coordinate Base Resolution
        if zone_base_resolution is not None:
            zone_base_w, zone_base_h = zone_base_resolution
        elif "base_resolution" in zones and isinstance(zones["base_resolution"], (list, tuple)):
            zone_base_w, zone_base_h = zones["base_resolution"]
        else:
            # Auto-detect from coordinates: if any x > 640 or y > 480 -> 1080p, else 640p
            max_zx = 0
            max_zy = 0
            for k, v in zones.items():
                if isinstance(v, list) and not k.startswith("_") and k != "base_resolution":
                    for pt in v:
                        if len(pt) >= 2:
                            max_zx = max(max_zx, pt[0])
                            max_zy = max(max_zy, pt[1])
            if max_zx > 640 or max_zy > 480:
                zone_base_w, zone_base_h = (1920, 1080)
            else:
                zone_base_w, zone_base_h = (640, 480)

        zone_scale_x = w / float(zone_base_w) if zone_base_w > 0 else 1.0
        zone_scale_y = h / float(zone_base_h) if zone_base_h > 0 else 1.0

        # Scaled typography and stroke properties for crisp monitor rendering
        zone_thickness_safe = max(1, int(round(1.5 * scale_factor)))
        zone_thickness_violation = max(2, int(round(2.5 * scale_factor)))
        box_thickness_normal = 1
        box_thickness_alert = 2
        centroid_radius = max(2, int(round(3.0 * scale_factor)))
        
        bag_font_scale = 0.43
        bag_font_thick = 1
        pad_x = 4
        pad_y = 2

        # Identify which zones currently contain triggered violations (strictly for unattended bags)
        violated_zones = {
            obj.zone_id
            for obj in tracked_objects
            if getattr(obj, "is_triggered", False)
            and getattr(obj, "class_label", "") in ("tas", "backpack", "handbag", "suitcase")
        }

        # 1. Draw ROI Zones (Dynamically scaled from zone_base_resolution to canvas)
        for zone_id, pts in zones.items():
            if not isinstance(pts, list) or zone_id.startswith("_") or zone_id == "base_resolution":
                continue
            if len(pts) < 3:
                continue
            scaled_pts = [[int(round(pt[0] * zone_scale_x)), int(round(pt[1] * zone_scale_y))] for pt in pts]
            np_pts = np.array(scaled_pts, dtype=np.int32).reshape((-1, 1, 2))
            is_violated = zone_id in violated_zones

            if is_violated:
                zone_color = COLOR_VIOLATION if blink_state else COLOR_WARNING
                thickness = zone_thickness_violation
            else:
                zone_color = COLOR_SAFE
                thickness = zone_thickness_safe

            cv2.polylines(canvas, [np_pts], isClosed=True, color=zone_color, thickness=thickness, lineType=cv2.LINE_AA)

        # 2. Draw Bounding Boxes and Status Badges
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

            # VISUAL CLUTTER REDUCTION: Hide person bounding boxes & badges from screen.
            # Person detection, tracking, and owner proximity logic continue uninterrupted in background.
            if not is_bag:
                continue

            # Scale bag coordinates from 640x480 space to canvas
            sx = int(round(x * scale_x))
            sy = int(round(y * scale_y))
            sbw = int(round(bw * scale_x))
            sbh = int(round(bh * scale_y))
            scx = int(round(obj.centroid[0] * scale_x))
            scy = int(round(obj.centroid[1] * scale_y))

            is_attended = getattr(obj, "is_attended", False)
            dwell_max = float(getattr(obj, "dwell_threshold", 3600.0))
            max_mins = max(1, int(round(dwell_max / 60.0)))

            # User-friendly minute-based dwell time format (strictly no raw seconds)
            dwell_str = f"{dwell / 60.0:.1f}m/{max_mins}m"

            # OBJECT IN STERILE ZONE (ATTENDED / UNATTENDED / ALERT / PRE-ALARM / WARNING)
            is_alert = (getattr(obj, "is_triggered", False) or dwell >= dwell_max)
            is_pre_alarm = getattr(obj, "is_pre_alarm", False) or (dwell >= dwell_max * 0.85)
            is_warning = getattr(obj, "is_warning", False) or (dwell >= dwell_max * 0.50)
            is_occluded = getattr(obj, "is_occluded", False)

            owner_info = getattr(obj, "last_owner_info", None)
            owner_name = owner_info.get("name") if owner_info else None
            owner_suffix = f" [{owner_name}]" if owner_name and owner_name != "Unknown" else ""

            if is_attended:
                box_color = COLOR_SAFE
                badge_text = f"ID {track_id} | Objek [AMAN]{owner_suffix}"
            elif is_occluded:
                box_color = (0, 215, 255)  # Amber / Soft Gold
                badge_text = f"ID {track_id} | Objek [TERTUTUP] ({dwell_str}){owner_suffix}"
            elif is_alert:
                box_color = (0, 0, 255) if blink_state else (0, 165, 255)
                badge_text = f"[ALERT] CLEAR AREA ({dwell / 60.0:.0f}m){owner_suffix}"
            elif is_pre_alarm:
                box_color = (0, 100, 255)  # Deep Orange-Red
                badge_text = f"ID {track_id} | Objek [PRE-ALARM 85%] ({dwell_str}){owner_suffix}"
            elif is_warning:
                box_color = (0, 165, 255)  # Amber / Orange
                badge_text = f"ID {track_id} | Objek [WARNING 50%] ({dwell_str}){owner_suffix}"
            elif getattr(obj, "is_stationary", False) or is_bag:
                if dwell <= (dwell_max * 0.5):
                    box_color = (0, 255, 255)  # Kuning
                else:
                    box_color = (0, 165, 255)  # Oranye Warning
                badge_text = f"ID {track_id} | Objek ({dwell_str}){owner_suffix}"
            else:
                box_color = COLOR_SAFE
                badge_text = f"ID {track_id} | Objek{owner_suffix}"

            # Draw crisp thin bounding rectangle and centroid (1px normal, max 2px alert)
            cur_box_thick = box_thickness_alert if is_alert else box_thickness_normal
            cv2.rectangle(canvas, (sx, sy), (sx + sbw, sy + sbh), box_color, cur_box_thick, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (scx, scy), centroid_radius, box_color, -1, lineType=cv2.LINE_AA)

            # Compact badge background tightly touching top edge of bounding box
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, bag_font_scale, bag_font_thick)
            badge_h = text_h + (pad_y * 2) + 1
            badge_w = text_w + (pad_x * 2)
            if sy - badge_h >= 0:
                badge_y1 = sy - badge_h
                badge_y2 = sy
            else:
                badge_y1 = sy
                badge_y2 = min(h, sy + badge_h)
            badge_x1 = max(0, min(sx, w - badge_w))
            badge_x2 = min(w, badge_x1 + badge_w)

            # Semi-transparent badge background (65% tint, 35% frame)
            sub_badge = canvas[badge_y1:badge_y2, badge_x1:badge_x2]
            if sub_badge.size > 0:
                color_rect = np.full_like(sub_badge, box_color, dtype=np.uint8)
                cv2.addWeighted(color_rect, 0.65, sub_badge, 0.35, 0, sub_badge)
            
            # High-contrast text color: white on red, black on green/yellow/orange
            is_dark_bg = box_color == (0, 0, 255)
            text_color = (255, 255, 255) if is_dark_bg else (0, 0, 0)

            cv2.putText(
                canvas,
                badge_text,
                (badge_x1 + pad_x, badge_y2 - pad_y - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                bag_font_scale,
                text_color,
                bag_font_thick,
                cv2.LINE_AA,
            )

        # 2a. Draw Faces (Cyan BGR: 255, 200, 0) across all areas
        if active_faces:
            face_font_scale = 0.42
            face_font_thick = 1
            face_pad_x = 4
            face_pad_y = 2
            face_box_thick = 1

            for item in active_faces:
                if len(item) >= 3:
                    (fx, fy, fw_f, fh_f), f_score, face_label = item[0], item[1], item[2]
                    if isinstance(face_label, str) and face_label:
                        f_badge = face_label
                    else:
                        f_badge = f"Face {f_score:.2f}"
                elif len(item) == 2:
                    (fx, fy, fw_f, fh_f), f_score_or_label = item[0], item[1]
                    if isinstance(f_score_or_label, str):
                        f_badge = f_score_or_label
                    else:
                        f_badge = f"Face {f_score_or_label:.2f}"
                else:
                    continue

                if fx > 640 or fy > 360:
                    sfx = int(round(fx))
                    sfy = int(round(fy))
                    sfw = int(round(fw_f))
                    sfh = int(round(fh_f))
                else:
                    face_scale_x = w / 640.0
                    face_scale_y = h / 360.0
                    sfx = int(round(fx * face_scale_x))
                    sfy = int(round(fy * face_scale_y))
                    sfw = int(round(fw_f * face_scale_x))
                    sfh = int(round(fh_f * face_scale_y))

                # Crisp 1px bounding box
                cv2.rectangle(canvas, (sfx, sfy), (sfx + sfw, sfy + sfh), COLOR_FACE_OUTSIDE, face_box_thick, lineType=cv2.LINE_AA)
                (f_tw, f_th), _ = cv2.getTextSize(f_badge, cv2.FONT_HERSHEY_SIMPLEX, face_font_scale, face_font_thick)
                f_badge_h = f_th + (face_pad_y * 2) + 1
                f_badge_w = f_tw + (face_pad_x * 2)
                if sfy - f_badge_h >= 0:
                    f_by1 = sfy - f_badge_h
                    f_by2 = sfy
                else:
                    f_by1 = sfy
                    f_by2 = min(h, sfy + f_badge_h)
                f_bx1 = max(0, min(sfx, w - f_badge_w))
                f_bx2 = min(w, f_bx1 + f_badge_w)

                # Semi-transparent badge background (65% tint, 35% frame)
                sub_f = canvas[f_by1:f_by2, f_bx1:f_bx2]
                if sub_f.size > 0:
                    color_rect = np.full_like(sub_f, COLOR_FACE_OUTSIDE, dtype=np.uint8)
                    cv2.addWeighted(color_rect, 0.65, sub_f, 0.35, 0, sub_f)

                cv2.putText(
                    canvas,
                    f_badge,
                    (f_bx1 + face_pad_x, f_by2 - face_pad_y - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    face_font_scale,
                    (0, 0, 0),
                    face_font_thick,
                    cv2.LINE_AA,
                )

        # 3. Top-Right Status Card (Leaves left & center open for native Hikvision camera OSD)
        conn_text = "ONLINE" if is_connected else "RECONNECTING"
        conn_color = (0, 255, 0) if is_connected else (0, 0, 255)
        text_color_main = (255, 255, 255)
        bag_count = len([
            o for o in tracked_objects
            if getattr(o, "class_label", "") in ("tas", "backpack", "handbag", "suitcase")
            and getattr(o, "zone_id", None) in zones
            and (not hasattr(o, "should_render") or o.should_render)
        ])
        face_count = len(active_faces)

        line_1 = f"CAM: {camera_id} | {w}x{h} | FPS: {fps:.1f}"
        line_2 = f"CLEAR AREA: {bag_count} | FACES: {face_count} | RTSP: {conn_text}"

        font_scale_hud = 0.38 * scale_factor
        font_thick_hud = max(1, int(round(1.2 * scale_factor)))

        size_1, _ = cv2.getTextSize(line_1, cv2.FONT_HERSHEY_SIMPLEX, font_scale_hud, font_thick_hud)
        size_2, _ = cv2.getTextSize(line_2, cv2.FONT_HERSHEY_SIMPLEX, font_scale_hud, font_thick_hud)

        pad_card_x = int(round(12.0 * scale_factor))
        pad_card_y = int(round(8.0 * scale_factor))
        line_spacing = int(round(6.0 * scale_factor))

        text_max_w = max(size_1[0], size_2[0])
        card_w = text_max_w + (pad_card_x * 2)
        card_h = size_1[1] + size_2[1] + (pad_card_y * 2) + line_spacing

        margin_right = int(round(10.0 * scale_factor))
        margin_top = int(round(8.0 * scale_factor))
        card_x1 = max(0, w - card_w - margin_right)
        card_y1 = margin_top
        card_x2 = min(w, card_x1 + card_w)
        card_y2 = card_y1 + card_h

        # Semi-transparent dark card background only at top-right corner (zero-allocation in-place darkening)
        sub_img = canvas[card_y1:card_y2, card_x1:card_x2]
        if sub_img.size > 0:
            cv2.convertScaleAbs(sub_img, sub_img, alpha=0.30, beta=10)
            cv2.rectangle(canvas, (card_x1, card_y1), (card_x2, card_y2), (60, 60, 60), max(1, int(round(1.0 * scale_factor))))

        # Draw 2 lines of status info
        line1_y = card_y1 + pad_card_y + size_1[1]
        line2_y = line1_y + line_spacing + size_2[1]

        cv2.putText(
            canvas,
            line_1,
            (card_x1 + pad_card_x, line1_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_hud,
            text_color_main,
            font_thick_hud,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line_2,
            (card_x1 + pad_card_x, line2_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_hud,
            conn_color,
            font_thick_hud,
            cv2.LINE_AA,
        )

        return canvas
