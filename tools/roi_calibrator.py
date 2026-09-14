"""Interactive OpenCV Multi-Zone Polygon & Virtual Tripwire ROI Calibrator.

Allows operators to define dynamic multi-zone polygons (N >= 1) and virtual tripwire
lines (Line Crossing) for camera workspaces, and exports backward-compatible structured
JSON directly to cameras/<camera_id>/roi_zones.json.
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure workspace root is in sys.path
_WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_DIR))

from engine.config_loader import load_camera_config
from engine.zone_filter import load_roi_data

# Force TCP transport for RTSP streaming
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

# High-visibility distinct color palette for 9 zones
ZONE_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0),     # Zone 1: Bright Lime Green
    (0, 215, 255),   # Zone 2: Amber / Gold
    (255, 255, 0),   # Zone 3: Cyan / Sky Blue
    (255, 0, 255),   # Zone 4: Magenta / Fuchsia
    (255, 180, 0),   # Zone 5: Deep Teal / Ocean Blue
    (0, 165, 255),   # Zone 6: Vivid Orange
    (180, 105, 255), # Zone 7: Violet / Hot Pink
    (50, 205, 50),   # Zone 8: Emerald Green
    (230, 216, 173), # Zone 9: Light Blue / Ice
]

COLOR_TRIPWIRE_INACTIVE: Tuple[int, int, int] = (0, 80, 255)   # Neon Red / Orange
COLOR_TRIPWIRE_ACTIVE: Tuple[int, int, int] = (0, 255, 255)     # Electric Yellow
COLOR_PREVIEW: Tuple[int, int, int] = (200, 200, 200)          # Light Gray Guide
COLOR_HUD_BG: Tuple[int, int, int] = (20, 20, 20)


class ROICalibrator:
    """Interactive multi-zone polygon & virtual tripwire calibrator using OpenCV highgui."""

    def __init__(
        self,
        camera_id: str,
        image_path: Optional[Path] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        cam_dir: Optional[Path] = None,
        interactive: bool = True,
        frame_override: Optional[np.ndarray] = None,
    ) -> None:
        self.camera_id: str = camera_id
        self.image_path: Optional[Path] = image_path
        self.target_resolution: Optional[Tuple[int, int]] = target_resolution
        self.interactive: bool = interactive

        self.base_dir: Path = Path(__file__).resolve().parent.parent
        self.cam_dir: Path = Path(cam_dir) if cam_dir is not None else (self.base_dir / "cameras" / camera_id)
        self.config_path: Path = self.cam_dir / "config.json"
        self.roi_path: Path = self.cam_dir / "roi_zones.json"

        # Mode: 'zone' (polygons) or 'line' (tripwires)
        self.mode: str = "zone"
        self.zones: Dict[str, List[List[int]]] = {}
        self.lines: Dict[str, dict] = {}
        self.active_zone_key: str = "zone_1_koridor"
        self.active_line_key: str = "line_1"

        self.mouse_pos: Optional[Tuple[int, int]] = None
        self.window_name: str = f"ROI Calibrator - {self.camera_id}"

        if self.interactive and self.cam_dir.exists():
            self._load_camera_config()

        if frame_override is not None:
            self.base_frame = frame_override
        elif self.interactive and (self.cam_dir.exists() or self.image_path):
            self.base_frame = self._acquire_reference_frame()
        else:
            w = target_resolution[0] if target_resolution else 1920
            h = target_resolution[1] if target_resolution else 1080
            self.base_frame = np.zeros((h, w, 3), dtype=np.uint8)

        self.base_h, self.base_w = self.base_frame.shape[:2]
        self.current_win_w: int = self.base_w
        self.current_win_h: int = self.base_h

        if self.interactive and self.roi_path.exists():
            self._load_existing_roi_data()
        else:
            self.zones = {"zone_1": [], "zone_2": []}
            self.lines = {
                "line_1": {
                    "name": "Tripwire 1",
                    "p1": [],
                    "p2": [],
                    "direction": "both",
                    "target_classes": ["person", "car", "bus", "truck"],
                }
            }
            self.active_zone_key = "zone_1"
            self.active_line_key = "line_1"

    @property
    def active_target(self) -> str:
        """Return identifier of the currently selected zone or tripwire."""
        return self.active_zone_key if self.mode == "zone" else self.active_line_key

    def _load_camera_config(self) -> None:
        """Load camera configuration to determine target resolution and stream source."""
        if not self.cam_dir.exists():
            print(f"[ERROR] Camera directory does not exist: {self.cam_dir}")
            sys.exit(1)

        self.config: dict = {}
        if self.config_path.exists():
            try:
                self.config = load_camera_config(self.config_path)
                configured_res = self.config.get("target_resolution")
                if self.target_resolution is None and configured_res and len(configured_res) == 2:
                    self.target_resolution = (int(configured_res[0]), int(configured_res[1]))
            except Exception as e:
                print(f"[WARN] Failed to load {self.config_path}: {e}")

    def _load_existing_roi_data(self) -> None:
        """Load existing zones and tripwire lines with backward-compatible scaling."""
        if not self.roi_path.exists():
            self.zones = {
                "zone_1_koridor": [],
                "zone_2_transit": [],
            }
            return

        loaded_base, loaded_zones, loaded_lines = load_roi_data(self.roi_path)

        # Compute scaling factor if loaded base resolution differs from current calibrator frame
        sx = self.base_w / float(loaded_base[0]) if loaded_base[0] > 0 else 1.0
        sy = self.base_h / float(loaded_base[1]) if loaded_base[1] > 0 else 1.0
        needs_scale = (abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01)

        # 1. Populate and scale zones
        for k, pts in loaded_zones.items():
            if needs_scale:
                scaled_pts = [[int(round(pt[0] * sx)), int(round(pt[1] * sy))] for pt in pts if len(pt) >= 2]
                self.zones[k] = scaled_pts
                print(f"[INFO] Auto-scaled zone '{k}' from {loaded_base[0]}x{loaded_base[1]} to {self.base_w}x{self.base_h}")
            else:
                self.zones[k] = pts

        if not self.zones:
            self.zones = {
                "zone_1_koridor": [],
                "zone_2_transit": [],
            }
        self.active_zone_key = list(self.zones.keys())[0]

        # 2. Populate and scale tripwire lines
        for l_id, l_data in loaded_lines.items():
            line_copy = dict(l_data)
            if needs_scale:
                if "p1" in line_copy and len(line_copy["p1"]) >= 2:
                    line_copy["p1"] = [int(round(line_copy["p1"][0] * sx)), int(round(line_copy["p1"][1] * sy))]
                if "p2" in line_copy and len(line_copy["p2"]) >= 2:
                    line_copy["p2"] = [int(round(line_copy["p2"][0] * sx)), int(round(line_copy["p2"][1] * sy))]
                print(f"[INFO] Auto-scaled tripwire '{l_id}' from {loaded_base[0]}x{loaded_base[1]} to {self.base_w}x{self.base_h}")
            self.lines[l_id] = line_copy

        if self.lines:
            self.active_line_key = list(self.lines.keys())[0]

        print(f"[INFO] Loaded existing ROI configurations from {self.roi_path}: "
              f"{len(self.zones)} zones, {len(self.lines)} tripwires.")

    # Backward compatibility alias
    _load_existing_zones = _load_existing_roi_data

    def _acquire_reference_frame(self) -> np.ndarray:
        """Fetch reference frame from image file or camera stream/webcam."""
        frame: Optional[np.ndarray] = None

        if self.image_path and self.image_path.exists():
            frame = cv2.imread(str(self.image_path))
            if frame is None:
                print(f"[ERROR] Could not decode image at: {self.image_path}")
                sys.exit(1)
        else:
            source: Union[int, str] = self.config.get("source", 0)
            target_w = self.target_resolution[0] if self.target_resolution else 1920
            target_h = self.target_resolution[1] if self.target_resolution else 1080

            def _build_fallback_canvas(reason_msg: str) -> np.ndarray:
                fb = np.full((target_h, target_w, 3), (25, 18, 14), dtype=np.uint8)
                for gx in range(0, target_w, 160):
                    cv2.line(fb, (gx, 0), (gx, target_h), (40, 30, 24), 1)
                for gy in range(0, target_h, 90):
                    cv2.line(fb, (0, gy), (target_w, gy), (40, 30, 24), 1)

                cv2.putText(fb, f"Camera [{self.camera_id}] Feed Offline / Auth Pending",
                            (100, target_h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 2, cv2.LINE_AA)
                cv2.putText(fb, reason_msg, (100, target_h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(fb, f"Kanvas Kalibrasi Aktif ({target_w}x{target_h}). Anda tetap dapat menandai zona.",
                            (100, target_h // 2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 180), 1, cv2.LINE_AA)
                return fb

            print(f"[INFO] Connecting to video source: {source}...")
            backend = cv2.CAP_FFMPEG if isinstance(source, str) and source.startswith("rtsp://") else cv2.CAP_ANY
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                print(f"[WARN] Cannot open video source: {source}. Fallback to reference calibration canvas.")
                frame = _build_fallback_canvas("Stream RTSP offline atau autentikasi belum berhasil.")
            else:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, captured = cap.read()
                cap.release()
                if not ret or captured is None:
                    print(f"[WARN] Failed to decode frame from capture device. Fallback to calibration canvas.")
                    frame = _build_fallback_canvas("Terkoneksi ke stream, namun decoding frame gagal.")
                else:
                    frame = captured

        # Apply target resolution resizing if configured
        if self.target_resolution is not None:
            w, h = self.target_resolution
            if frame.shape[1] != w or frame.shape[0] != h:
                print(f"[INFO] Resizing canvas from {frame.shape[1]}x{frame.shape[0]} to {w}x{h}")
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        return frame

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: Optional[object]) -> None:
        """Handle mouse clicks and cursor position updates with coordinate scaling."""
        scale_x = self.base_w / self.current_win_w if self.current_win_w > 0 else 1.0
        scale_y = self.base_h / self.current_win_h if self.current_win_h > 0 else 1.0
        native_x = max(0, min(self.base_w - 1, int(round(x * scale_x))))
        native_y = max(0, min(self.base_h - 1, int(round(y * scale_y))))

        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (native_x, native_y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "zone":
                # Add polygon vertex
                if self.active_zone_key not in self.zones:
                    self.zones[self.active_zone_key] = []
                self.zones[self.active_zone_key].append([native_x, native_y])
                print(f"[{self.active_zone_key}] Added vertex: ({native_x}, {native_y}) "
                      f"(Total: {len(self.zones[self.active_zone_key])} pts)")
            elif self.mode == "line":
                # 2-click tripwire line placement
                if self.active_line_key not in self.lines:
                    self.lines[self.active_line_key] = {
                        "name": f"Tripwire {len(self.lines) + 1}",
                        "p1": [],
                        "p2": [],
                        "direction": "both",
                        "target_classes": ["person", "car", "bus", "truck"],
                    }
                cur_line = self.lines[self.active_line_key]
                if not cur_line.get("p1") or (cur_line.get("p1") and cur_line.get("p2")):
                    # Click 1: set P1
                    cur_line["p1"] = [native_x, native_y]
                    cur_line["p2"] = []
                    print(f"[{self.active_line_key}] P1 set to: ({native_x}, {native_y}). Klik titik kedua untuk P2.")
                else:
                    # Click 2: set P2
                    cur_line["p2"] = [native_x, native_y]
                    print(f"[{self.active_line_key}] P2 set to: ({native_x}, {native_y}). "
                          f"Tripwire lengkap: P1={cur_line['p1']} -> P2={cur_line['p2']} (Arah: {cur_line.get('direction', 'both')})")

    def _draw_hud(self, canvas: np.ndarray) -> None:
        """Render HUD status and keyboard shortcut overlay."""
        h, w = canvas.shape[:2]
        hud_height: int = 42
        hud_bar: np.ndarray = np.zeros((hud_height, w, 3), dtype=np.uint8)
        hud_bar[:] = COLOR_HUD_BG
        canvas[0:hud_height, 0:w] = cv2.addWeighted(canvas[0:hud_height, 0:w], 0.25, hud_bar, 0.75, 0)

        # Status text
        if self.mode == "zone":
            zone_keys = list(self.zones.keys())
            active_idx = zone_keys.index(self.active_zone_key) if self.active_zone_key in zone_keys else 0
            cur_pts = len(self.zones.get(self.active_zone_key, []))
            mode_color = ZONE_COLORS[active_idx % len(ZONE_COLORS)]
            status_line = (
                f"Cam: {self.camera_id} ({w}x{h}) | MODE: [POLYGON ZONES] | "
                f"Active: #{active_idx + 1} '{self.active_zone_key}' ({cur_pts} pts) | Total Zones: {len(self.zones)}"
            )
        else:
            line_keys = list(self.lines.keys())
            active_idx = line_keys.index(self.active_line_key) if self.active_line_key in line_keys else 0
            cur_l = self.lines.get(self.active_line_key, {})
            dir_str = cur_l.get("direction", "both")
            has_p1 = bool(cur_l.get("p1"))
            has_p2 = bool(cur_l.get("p2"))
            pt_state = "P1 & P2 OK" if (has_p1 and has_p2) else ("Waiting P2" if has_p1 else "Waiting P1")
            mode_color = COLOR_TRIPWIRE_ACTIVE
            status_line = (
                f"Cam: {self.camera_id} ({w}x{h}) | MODE: [TRIPWIRE LINES] | "
                f"Active: #{active_idx + 1} '{self.active_line_key}' ({pt_state}, Dir: {dir_str}) | Total Lines: {len(self.lines)}"
            )

        cmd_line = "[1-9] Select | [N] New Zone | [L] Toggle Tripwire | [D] Direction | [U/Z] Undo | [C] Clear | [S] Save | [Q] Quit"

        cv2.putText(canvas, status_line, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, mode_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, cmd_line, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_zones(self, canvas: np.ndarray) -> None:
        """Render all polygon zones with ultra-thin crisp outlines via alpha blending."""
        overlay = canvas.copy()
        zone_keys = list(self.zones.keys())

        for idx, z_key in enumerate(zone_keys):
            pts: List[List[int]] = self.zones[z_key]
            if not pts:
                continue

            color = ZONE_COLORS[idx % len(ZONE_COLORS)]
            is_active = (self.mode == "zone" and z_key == self.active_zone_key)
            line_thick = 1  # Ultra-thin crisp line

            np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

            # Draw polygon perimeter outline (no floor fillPoly)
            is_closed = len(pts) >= 3
            cv2.polylines(overlay, [np_pts], is_closed, color, line_thick, cv2.LINE_AA)

            # Draw refined vertex markers (compact dots: radius=2, thickness=1, no text)
            for pt in pts:
                cv2.circle(overlay, (pt[0], pt[1]), 2, color, 1, cv2.LINE_AA)

        # Blend semi-transparent outline (~0.5px optical effect: alpha=0.35 for inactive, 0.70 for active)
        active_has_pts = (self.mode == "zone" and bool(self.zones.get(self.active_zone_key)))
        blend_alpha = 0.70 if active_has_pts else 0.35
        cv2.addWeighted(overlay, blend_alpha, canvas, 1.0 - blend_alpha, 0, canvas)

        # Draw interactive cursor guide line for active zone
        if self.mode == "zone" and self.active_zone_key in self.zones:
            active_pts = self.zones[self.active_zone_key]
            if active_pts and self.mouse_pos:
                last_pt = active_pts[-1]
                active_color = ZONE_COLORS[zone_keys.index(self.active_zone_key) % len(ZONE_COLORS)]
                cv2.line(canvas, (last_pt[0], last_pt[1]), self.mouse_pos, active_color, 1, cv2.LINE_AA)

    def _draw_tripwires(self, canvas: np.ndarray) -> None:
        """Render all tripwire lines with ultra-thin crisp lines and direction arrows (no floating text clutter)."""
        line_keys = list(self.lines.keys())
        if not line_keys:
            return

        overlay = canvas.copy()

        for idx, l_key in enumerate(line_keys):
            l_data = self.lines[l_key]
            p1 = l_data.get("p1")
            p2 = l_data.get("p2")
            is_active = (self.mode == "line" and l_key == self.active_line_key)
            color = COLOR_TRIPWIRE_ACTIVE if is_active else COLOR_TRIPWIRE_INACTIVE

            if p1 and len(p1) >= 2:
                # Draw P1 compact dot: radius=2, thickness=1, no floating text
                cv2.circle(overlay, (p1[0], p1[1]), 2, (0, 255, 255), 1, cv2.LINE_AA)

            if p1 and p2 and len(p1) >= 2 and len(p2) >= 2:
                # Draw P2 compact dot: radius=2, thickness=1, no floating text
                cv2.circle(overlay, (p2[0], p2[1]), 2, (0, 165, 255), 1, cv2.LINE_AA)

                # Draw ultra-thin tripwire line (thickness=1)
                cv2.line(overlay, (p1[0], p1[1]), (p2[0], p2[1]), color, 1, cv2.LINE_AA)

                # Compute midpoint and direction vector for arrow
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2

                dx = float(p2[0] - p1[0])
                dy = float(p2[1] - p1[1])
                length = float(np.hypot(dx, dy))

                dir_mode = str(l_data.get("direction", "both")).lower()

                # Direction arrows (thin stroke)
                if length > 20:
                    nx = -dy / length
                    ny = dx / length
                    arrow_len = 12.0

                    if dir_mode in ("both", "a_to_b"):
                        tip_x = int(round(mid_x + nx * arrow_len))
                        tip_y = int(round(mid_y + ny * arrow_len))
                        cv2.arrowedLine(overlay, (mid_x, mid_y), (tip_x, tip_y), color, 1, cv2.LINE_AA, 0, 0.35)

                    if dir_mode in ("both", "b_to_a"):
                        tip_x2 = int(round(mid_x - nx * arrow_len))
                        tip_y2 = int(round(mid_y - ny * arrow_len))
                        cv2.arrowedLine(overlay, (mid_x, mid_y), (tip_x2, tip_y2), color, 1, cv2.LINE_AA, 0, 0.35)

            elif p1 and len(p1) >= 2 and self.mode == "line" and is_active and self.mouse_pos:
                # Interactive cursor guide line to mouse position
                cv2.line(canvas, (p1[0], p1[1]), self.mouse_pos, COLOR_PREVIEW, 1, cv2.LINE_AA)

        # Blend tripwire layer (~0.5px optical effect: alpha=0.35 for inactive, 0.70 for active)
        blend_alpha = 0.70 if (self.mode == "line") else 0.35
        cv2.addWeighted(overlay, blend_alpha, canvas, 1.0 - blend_alpha, 0, canvas)

    def save_zones(self) -> None:
        """Write polygon coordinates and tripwire lines to camera roi_zones.json."""
        out_data = {
            "base_resolution": [self.base_w, self.base_h],
            "zones": self.zones,
            "lines": self.lines,
        }
        # Top-level mirroring for 100% legacy reader backward compatibility
        for zk, zv in self.zones.items():
            if zk not in out_data:
                out_data[zk] = zv

        with open(self.roi_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\n[SUCCESS] Saved ROI configurations (base: {self.base_w}x{self.base_h}) to: {self.roi_path}")
        print(f" - {len(self.zones)} Zones: {list(self.zones.keys())}")
        print(f" - {len(self.lines)} Tripwires: {list(self.lines.keys())}\n")

    on_mouse = _mouse_callback

    def render(self, canvas: Optional[np.ndarray] = None) -> np.ndarray:
        """Render complete calibration preview including zones, tripwires, and HUD."""
        out = canvas.copy() if canvas is not None else self.base_frame.copy()
        self._draw_zones(out)
        self._draw_tripwires(out)
        self._draw_hud(out)
        return out

    def handle_key(self, key: int) -> bool:
        """Process keyboard input. Returns False if requested to quit, else True."""
        if key in (ord("q"), 27):  # 'q' or ESC
            print("[INFO] Exiting calibrator.")
            return False

        # Number keys '1' through '9'
        elif ord("1") <= key <= ord("9"):
            target_idx = key - ord("1")
            if self.mode == "zone":
                zone_keys = list(self.zones.keys())
                if target_idx < len(zone_keys):
                    self.active_zone_key = zone_keys[target_idx]
                else:
                    new_key = f"zone_{target_idx + 1}"
                    self.zones[new_key] = []
                    self.active_zone_key = new_key
                print(f"[STATUS] Switched to Zone #{target_idx + 1}: '{self.active_zone_key}'")
            else:
                line_keys = list(self.lines.keys())
                if target_idx < len(line_keys):
                    self.active_line_key = line_keys[target_idx]
                else:
                    new_line_key = f"line_{target_idx + 1}"
                    self.lines[new_line_key] = {
                        "name": f"Tripwire {target_idx + 1}",
                        "p1": [],
                        "p2": [],
                        "direction": "both",
                        "target_classes": ["person", "car", "bus", "truck"],
                    }
                    self.active_line_key = new_line_key
                print(f"[STATUS] Switched to Tripwire #{target_idx + 1}: '{self.active_line_key}'")

        # 'n': Create New Zone
        elif key == ord("n"):
            self.mode = "zone"
            new_idx = len(self.zones) + 1
            new_zkey = f"zone_{new_idx}"
            while new_zkey in self.zones:
                new_idx += 1
                new_zkey = f"zone_{new_idx}"
            self.zones[new_zkey] = []
            self.active_zone_key = new_zkey
            print(f"[STATUS] Created and switched to new Polygon Zone: '{self.active_zone_key}'")

        # 'l': Toggle Tripwire Mode
        elif key == ord("l"):
            if self.mode == "zone":
                self.mode = "line"
                if not self.lines:
                    self.lines["line_1"] = {
                        "name": "Tripwire 1",
                        "p1": [],
                        "p2": [],
                        "direction": "both",
                        "target_classes": ["person", "car", "bus", "truck"],
                    }
                self.active_line_key = list(self.lines.keys())[0]
                print(f"[STATUS] Switched to TRIPWIRE MODE. Active: '{self.active_line_key}'. "
                      f"Klik 2 titik untuk membuat garis virtual.")
            else:
                self.mode = "zone"
                print(f"[STATUS] Switched to POLYGON ZONE MODE. Active: '{self.active_zone_key}'.")

        # 'd': Cycle Line Direction
        elif key == ord("d"):
            if self.mode == "line" and self.active_line_key in self.lines:
                cur_l = self.lines[self.active_line_key]
                cur_dir = cur_l.get("direction", "both")
                cycle = {"both": "A_to_B", "A_to_B": "B_to_A", "B_to_A": "both"}
                new_dir = cycle.get(cur_dir, "both")
                cur_l["direction"] = new_dir
                print(f"[{self.active_line_key}] Direction changed to: {new_dir}")
            else:
                print("[INFO] Tekan 'L' untuk beralih ke Mode Tripwire sebelum mengubah arah.")

        # 'u' or 'z': Undo last point
        elif key in (ord("u"), ord("z")):
            if self.mode == "zone":
                if self.zones.get(self.active_zone_key):
                    removed = self.zones[self.active_zone_key].pop()
                    print(f"[{self.active_zone_key}] Removed vertex: {removed}")
            else:
                if self.active_line_key in self.lines:
                    cur_l = self.lines[self.active_line_key]
                    if cur_l.get("p2"):
                        cur_l["p2"] = []
                        print(f"[{self.active_line_key}] Removed P2. Klik untuk menentukan P2 kembali.")
                    elif cur_l.get("p1"):
                        cur_l["p1"] = []
                        print(f"[{self.active_line_key}] Removed P1. Klik untuk menentukan P1 kembali.")

        # 'c': Clear active zone/line
        elif key == ord("c"):
            if self.mode == "zone":
                if self.active_zone_key in self.zones:
                    self.zones[self.active_zone_key].clear()
                    print(f"[{self.active_zone_key}] Cleared all vertices.")
            else:
                if self.active_line_key in self.lines:
                    self.lines[self.active_line_key]["p1"] = []
                    self.lines[self.active_line_key]["p2"] = []
                    print(f"[{self.active_line_key}] Cleared line endpoints P1 and P2.")

        # 's': Save to JSON
        elif key == ord("s"):
            self.save_zones()

        return True

    def run(self) -> None:
        """Main calibration display loop with interactive keyboard shortcuts."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n=======================================================")
        print(f" ROI Calibrator Running for [{self.camera_id}]")
        print(" [1-9] Pilih Zona / Garis")
        print(" [N] Buat Zona Poligon Baru")
        print(" [L] Beralih ke Mode Tripwire (Garis Virtual)")
        print(" [D] Ganti Arah Tripwire (Both / A->B / B->A)")
        print(" [U / Z] Undo Titik Terakhir | [C] Clear Titik")
        print(" [S] Simpan ke JSON         | [Q / ESC] Keluar")
        print("=======================================================\n")

        while True:
            display_frame = self.render()

            # Auto-resize canvas to match actual window dimensions (eliminates gray bars)
            rect = cv2.getWindowImageRect(self.window_name)
            if rect and rect[2] > 50 and rect[3] > 50:
                self.current_win_w, self.current_win_h = rect[2], rect[3]
                if display_frame.shape[1] != self.current_win_w or display_frame.shape[0] != self.current_win_h:
                    display_frame = cv2.resize(
                        display_frame,
                        (self.current_win_w, self.current_win_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(20) & 0xFF
            if not self.handle_key(key):
                break

        cv2.destroyAllWindows()

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Interactive OpenCV Multi-Zone & Tripwire ROI Calibrator")
    parser.add_argument(
        "--cam",
        type=str,
        default="cam_01",
        help="Target camera ID workspace (e.g. cam_01)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional static reference image path instead of live capture",
    )
    parser.add_argument(
        "--res",
        type=str,
        default=None,
        help="Target resolution override formatted as WIDTHxHEIGHT (e.g. 1280x720)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    image_file: Optional[Path] = Path(args.image) if args.image else None

    target_res: Optional[Tuple[int, int]] = None
    if args.res:
        parts = args.res.lower().split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            target_res = (int(parts[0]), int(parts[1]))
        else:
            print(f"[ERROR] Invalid resolution format '{args.res}'. Use WIDTHxHEIGHT (e.g. 1280x720)")
            sys.exit(1)

    calibrator = ROICalibrator(
        camera_id=args.cam,
        image_path=image_file,
        target_resolution=target_res,
    )
    calibrator.run()
