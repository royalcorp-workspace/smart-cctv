"""Interactive OpenCV Polygon ROI Calibrator.

Allows operators to define multi-zone polygons for camera workspaces and export
coordinates directly to cameras/<camera_id>/roi_zones.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure workspace root is in sys.path
_WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_DIR))

from engine.config_loader import load_camera_config

import os
# Force TCP transport for RTSP streaming
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

COLOR_ZONE_1: Tuple[int, int, int] = (0, 255, 0)      # Green
COLOR_ZONE_2: Tuple[int, int, int] = (0, 215, 255)    # Gold / Amber
COLOR_PREVIEW: Tuple[int, int, int] = (200, 200, 200) # Gray / White guide
COLOR_HUD_BG: Tuple[int, int, int] = (20, 20, 20)


class ROICalibrator:
    """Interactive polygon ROI calibrator using OpenCV highgui."""

    def __init__(
        self,
        camera_id: str,
        image_path: Optional[Path] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
        cam_dir: Optional[Path] = None,
    ) -> None:
        self.camera_id: str = camera_id
        self.image_path: Optional[Path] = image_path
        self.target_resolution: Optional[Tuple[int, int]] = target_resolution
        
        self.base_dir: Path = Path(__file__).resolve().parent.parent
        self.cam_dir: Path = Path(cam_dir) if cam_dir is not None else (self.base_dir / "cameras" / camera_id)
        self.config_path: Path = self.cam_dir / "config.json"
        self.roi_path: Path = self.cam_dir / "roi_zones.json"

        self.zones: Dict[str, List[List[int]]] = {
            "zone_1_koridor": [],
            "zone_2_transit": [],
        }
        self.active_zone_key: str = "zone_1_koridor"
        self.mouse_pos: Optional[Tuple[int, int]] = None
        self.window_name: str = f"ROI Calibrator - {self.camera_id}"
        
        self._load_camera_config()
        self.base_frame: np.ndarray = self._acquire_reference_frame()
        self.base_h, self.base_w = self.base_frame.shape[:2]
        self.current_win_w: int = self.base_w
        self.current_win_h: int = self.base_h

        self._load_existing_zones()

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

    def _load_existing_zones(self) -> None:
        """Load existing polygon points and upscale/downscale to current base_frame resolution if needed."""
        if not self.roi_path.exists():
            return

        try:
            with open(self.roi_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)

            if not isinstance(loaded_data, dict):
                return

            # Determine loaded resolution
            if "base_resolution" in loaded_data and isinstance(loaded_data["base_resolution"], (list, tuple)):
                loaded_base = (int(loaded_data["base_resolution"][0]), int(loaded_data["base_resolution"][1]))
            else:
                # Heuristic: check coordinate bounds
                max_x = 0
                max_y = 0
                for k, v in loaded_data.items():
                    if isinstance(v, list) and not k.startswith("_") and k != "base_resolution":
                        for pt in v:
                            if len(pt) >= 2:
                                max_x = max(max_x, pt[0])
                                max_y = max(max_y, pt[1])
                loaded_base = (1920, 1080) if (max_x > 640 or max_y > 480) else (640, 480)

            # Compute scaling factor if loaded base resolution differs from current calibrator frame
            sx = self.base_w / float(loaded_base[0]) if loaded_base[0] > 0 else 1.0
            sy = self.base_h / float(loaded_base[1]) if loaded_base[1] > 0 else 1.0

            for k, v in loaded_data.items():
                if isinstance(v, list) and not k.startswith("_") and k != "base_resolution":
                    if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
                        scaled_pts = [[int(round(pt[0] * sx)), int(round(pt[1] * sy))] for pt in v]
                        self.zones[k] = scaled_pts
                        print(f"[INFO] Auto-scaled existing zone '{k}' from {loaded_base[0]}x{loaded_base[1]} to {self.base_w}x{self.base_h}")
                    else:
                        self.zones[k] = v

            print(f"[INFO] Loaded existing ROI coordinates from {self.roi_path}")
        except Exception as e:
            print(f"[WARN] Failed to parse existing {self.roi_path}: {e}")

    def _acquire_reference_frame(self) -> np.ndarray:
        """Fetch frame from image file or camera stream/webcam."""
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
                # Draw grid lines
                for gx in range(0, target_w, 160):
                    cv2.line(fb, (gx, 0), (gx, target_h), (40, 30, 24), 1)
                for gy in range(0, target_h, 90):
                    cv2.line(fb, (0, gy), (target_w, gy), (40, 30, 24), 1)

                cv2.putText(
                    fb,
                    f"Camera [{self.camera_id}] Feed Offline / Auth Pending",
                    (100, target_h // 2 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 215, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    fb,
                    reason_msg,
                    (100, target_h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    fb,
                    f"Kanvas Kalibrasi 1080p Aktif ({target_w}x{target_h}). Anda tetap dapat menandai zona.",
                    (100, target_h // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 180),
                    1,
                    cv2.LINE_AA,
                )
                return fb

            print(f"[INFO] Connecting to video source: {source}...")
            backend = cv2.CAP_FFMPEG if isinstance(source, str) and source.startswith("rtsp://") else cv2.CAP_ANY
            cap = cv2.VideoCapture(source, backend)
            if not cap.isOpened():
                print(f"[WARN] Cannot open video source: {source}. Fallback to 1080p calibration canvas.")
                frame = _build_fallback_canvas("Stream RTSP offline atau autentikasi belum berhasil.")
            else:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, captured = cap.read()
                cap.release()
                if not ret or captured is None:
                    print(f"[WARN] Failed to decode frame from capture device. Fallback to 1080p calibration canvas.")
                    frame = _build_fallback_canvas("Terkoneksi ke stream, namun decoding frame gagal.")
                else:
                    frame = captured

        # Apply target resolution resizing to maintain coordinate consistency
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
            self.zones[self.active_zone_key].append([native_x, native_y])
            print(f"[{self.active_zone_key}] Added point: ({native_x}, {native_y})")

    def _draw_hud(self, canvas: np.ndarray) -> None:
        """Render HUD status and shortcuts overlay."""
        h, w = canvas.shape[:2]
        hud_height: int = 40
        hud_bar: np.ndarray = np.zeros((hud_height, w, 3), dtype=np.uint8)
        hud_bar[:] = COLOR_HUD_BG
        canvas[0:hud_height, 0:w] = cv2.addWeighted(canvas[0:hud_height, 0:w], 0.3, hud_bar, 0.7, 0)

        active_color = COLOR_ZONE_1 if self.active_zone_key == "zone_1_koridor" else COLOR_ZONE_2
        status_line = (
            f"Cam: {self.camera_id} ({w}x{h}) | Active: {self.active_zone_key} "
            f"| Z1: {len(self.zones['zone_1_koridor'])} pts | Z2: {len(self.zones['zone_2_transit'])} pts"
        )
        cmd_line = "[1] Z1  [2] Z2  [U/Z] Undo  [C] Clear  [S] Save  [Q/ESC] Quit"

        cv2.putText(canvas, status_line, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, active_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, cmd_line, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_zones(self, canvas: np.ndarray) -> np.ndarray:
        """Render polygon vertices, borders, and semi-transparent masks."""
        overlay = canvas.copy()
        
        zone_specs = [
            ("zone_1_koridor", COLOR_ZONE_1),
            ("zone_2_transit", COLOR_ZONE_2),
        ]

        for key, color in zone_specs:
            pts: List[List[int]] = self.zones[key]
            if not pts:
                continue

            np_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            
            # Fill polygon if >= 3 points
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [np_pts], color)

            # Draw polygon perimeter or segments (clean thickness = 1)
            is_closed = len(pts) >= 3
            cv2.polylines(canvas, [np_pts], is_closed, color, 1, cv2.LINE_AA)

            # Draw vertices
            for idx, pt in enumerate(pts):
                cv2.circle(canvas, (pt[0], pt[1]), 3, color, -1)
                cv2.putText(
                    canvas,
                    str(idx + 1),
                    (pt[0] + 4, pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Blend semi-transparent polygon fill
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        # Draw interactive cursor guide line
        active_pts = self.zones[self.active_zone_key]
        if active_pts and self.mouse_pos:
            last_pt = active_pts[-1]
            cv2.line(canvas, (last_pt[0], last_pt[1]), self.mouse_pos, COLOR_PREVIEW, 1, cv2.LINE_AA)

        return canvas

    def save_zones(self) -> None:
        """Write polygon coordinates and base_resolution to camera roi_zones.json."""
        out_data = {
            "base_resolution": [self.base_w, self.base_h],
        }
        for k, v in self.zones.items():
            if not k.startswith("_") and k != "base_resolution":
                out_data[k] = v

        with open(self.roi_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\n[SUCCESS] Saved ROI configurations (base: {self.base_w}x{self.base_h}) to: {self.roi_path}")

    def run(self) -> None:
        """Main calibration display loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n=======================================================")
        print(f" ROI Calibrator Running for [{self.camera_id}]")
        print(" [1] Edit Zone 1 (Koridor)")
        print(" [2] Edit Zone 2 (Transit)")
        print(" [U / Z] Undo Last Point | [C] Clear Active Zone")
        print(" [S] Save to JSON       | [Q / ESC] Quit")
        print("=======================================================\n")

        while True:
            display_frame = self.base_frame.copy()
            self._draw_zones(display_frame)
            self._draw_hud(display_frame)

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

            if key in (ord("q"), 27):  # 'q' or ESC
                print("[INFO] Exiting calibrator.")
                break
            elif key == ord("1"):
                self.active_zone_key = "zone_1_koridor"
                print("[STATUS] Switched to Zone 1 (Koridor Utama)")
            elif key == ord("2"):
                self.active_zone_key = "zone_2_transit"
                print("[STATUS] Switched to Zone 2 (Area Transit)")
            elif key in (ord("u"), ord("z")):
                if self.zones[self.active_zone_key]:
                    removed = self.zones[self.active_zone_key].pop()
                    print(f"[{self.active_zone_key}] Removed point: {removed}")
            elif key == ord("c"):
                self.zones[self.active_zone_key].clear()
                print(f"[{self.active_zone_key}] Cleared all points.")
            elif key == ord("s"):
                self.save_zones()

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Interactive OpenCV Polygon ROI Calibrator")
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
