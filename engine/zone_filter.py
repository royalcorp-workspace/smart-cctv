"""Multi-Zone ROI filtering and spatial polygon validation."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class ZoneFilter:
    """Manages ROI polygons and filters detected contours based on zone bounds and area thresholds."""

    def __init__(
        self,
        zones: Dict[str, List[List[int]]],
        zone_configs: Optional[Dict[str, dict]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        base_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        # Filter out non-polygon keys (such as 'base_resolution' or metadata keys)
        self.raw_zones: Dict[str, List[List[int]]] = {
            k: v for k, v in zones.items()
            if isinstance(v, list) and not k.startswith("_") and k != "base_resolution"
        }
        self.zone_configs: Dict[str, dict] = zone_configs or {}
        self.frame_shape: Optional[Tuple[int, int]] = frame_shape

        # Detect base resolution if not provided
        if base_resolution is not None:
            self.base_resolution: Tuple[int, int] = (int(base_resolution[0]), int(base_resolution[1]))
        elif "base_resolution" in zones and isinstance(zones["base_resolution"], (list, tuple)):
            self.base_resolution = (int(zones["base_resolution"][0]), int(zones["base_resolution"][1]))
        else:
            # Auto-detect from coordinates: if any x > 640 or y > 480 -> 1080p, else 640p
            max_x = 0
            max_y = 0
            for pts in self.raw_zones.values():
                for pt in pts:
                    if len(pt) >= 2:
                        max_x = max(max_x, pt[0])
                        max_y = max(max_y, pt[1])
            if max_x > 640 or max_y > 480:
                self.base_resolution = (1920, 1080)
            else:
                self.base_resolution = (640, 480)

        # Precompute downscaled polygon contours matching frame_shape (inference space, e.g. 640x480)
        self.scaled_zones: Dict[str, List[List[int]]] = {}
        self.polygon_contours: Dict[str, np.ndarray] = {}

        if self.frame_shape is not None and len(self.frame_shape) == 2:
            infer_h, infer_w = self.frame_shape
            base_w, base_h = self.base_resolution
            downscale_x = float(infer_w) / float(base_w) if base_w > 0 else 1.0
            downscale_y = float(infer_h) / float(base_h) if base_h > 0 else 1.0
        else:
            downscale_x = 1.0
            downscale_y = 1.0

        for zone_id, pts in self.raw_zones.items():
            if len(pts) >= 3:
                scaled_pts = [
                    [int(round(pt[0] * downscale_x)), int(round(pt[1] * downscale_y))]
                    for pt in pts
                ]
                self.scaled_zones[zone_id] = scaled_pts
                self.polygon_contours[zone_id] = np.array(scaled_pts, dtype=np.int32).reshape((-1, 1, 2))

    def get_zone_mask(self, zone_id: str, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Generate binary mask for a specific polygon zone."""
        target_shape = shape if shape is not None else self.frame_shape
        if target_shape is None:
            raise ValueError("Frame shape must be specified to generate zone mask.")

        mask = np.zeros((target_shape[0], target_shape[1]), dtype=np.uint8)
        if zone_id in self.polygon_contours:
            cv2.fillPoly(mask, [self.polygon_contours[zone_id]], 255)
        return mask

    def get_all_zones_mask(self, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Generate combined binary mask for all active ROI polygon zones."""
        target_shape = shape if shape is not None else self.frame_shape
        if target_shape is None:
            raise ValueError("Frame shape must be specified to generate zone mask.")

        mask = np.zeros((target_shape[0], target_shape[1]), dtype=np.uint8)
        all_polys = list(self.polygon_contours.values())
        if all_polys:
            cv2.fillPoly(mask, all_polys, 255)
        return mask

    def get_unattended_zones_mask(self, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Generate combined binary mask ONLY for zones configured for unattended bag detection."""
        target_shape = shape if shape is not None else self.frame_shape
        if target_shape is None:
            raise ValueError("Frame shape must be specified to generate zone mask.")

        mask = np.zeros((target_shape[0], target_shape[1]), dtype=np.uint8)
        unattended_polys = []

        for zone_id, contour in self.polygon_contours.items():
            zone_cfg = self.zone_configs.get(zone_id, {})
            if not zone_cfg and zone_id.replace("__", "_") in self.zone_configs:
                zone_cfg = self.zone_configs[zone_id.replace("__", "_")]

            is_unattended = zone_cfg.get("detect_unattended", None)
            if is_unattended is None:
                # Default: True for transit/penitipan zones; False for corridor/walkway
                is_unattended = ("transit" in zone_id.lower()) and ("koridor" not in zone_id.lower())

            if is_unattended:
                unattended_polys.append(contour)

        if unattended_polys:
            cv2.fillPoly(mask, unattended_polys, 255)
        return mask

    def check_point_in_zone(self, point: Tuple[int, int], zone_id: str, margin_px: float = 0.0) -> bool:
        """Check if a point (x, y) resides inside the specified polygon (or within margin_px tolerance)."""
        if zone_id not in self.polygon_contours:
            return False
        dist = float(cv2.pointPolygonTest(self.polygon_contours[zone_id], (float(point[0]), float(point[1])), True))
        return dist >= -margin_px

    def get_distance_to_nearest_zone(self, point: Tuple[int, int]) -> float:
        """Get signed distance in pixels to nearest zone boundary across all zones."""
        if not self.polygon_contours:
            return -999.0
        max_dist = -999.0
        for contour in self.polygon_contours.values():
            dist = float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))
            if dist > max_dist:
                max_dist = dist
        return max_dist

    def get_point_zone_distance(self, point: Tuple[int, int], zone_id: str) -> float:
        """Get signed distance in pixels from point to zone boundary (positive inside, negative outside)."""
        if zone_id not in self.polygon_contours:
            return -999.0
        return float(cv2.pointPolygonTest(self.polygon_contours[zone_id], (float(point[0]), float(point[1])), True))

    def is_point_outside_all_zones(self, point: Tuple[int, int], margin_px: float = 0.0) -> bool:
        """Return True if point resides strictly outside all registered ROI zones."""
        for contour in self.polygon_contours.values():
            dist = float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))
            if dist >= -margin_px:
                return False
        return True

    def find_zone_and_distance(
        self, point: Tuple[int, int], margin_px: float = 0.0
    ) -> Tuple[Optional[str], float]:
        """Find the matching zone_id and signed distance to nearest edge (positive inside, negative outside)."""
        best_zone: Optional[str] = None
        max_dist: float = -999.0

        for zone_id, contour in self.polygon_contours.items():
            dist = float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True))
            if dist >= -margin_px:
                if dist > max_dist:
                    max_dist = dist
                    best_zone = zone_id

        return best_zone, max_dist

    def find_zone_for_point(self, point: Tuple[int, int], margin_px: float = 0.0) -> Optional[str]:
        """Find the matching zone_id containing the point, if any."""
        zone, _ = self.find_zone_and_distance(point, margin_px=margin_px)
        return zone

    def filter_contours(
        self,
        mask: np.ndarray,
        min_area_default: int = 400,
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int], str]]:
        """Find contours in mask, validate polygon inclusion, and filter by minimum area.

        Returns list of tuples: (contour, bbox=(x, y, w, h), centroid=(cx, cy), zone_id)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_detections: List[Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int], str]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue

            # Calculate centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)

            # Spatial polygon matching
            matched_zone = self.find_zone_for_point(centroid)
            if matched_zone is None:
                continue

            # Area threshold validation per zone (support single or double underscore)
            zone_cfg = self.zone_configs.get(matched_zone, {})
            if not zone_cfg and matched_zone.replace("__", "_") in self.zone_configs:
                zone_cfg = self.zone_configs[matched_zone.replace("__", "_")]

            # Strict Zone Role Separation: Skip unattended object extraction if zone disables it
            is_unattended = zone_cfg.get("detect_unattended", None)
            if is_unattended is None:
                is_unattended = ("transit" in matched_zone.lower()) and ("koridor" not in matched_zone.lower())
            if not is_unattended:
                continue

            min_area = zone_cfg.get("min_contour_area", min_area_default)

            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Anti-reflection and tile glare filter (prevent duplicate boxes on glossy tiles)
                if w < 12 or h < 12:
                    continue
                aspect_ratio = w / float(h)
                if aspect_ratio < 0.15 or aspect_ratio > 6.0:
                    continue

                valid_detections.append((cnt, (x, y, w, h), centroid, matched_zone))

        return valid_detections
