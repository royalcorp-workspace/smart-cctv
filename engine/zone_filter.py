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
    ) -> None:
        self.raw_zones: Dict[str, List[List[int]]] = zones
        self.zone_configs: Dict[str, dict] = zone_configs or {}
        self.frame_shape: Optional[Tuple[int, int]] = frame_shape

        # Precompute polygon contours
        self.polygon_contours: Dict[str, np.ndarray] = {}
        for zone_id, pts in zones.items():
            if len(pts) >= 3:
                self.polygon_contours[zone_id] = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

    def get_zone_mask(self, zone_id: str, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Generate binary mask for a specific polygon zone."""
        target_shape = shape if shape is not None else self.frame_shape
        if target_shape is None:
            raise ValueError("Frame shape must be specified to generate zone mask.")

        mask = np.zeros((target_shape[0], target_shape[1]), dtype=np.uint8)
        if zone_id in self.polygon_contours:
            cv2.fillPoly(mask, [self.polygon_contours[zone_id]], 255)
        return mask

    def check_point_in_zone(self, point: Tuple[int, int], zone_id: str) -> bool:
        """Check if a point (x, y) resides inside the specified polygon."""
        if zone_id not in self.polygon_contours:
            return False
        res = cv2.pointPolygonTest(self.polygon_contours[zone_id], (float(point[0]), float(point[1])), False)
        return res >= 0

    def find_zone_for_point(self, point: Tuple[int, int]) -> Optional[str]:
        """Find the matching zone_id containing the point, if any."""
        for zone_id, contour in self.polygon_contours.items():
            if cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0:
                return zone_id
        return None

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

            # Area threshold validation per zone
            zone_cfg = self.zone_configs.get(matched_zone, {})
            min_area = zone_cfg.get("min_contour_area", min_area_default)

            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                valid_detections.append((cnt, (x, y, w, h), centroid, matched_zone))

        return valid_detections
