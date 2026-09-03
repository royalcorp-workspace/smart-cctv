"""Centroid tracking engine with dwell calculation, deregistration hooks, and anti-leak."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrackedObject:
    """State of an individually tracked target."""
    track_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    zone_id: str
    contour_area: float
    first_seen: float
    last_seen: float
    stationary_start: float
    dwell_duration: float = 0.0
    is_stationary: bool = False
    is_triggered: bool = False
    db_event_id: Optional[int] = None


class CentroidTracker:
    """Associates centroids across frames, tracks dwell duration, and purges stale records."""

    def __init__(
        self,
        max_distance_px: float = 50.0,
        movement_threshold_px: float = 8.0,
        max_disappeared_sec: float = 5.0,
    ) -> None:
        self.max_distance_px: float = max_distance_px
        self.movement_threshold_px: float = movement_threshold_px
        self.max_disappeared_sec: float = max_disappeared_sec

        self._next_id: int = 1
        self.objects: Dict[int, TrackedObject] = {}

    def _register(
        self,
        centroid: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        zone_id: str,
        area: float,
        timestamp: float,
    ) -> TrackedObject:
        """Register a new tracked object."""
        new_obj = TrackedObject(
            track_id=self._next_id,
            centroid=centroid,
            bbox=bbox,
            zone_id=zone_id,
            contour_area=area,
            first_seen=timestamp,
            last_seen=timestamp,
            stationary_start=timestamp,
            dwell_duration=0.0,
            is_stationary=False,
            is_triggered=False,
            db_event_id=None,
        )
        self.objects[self._next_id] = new_obj
        self._next_id += 1
        return new_obj

    def update(
        self,
        detections: List[Tuple[Tuple[int, int, int, int], Tuple[int, int], str, float]],
        timestamp: Optional[float] = None,
    ) -> Tuple[List[TrackedObject], List[TrackedObject]]:
        """Update tracker state with new frame detections.

        detections format: list of (bbox, centroid, zone_id, area)
        Returns: (active_tracks: List[TrackedObject], purged_tracks: List[TrackedObject])
        """
        now: float = timestamp if timestamp is not None else time.time()

        # If no detections provided, purge stale objects and return
        if len(detections) == 0:
            purged = self._purge_stale_objects(now)
            return list(self.objects.values()), purged

        # If no tracked objects currently exist, register all detections
        if len(self.objects) == 0:
            for bbox, centroid, zone_id, area in detections:
                self._register(centroid, bbox, zone_id, area, now)
            return list(self.objects.values()), []

        # Build distance matrix between existing objects and incoming detections
        existing_ids = list(self.objects.keys())
        existing_centroids = np.array([self.objects[oid].centroid for oid in existing_ids], dtype=np.float32)
        incoming_centroids = np.array([d[1] for d in detections], dtype=np.float32)

        # Distance matrix: shape (N_existing, M_detections)
        dist_matrix = np.linalg.norm(
            existing_centroids[:, np.newaxis] - incoming_centroids[np.newaxis, :],
            axis=2,
        )

        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for row, col in zip(rows, cols):
            if row in assigned_rows or col in assigned_cols:
                continue

            # Check if distance exceeds matching threshold
            if dist_matrix[row, col] > self.max_distance_px:
                continue

            obj_id = existing_ids[row]
            obj = self.objects[obj_id]
            bbox, new_centroid, zone_id, area = detections[col]

            # Calculate displacement
            displacement = float(
                np.linalg.norm(
                    np.array(obj.centroid, dtype=np.float32) - np.array(new_centroid, dtype=np.float32)
                )
            )

            # Update stationary vs transient logic
            if displacement <= self.movement_threshold_px:
                obj.is_stationary = True
                obj.dwell_duration = now - obj.stationary_start
            else:
                # Object moved significantly -> reset stationary clock
                obj.is_stationary = False
                obj.stationary_start = now
                obj.dwell_duration = 0.0

            obj.centroid = new_centroid
            obj.bbox = bbox
            obj.zone_id = zone_id
            obj.contour_area = area
            obj.last_seen = now

            assigned_rows.add(row)
            assigned_cols.add(col)

        # Register unassigned new detections
        for col, det in enumerate(detections):
            if col not in assigned_cols:
                self._register(det[1], det[0], det[2], det[3], now)

        # Anti-memory leak: purge tracks unobserved for > max_disappeared_sec
        purged = self._purge_stale_objects(now)

        return list(self.objects.values()), purged

    def _purge_stale_objects(self, now: float) -> List[TrackedObject]:
        """Deregister objects that disappeared longer than max_disappeared_sec."""
        stale_ids = [
            obj_id
            for obj_id, obj in self.objects.items()
            if (now - obj.last_seen) > self.max_disappeared_sec
        ]
        purged_objects: List[TrackedObject] = []
        for obj_id in stale_ids:
            purged_objects.append(self.objects[obj_id])
            del self.objects[obj_id]
        return purged_objects
