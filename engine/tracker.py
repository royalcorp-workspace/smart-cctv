"""Centroid tracking engine with dwell calculation, deregistration hooks, and flicker tolerance."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class TrackedObject:
    """State of an individually tracked target."""
    track_id: int
    centroid: Tuple[int, int]
    anchor_centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    zone_id: str
    contour_area: float
    first_seen: float
    last_seen: float
    stationary_start: float
    dwell_duration: float = 0.0
    is_stationary: bool = False
    is_triggered: bool = False
    alert_sent: bool = False
    is_attended: bool = False
    is_active_this_frame: bool = True
    class_label: str = "object"
    db_event_id: Optional[int] = None
    missed_frames: int = 0
    edge_distance: float = 20.0
    confidence: float = 0.0
    initial_centroid: Tuple[int, int] = (0, 0)
    max_displacement_from_start: float = 0.0
    frame_count: int = 1
    attended_start: Optional[float] = None
    attended_duration: float = 0.0
    dwell_threshold: float = 3600.0

    @property
    def is_static_artifact(self) -> bool:
        """Check if person is a static background artifact (e.g. door frame / pillar).

        If person stays virtually 100% identical (displacement <= 2.0 px from registration)
        with confidence < 0.42 after initial frames (frame_count >= 5), dismiss as artifact.
        """
        if self.class_label == "person":
            if self.frame_count >= 5 and self.max_displacement_from_start <= 2.0 and self.confidence < 0.42:
                return True
        return False

    @property
    def should_render(self) -> bool:
        """Edge-aware visual grace period and static artifact dismissal.

        1. If identified as static background artifact -> do not render.
        2. Active in this frame -> always render.
        3. Missed in this frame:
           - For stationary objects or baggage (tas, backpack, handbag, suitcase):
             hold visual render for at least 30 frames (track buffer / coasting)
             so if detection drops for 1-2 frames, the box does not immediately disappear.
           - For moving persons inside zone (edge_distance >= 8.0 px): allow grace period up to 10 frames.
           - Near or outside boundary (edge_distance < 8.0 px): immediately hide (0 frames grace period).
        """
        if self.is_static_artifact:
            return False
        if self.is_active_this_frame:
            return True

        # Hold stationary objects and bags for at least 150 frames (~8-10 seconds at 15-18 FPS)
        is_bag = self.class_label in ("tas", "backpack", "handbag", "suitcase")
        if self.is_stationary or is_bag:
            return self.missed_frames <= 150

        return self.edge_distance >= 8.0 and self.missed_frames <= 10


class CentroidTracker:
    """Associates centroids across frames, tracks dwell duration, and purges stale records."""

    def __init__(
        self,
        max_distance_px: float = 50.0,
        movement_threshold_px: float = 15.0,
        anchor_radius_px: float = 15.0,
        flicker_tolerance_sec: float = 2.0,
        max_disappeared_sec: float = 12.0,
        max_age_frames: int = 150,
        ema_alpha: float = 0.3,
    ) -> None:
        self.max_distance_px: float = max_distance_px
        self.movement_threshold_px: float = movement_threshold_px
        self.anchor_radius_px: float = anchor_radius_px
        self.flicker_tolerance_sec: float = flicker_tolerance_sec
        self.max_disappeared_sec: float = max_disappeared_sec
        self.max_age_frames: int = max_age_frames
        self.ema_alpha: float = ema_alpha

        self._next_id: int = 1
        self.objects: Dict[int, TrackedObject] = {}

    def _register(
        self,
        centroid: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        zone_id: str,
        area: float,
        timestamp: float,
        label: Optional[str] = None,
        edge_dist: float = 20.0,
        conf: float = 0.0,
    ) -> TrackedObject:
        """Register a new tracked object."""
        if label is None:
            label = "tas" if zone_id else "person"

        is_bag_obj = label in ("tas", "backpack", "handbag", "suitcase")
        new_obj = TrackedObject(
            track_id=self._next_id,
            centroid=centroid,
            anchor_centroid=centroid,
            bbox=bbox,
            zone_id=zone_id,
            contour_area=area,
            first_seen=timestamp,
            last_seen=timestamp,
            stationary_start=timestamp,
            dwell_duration=0.0,
            is_stationary=is_bag_obj,
            is_triggered=False,
            alert_sent=False,
            is_attended=False,
            is_active_this_frame=True,
            class_label=label,
            db_event_id=None,
            missed_frames=0,
            edge_distance=edge_dist,
            confidence=conf,
            initial_centroid=centroid,
            max_displacement_from_start=0.0,
            frame_count=1,
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

        detections format: list of (bbox, centroid, zone_id, area, [label], [edge_dist], [conf])
        Returns: (active_tracks: List[TrackedObject], purged_tracks: List[TrackedObject])
        """
        now: float = timestamp if timestamp is not None else time.time()

        # Mark all existing objects as inactive for current frame by default
        for obj in self.objects.values():
            obj.is_active_this_frame = False

        # If no detections provided, hold dwell duration and purge stale objects
        if len(detections) == 0:
            for obj in self.objects.values():
                obj.missed_frames += 1
            purged = self._purge_stale_objects(now)
            return list(self.objects.values()), purged

        # If no tracked objects currently exist, register all detections
        if len(self.objects) == 0:
            for det in detections:
                bbox, centroid, zone_id, area = det[0], det[1], det[2], det[3]
                det_label = det[4] if len(det) >= 5 else None
                edge_dist = float(det[5]) if len(det) >= 6 else 20.0
                conf = float(det[6]) if len(det) >= 7 else 0.0
                self._register(centroid, bbox, zone_id, area, now, label=det_label, edge_dist=edge_dist, conf=conf)
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

            # Nearest-centroid matching: Euclidean distance threshold <= 50 px
            if dist_matrix[row, col] > self.max_distance_px:
                continue

            obj_id = existing_ids[row]
            obj = self.objects[obj_id]
            det = detections[col]
            bbox, new_centroid, zone_id, area = det[0], det[1], det[2], det[3]
            det_label = det[4] if len(det) >= 5 else None
            edge_dist = float(det[5]) if len(det) >= 6 else 20.0
            conf = float(det[6]) if len(det) >= 7 else 0.0

            # 1. EMA Centroid Smoothing: centroid = alpha * new_centroid + (1 - alpha) * old_centroid (alpha = 0.3)
            old_cx, old_cy = obj.centroid
            new_cx, new_cy = new_centroid
            smooth_cx = int(round(self.ema_alpha * float(new_cx) + (1.0 - self.ema_alpha) * float(old_cx)))
            smooth_cy = int(round(self.ema_alpha * float(new_cy) + (1.0 - self.ema_alpha) * float(old_cy)))
            smoothed_centroid = (smooth_cx, smooth_cy)

            # Calculate distance from anchor position using smoothed centroid
            anchor_dist = float(
                np.linalg.norm(
                    np.array(obj.anchor_centroid, dtype=np.float32) - np.array(smoothed_centroid, dtype=np.float32)
                )
            )

            # Update displacement from initial registration point
            dist_from_start = float(
                np.linalg.norm(
                    np.array(obj.initial_centroid, dtype=np.float32) - np.array(smoothed_centroid, dtype=np.float32)
                )
            )
            if dist_from_start > obj.max_displacement_from_start:
                obj.max_displacement_from_start = dist_from_start

            obj.confidence = conf
            obj.frame_count += 1

            # 2. Universal Stationary Logic (displacement from anchor_point <= 15 px)
            is_bag_obj = (det_label or obj.class_label) in ("tas", "backpack", "handbag", "suitcase")
            if is_bag_obj:
                if anchor_dist <= self.anchor_radius_px:
                    # Fluctuation / jitter <= 15 px is DIAM (stationary) -> continuously accumulate dwell time
                    if not obj.is_stationary:
                        obj.is_stationary = True
                        if obj.dwell_duration > 0.0:
                            # Resume smoothly from held dwell duration
                            obj.stationary_start = now - obj.dwell_duration
                    if not obj.is_attended:
                        obj.dwell_duration = now - obj.stationary_start
                    else:
                        # Hold/pause dwell time while attended
                        obj.stationary_start = now - obj.dwell_duration
                else:
                    # Objek berpindah posisi nyata (> 15 px) -> reset dwell timer
                    obj.is_stationary = False
                    obj.stationary_start = now
                    obj.anchor_centroid = smoothed_centroid
                    obj.dwell_duration = 0.0
                    obj.is_triggered = False
                    obj.alert_sent = False
            else:
                # Person never triggers dwell violation
                obj.is_stationary = False
                obj.stationary_start = now
                obj.dwell_duration = 0.0
                obj.is_triggered = False
                obj.alert_sent = False

            if zone_id and zone_id != obj.zone_id:
                obj.zone_id = zone_id

            obj.centroid = smoothed_centroid
            obj.bbox = bbox
            obj.contour_area = area
            obj.last_seen = now
            obj.is_active_this_frame = True
            obj.missed_frames = 0
            obj.edge_distance = edge_dist

            if det_label is not None:
                obj.class_label = det_label

            assigned_rows.add(row)
            assigned_cols.add(col)

        # 3. Grace Period / Coasting: hold/pause accumulated dwell duration for objects missed this frame
        for row, obj_id in enumerate(existing_ids):
            if row not in assigned_rows:
                unmatched_obj = self.objects[obj_id]
                unmatched_obj.missed_frames += 1
                # Dwell duration is held intact (not reset to 0.0)

        # 4. Anti ID Churning Guard for unassigned detections
        # If an unassigned detection is within 50 px of ANY existing track in memory, REUSE that ID!
        for col, det in enumerate(detections):
            if col in assigned_cols:
                continue

            det_bbox, det_centroid, zone_id, area = det[0], det[1], det[2], det[3]
            det_label = det[4] if len(det) >= 5 else None
            edge_dist = float(det[5]) if len(det) >= 6 else 20.0
            conf = float(det[6]) if len(det) >= 7 else 0.0

            matched_existing = None
            min_existing_dist = self.max_distance_px  # 50 px threshold

            for ex_id, ex_obj in self.objects.items():
                if ex_id in [existing_ids[r] for r in assigned_rows]:
                    continue  # already matched to another detection in this frame

                d_cent = float(np.linalg.norm(np.array(ex_obj.centroid, dtype=np.float32) - np.array(det_centroid, dtype=np.float32)))
                d_anch = float(np.linalg.norm(np.array(ex_obj.anchor_centroid, dtype=np.float32) - np.array(det_centroid, dtype=np.float32)))
                best_d = min(d_cent, d_anch)
                if best_d <= min_existing_dist:
                    min_existing_dist = best_d
                    matched_existing = ex_obj

            if matched_existing is not None:
                # REUSE EXISTING TRACK ID! DO NOT CREATE A NEW ID!
                old_cx, old_cy = matched_existing.centroid
                new_cx, new_cy = det_centroid
                smooth_cx = int(round(self.ema_alpha * float(new_cx) + (1.0 - self.ema_alpha) * float(old_cx)))
                smooth_cy = int(round(self.ema_alpha * float(new_cy) + (1.0 - self.ema_alpha) * float(old_cy)))
                matched_existing.centroid = (smooth_cx, smooth_cy)
                matched_existing.bbox = det_bbox
                matched_existing.contour_area = area
                matched_existing.last_seen = now
                matched_existing.is_active_this_frame = True
                matched_existing.missed_frames = 0
                matched_existing.edge_distance = edge_dist
                matched_existing.confidence = conf
                if det_label is not None:
                    matched_existing.class_label = det_label
                if zone_id and zone_id != matched_existing.zone_id:
                    matched_existing.zone_id = zone_id

                # Stationary logic: continue accumulating dwell time
                is_bag_obj = matched_existing.class_label in ("tas", "backpack", "handbag", "suitcase")
                if is_bag_obj:
                    anchor_d = float(np.linalg.norm(np.array(matched_existing.anchor_centroid, dtype=np.float32) - np.array(matched_existing.centroid, dtype=np.float32)))
                    if anchor_d <= self.anchor_radius_px:
                        if not matched_existing.is_stationary:
                            matched_existing.is_stationary = True
                            if matched_existing.dwell_duration > 0.0:
                                matched_existing.stationary_start = now - matched_existing.dwell_duration
                        if not matched_existing.is_attended:
                            matched_existing.dwell_duration = now - matched_existing.stationary_start
                        else:
                            matched_existing.stationary_start = now - matched_existing.dwell_duration
                    else:
                        matched_existing.is_stationary = False
                        matched_existing.stationary_start = now
                        matched_existing.anchor_centroid = matched_existing.centroid
                        matched_existing.dwell_duration = 0.0
                        matched_existing.is_triggered = False
                        matched_existing.alert_sent = False
            else:
                # Only register a new ID if completely outside 50 px of ANY known object
                self._register(det_centroid, det_bbox, zone_id, area, now, label=det_label, edge_dist=edge_dist, conf=conf)

        # 5. Universal owner proximity across all zones
        self.evaluate_owner_proximity(now)

        # Anti-memory leak: purge tracks unobserved for > max_disappeared_sec (4.5s) or > max_age_frames (45 frames)
        purged = self._purge_stale_objects(now)

        return list(self.objects.values()), purged

    def _purge_stale_objects(self, now: float) -> List[TrackedObject]:
        """Deregister objects that disappeared longer than max_disappeared_sec and max_age_frames."""
        stale_ids = []
        for obj_id, obj in self.objects.items():
            is_bag = obj.class_label in ("tas", "backpack", "handbag", "suitcase")
            if obj.is_stationary or is_bag:
                max_frames = max(150, self.max_age_frames)
                max_sec = max(12.0, self.max_disappeared_sec)
            else:
                max_frames = self.max_age_frames
                max_sec = self.max_disappeared_sec

            if (now - obj.last_seen) > max_sec and obj.missed_frames > max_frames:
                stale_ids.append(obj_id)

        purged_objects: List[TrackedObject] = []
        for obj_id in stale_ids:
            purged_objects.append(self.objects[obj_id])
            del self.objects[obj_id]
        return purged_objects

    def evaluate_owner_proximity(self, now: float) -> None:
        """Adaptive owner proximity evaluation and passerby debouncing.

        1. Adaptive proximity radius:
           Uses bottom-center (contact with floor) for person and bag:
           person_foot = (person_box.x + person_box.w / 2, person_box.y + person_box.h)
           bag_base = (bag_box.x + bag_box.w / 2, bag_box.y + bag_box.h)
           proximity_radius = max(45, min(90, int(bag_h * 2.0)))

        2. Passerby debouncing:
           - When person is within proximity_radius:
             is_attended = True (HUD turns Green [AMAN / ATTENDED])
             dwell_duration is PAUSED / HELD.
           - Reset to 0.0s ONLY if person stays for >= 4.0 consecutive seconds.
           - If person leaves before 4.0s (passerby), resume dwell_duration from held value.
        """
        all_objects = list(self.objects.values())
        bag_objects = [
            obj for obj in all_objects
            if obj.class_label in ("backpack", "handbag", "suitcase", "tas")
        ]

        for bag in bag_objects:
            bx, by, bw, bh = bag.bbox
            bag_base = (float(bx + bw / 2.0), float(by + bh))
            adaptive_radius = float(max(45, min(90, int(bh * 2.0))))

            owner_nearby = False
            for other in all_objects:
                if other.track_id == bag.track_id:
                    continue

                ox, oy, ow, oh = other.bbox
                other_foot = (float(ox + ow / 2.0), float(oy + oh))

                dist = float(
                    np.linalg.norm(
                        np.array(bag_base, dtype=np.float32) - np.array(other_foot, dtype=np.float32)
                    )
                )

                is_person_or_moving = (other.class_label == "person" or not other.is_stationary) and getattr(other, "is_active_this_frame", False)

                # Check overlap / carried containment with active person:
                # If bag centroid is inside person box or IoF > 0.3
                is_carried_or_overlapping = False
                if other.class_label == "person" and getattr(other, "is_active_this_frame", False):
                    bcx, bcy = bag.centroid
                    if ox <= bcx <= (ox + ow) and oy <= bcy <= (oy + oh):
                        is_carried_or_overlapping = True
                    else:
                        ix1 = max(bx, ox)
                        iy1 = max(by, oy)
                        ix2 = min(bx + bw, ox + ow)
                        iy2 = min(by + bh, oy + oh)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = float((ix2 - ix1) * (iy2 - iy1))
                            bag_area = float(bw * bh)
                            if bag_area > 0 and (inter / bag_area) > 0.3:
                                is_carried_or_overlapping = True

                if (is_person_or_moving and dist <= adaptive_radius) or is_carried_or_overlapping:
                    owner_nearby = True
                    break

            if owner_nearby:
                bag.is_attended = True
                if bag.attended_start is None:
                    bag.attended_start = now
                bag.attended_duration = now - bag.attended_start

                # Sustained attendance (>= 4.0 consecutive seconds): reset dwell timer to 0
                if bag.attended_duration >= 4.0:
                    bag.dwell_duration = 0.0
                    bag.stationary_start = now
                    bag.is_triggered = False
                    bag.alert_sent = False
                else:
                    # Passerby (< 4.0s): HOLD/PAUSE dwell timer at current accumulated duration
                    bag.stationary_start = now - bag.dwell_duration
            else:
                if bag.is_attended:
                    # Person walked away (passerby or former owner leaving)
                    bag.is_attended = False
                    bag.attended_start = None
                    bag.attended_duration = 0.0
                    # Resume smoothly from held dwell duration
                    bag.stationary_start = now - bag.dwell_duration
