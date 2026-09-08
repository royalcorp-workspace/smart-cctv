"""Centroid tracking engine with dwell calculation, deregistration hooks, and flicker tolerance."""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from engine.logger import logger


def compute_bbox_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) of two [x, y, w, h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter_area = float((xi2 - xi1) * (yi2 - yi1))
    area1 = float(w1 * h1)
    area2 = float(w2 * h2)
    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


@dataclass
class SpatialMemoryEntry:
    """Historical spatial snapshot of an unobserved or occluded stationary object."""
    track_id: int
    class_label: str
    zone_id: str
    anchor_centroid: Tuple[int, int]
    last_centroid: Tuple[int, int]
    last_bbox: Tuple[int, int, int, int]
    first_seen: float
    stationary_start: float
    accumulated_dwell: float
    last_seen: float
    deregistered_at: float
    occluded_by_person: bool = False
    contour_area: float = 0.0
    confidence: float = 0.0
    dwell_threshold: float = 3600.0
    last_owner_info: Optional[Dict[str, Any]] = None
    anchor_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)


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
    is_occluded: bool = False
    last_occluded_time: float = 0.0
    last_owner_info: Optional[Dict[str, Any]] = None
    is_warning: bool = False
    is_pre_alarm: bool = False
    pre_alarm_alerted: bool = False
    anchor_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    moved_confirmation_frames: int = 0
    last_moved_time: float = 0.0

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

        # Hold stationary objects and bags
        # Stationary Bag Latching: keep confirmed stationary bag rendering up to 600 frames (~45s)
        is_bag = self.class_label in ("tas", "backpack", "handbag", "suitcase")
        if is_bag and self.is_stationary:
            return self.missed_frames <= 600

        if self.is_stationary or is_bag:
            max_render_missed = 300 if self.is_occluded else 150
            return self.missed_frames <= max_render_missed

        return self.edge_distance >= 8.0 and self.missed_frames <= 10


class CentroidTracker:
    """Associates centroids across frames, tracks dwell duration, and purges stale records."""

    def __init__(
        self,
        max_distance_px: float = 60.0,
        movement_threshold_px: float = 15.0,
        anchor_radius_px: float = 40.0,
        flicker_tolerance_sec: float = 2.0,
        max_disappeared_sec: float = 12.0,
        max_age_frames: int = 150,
        ema_alpha: float = 0.3,
        spatial_memory_ttl_sec: float = 180.0,
        spatial_match_distance_px: float = 60.0,
        stationary_max_age_frames: int = 600,
        stationary_max_disappeared_sec: float = 45.0,
    ) -> None:
        self.max_distance_px: float = max_distance_px
        self.movement_threshold_px: float = movement_threshold_px
        self.anchor_radius_px: float = anchor_radius_px
        self.flicker_tolerance_sec: float = flicker_tolerance_sec
        self.max_disappeared_sec: float = max_disappeared_sec
        self.max_age_frames: int = max_age_frames
        self.ema_alpha: float = ema_alpha
        self.spatial_memory_ttl_sec: float = spatial_memory_ttl_sec
        self.spatial_match_distance_px: float = spatial_match_distance_px
        self.stationary_max_age_frames: int = stationary_max_age_frames
        self.stationary_max_disappeared_sec: float = stationary_max_disappeared_sec

        self._next_id: int = 1
        self.objects: Dict[int, TrackedObject] = {}
        self.spatial_memory: Dict[int, SpatialMemoryEntry] = {}

    def _match_spatial_memory(
        self,
        centroid: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        zone_id: str,
        label: Optional[str],
        now: float,
    ) -> Optional[SpatialMemoryEntry]:
        """Search spatial memory cache for a matching previously observed stationary bag."""
        det_label = label if label is not None else ("tas" if zone_id else "person")
        if det_label not in ("tas", "backpack", "handbag", "suitcase"):
            return None

        best_entry: Optional[SpatialMemoryEntry] = None
        min_dist = max(60.0, self.spatial_match_distance_px)

        for entry in self.spatial_memory.values():
            if (now - entry.deregistered_at) > self.spatial_memory_ttl_sec:
                continue

            # Check Euclidean distance to anchor and last observed centroid
            d_anchor = float(np.linalg.norm(np.array(entry.anchor_centroid, dtype=np.float32) - np.array(centroid, dtype=np.float32)))
            d_last = float(np.linalg.norm(np.array(entry.last_centroid, dtype=np.float32) - np.array(centroid, dtype=np.float32)))
            closest_d = min(d_anchor, d_last)

            # Check Spatial Bounding Box IoU
            box_iou = compute_bbox_iou(bbox, entry.last_bbox)

            if closest_d <= min_dist or box_iou >= 0.20:
                min_dist = closest_d
                best_entry = entry

        return best_entry

    def _recover_from_spatial_memory(
        self,
        entry: SpatialMemoryEntry,
        centroid: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        zone_id: str,
        area: float,
        label: Optional[str],
        edge_dist: float,
        conf: float,
        now: float,
    ) -> TrackedObject:
        """Revive a TrackedObject from SpatialMemoryEntry preserving Track ID and accumulated dwell time."""
        effective_label = label or entry.class_label
        saved_anchor_bbox = getattr(entry, "anchor_bbox", bbox)
        if saved_anchor_bbox == (0, 0, 0, 0):
            saved_anchor_bbox = bbox
        recovered_obj = TrackedObject(
            track_id=entry.track_id,
            centroid=centroid,
            anchor_centroid=entry.anchor_centroid,
            anchor_bbox=saved_anchor_bbox,
            bbox=bbox,
            zone_id=zone_id or entry.zone_id,
            contour_area=area if area > 0 else entry.contour_area,
            first_seen=entry.first_seen,
            last_seen=now,
            stationary_start=now - entry.accumulated_dwell,
            dwell_duration=entry.accumulated_dwell,
            is_stationary=True,
            is_triggered=False,
            alert_sent=False,
            is_attended=False,
            is_active_this_frame=True,
            class_label=effective_label,
            db_event_id=None,
            missed_frames=0,
            edge_distance=edge_dist,
            confidence=conf if conf > 0 else entry.confidence,
            initial_centroid=entry.anchor_centroid,
            max_displacement_from_start=0.0,
            frame_count=2,
            attended_start=None,
            attended_duration=0.0,
            dwell_threshold=entry.dwell_threshold,
            is_occluded=False,
            last_occluded_time=0.0,
            last_owner_info=entry.last_owner_info,
            is_warning=(entry.accumulated_dwell >= entry.dwell_threshold * 0.50),
            is_pre_alarm=(entry.accumulated_dwell >= entry.dwell_threshold * 0.85),
            pre_alarm_alerted=False,
            moved_confirmation_frames=0,
            last_moved_time=0.0,
        )
        self.objects[entry.track_id] = recovered_obj
        if entry.track_id in self.spatial_memory:
            del self.spatial_memory[entry.track_id]

        logger.info(
            f"[TRACKER] Track ID {entry.track_id} RECOVERED from Spatial Memory "
            f"(zone='{recovered_obj.zone_id}', preserved dwell={entry.accumulated_dwell:.1f}s, "
            f"absent_duration={now - entry.deregistered_at:.1f}s)."
        )
        return recovered_obj

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
            anchor_bbox=bbox,
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
            moved_confirmation_frames=0,
            last_moved_time=0.0,
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

        # If no tracked objects currently exist, check spatial memory before registering new objects
        if len(self.objects) == 0:
            for det in detections:
                bbox, centroid, zone_id, area = det[0], det[1], det[2], det[3]
                det_label = det[4] if len(det) >= 5 else None
                edge_dist = float(det[5]) if len(det) >= 6 else 20.0
                conf = float(det[6]) if len(det) >= 7 else 0.0

                matched_spatial = self._match_spatial_memory(centroid, bbox, zone_id, det_label, now)
                if matched_spatial is not None:
                    self._recover_from_spatial_memory(
                        matched_spatial, centroid, bbox, zone_id, area, det_label, edge_dist, conf, now
                    )
                else:
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
            det_label = det[4] if len(det) >= 5 else None

            # Class compatibility guard: Person can NEVER match a Bag, and Bag can NEVER match a Person
            is_obj_bag = obj.class_label in ("tas", "backpack", "handbag", "suitcase")
            is_det_bag = det_label in ("tas", "backpack", "handbag", "suitcase")
            if (is_obj_bag and det_label == "person") or (obj.class_label == "person" and is_det_bag):
                continue

            bbox, new_centroid, zone_id, area = det[0], det[1], det[2], det[3]
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

            # 2. Sticky Stationary State Machine for Baggage
            is_bag_obj = (det_label or obj.class_label) in ("tas", "backpack", "handbag", "suitcase")
            if is_bag_obj:
                anchor_box = getattr(obj, "anchor_bbox", obj.bbox)
                anchor_iou = compute_bbox_iou(anchor_box, bbox)
                is_near_anchor = (anchor_dist <= self.anchor_radius_px) or (anchor_iou >= 0.50)

                if is_near_anchor:
                    # Near anchor: normal coordinate jitter / fluctuation, reset moved confirmation counter
                    obj.moved_confirmation_frames = 0
                    if not obj.is_stationary:
                        obj.is_stationary = True
                        if obj.dwell_duration > 0.0:
                            obj.stationary_start = now - obj.dwell_duration
                else:
                    # Beyond anchor radius: check if genuinely moving (> 50.0 px)
                    if anchor_dist > 50.0:
                        obj.moved_confirmation_frames += 1
                    else:
                        obj.moved_confirmation_frames = max(0, obj.moved_confirmation_frames - 1)

                # STICKY STATIONARY LOGIC:
                if obj.is_stationary:
                    if obj.moved_confirmation_frames >= 30:
                        # Genuine displacement confirmed: moved > 50px for >= 30 consecutive frames!
                        obj.is_stationary = False
                        obj.moved_confirmation_frames = 0
                        obj.last_moved_time = now
                        obj.stationary_start = now
                        obj.anchor_centroid = smoothed_centroid
                        obj.anchor_bbox = bbox
                        obj.dwell_duration = 0.0
                        obj.is_triggered = False
                        obj.alert_sent = False
                        obj.is_warning = False
                        obj.is_pre_alarm = False
                        obj.pre_alarm_alerted = False
                        logger.info(
                            f"[TRACKER] Track ID {obj.track_id} confirmed moved (>50px for 30 consecutive frames). Resetting stationary & dwell."
                        )
                    else:
                        # STICKY: Maintain stationary status!
                        # Accumulate dwell_duration continuously unless attended by owner
                        if not obj.is_attended:
                            obj.dwell_duration = now - obj.stationary_start
                        else:
                            obj.stationary_start = now - obj.dwell_duration
                        obj.is_warning = (obj.dwell_duration >= obj.dwell_threshold * 0.50)
                        obj.is_pre_alarm = (obj.dwell_duration >= obj.dwell_threshold * 0.85)
                else:
                    if is_near_anchor:
                        obj.is_stationary = True
                        obj.stationary_start = now
                        obj.dwell_duration = 0.0
                    else:
                        obj.stationary_start = now
                        obj.dwell_duration = 0.0
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
            obj.is_occluded = False
            obj.last_occluded_time = 0.0

            if det_label is not None:
                obj.class_label = det_label

            assigned_rows.add(row)
            assigned_cols.add(col)

        # Extract person bounding boxes for occlusion evaluation
        person_boxes = [d[0] for d in detections if (len(d) >= 5 and d[4] == "person")]

        # 3. Grace Period / Coasting & Occlusion Detection for objects missed this frame
        for row, obj_id in enumerate(existing_ids):
            if row not in assigned_rows:
                unmatched_obj = self.objects[obj_id]
                unmatched_obj.missed_frames += 1

                is_bag = unmatched_obj.class_label in ("tas", "backpack", "handbag", "suitcase")
                if is_bag and unmatched_obj.is_stationary:
                    bx, by, bw, bh = unmatched_obj.bbox
                    bcx, bcy = unmatched_obj.centroid
                    bag_area = float(bw * bh)

                    occluded = False
                    for (px, py, pw, ph) in person_boxes:
                        # 1. Centroid inside person bounding box (with 10px margin)
                        if (px - 10) <= bcx <= (px + pw + 10) and (py - 10) <= bcy <= (py + ph + 10):
                            occluded = True
                            break
                        # 2. IoF / Overlap between bag and person box >= 0.20
                        ix1 = max(bx, px)
                        iy1 = max(by, py)
                        ix2 = min(bx + bw, px + pw)
                        iy2 = min(by + bh, py + ph)
                        if ix2 > ix1 and iy2 > iy1:
                            inter = float((ix2 - ix1) * (iy2 - iy1))
                            if bag_area > 0 and (inter / bag_area) >= 0.20:
                                occluded = True
                                break

                    if occluded:
                        unmatched_obj.is_occluded = True
                        unmatched_obj.last_occluded_time = now
                        # Freeze/Hold dwell time during occlusion
                        unmatched_obj.stationary_start = now - unmatched_obj.dwell_duration
                    else:
                        if unmatched_obj.is_occluded and (now - unmatched_obj.last_occluded_time) < 3.0:
                            # Hold occluded flag for 3s lingering grace
                            unmatched_obj.stationary_start = now - unmatched_obj.dwell_duration
                        else:
                            unmatched_obj.is_occluded = False
                            # Sticky stationary: continue accumulating dwell duration even when YOLO missed detection
                            if not unmatched_obj.is_attended:
                                unmatched_obj.dwell_duration = now - unmatched_obj.stationary_start
                            else:
                                unmatched_obj.stationary_start = now - unmatched_obj.dwell_duration
                            unmatched_obj.is_warning = (unmatched_obj.dwell_duration >= unmatched_obj.dwell_threshold * 0.50)
                            unmatched_obj.is_pre_alarm = (unmatched_obj.dwell_duration >= unmatched_obj.dwell_threshold * 0.85)

        # 4. Anti ID Churning Guard for unassigned detections
        # First check active memory within 60 px for bags (or 50px for person); if not found, check Spatial Memory!
        for col, det in enumerate(detections):
            if col in assigned_cols:
                continue

            det_bbox, det_centroid, zone_id, area = det[0], det[1], det[2], det[3]
            det_label = det[4] if len(det) >= 5 else None
            edge_dist = float(det[5]) if len(det) >= 6 else 20.0
            conf = float(det[6]) if len(det) >= 7 else 0.0

            is_det_bag = det_label in ("tas", "backpack", "handbag", "suitcase")
            max_match_dist = 60.0 if is_det_bag else self.max_distance_px

            matched_existing = None
            min_existing_dist = max_match_dist

            for ex_id, ex_obj in self.objects.items():
                if ex_id in [existing_ids[r] for r in assigned_rows]:
                    continue  # already matched to another detection in this frame

                # Class compatibility guard: Person cannot match Bag, and Bag cannot match Person
                is_ex_bag = ex_obj.class_label in ("tas", "backpack", "handbag", "suitcase")
                if (is_ex_bag and det_label == "person") or (ex_obj.class_label == "person" and is_det_bag):
                    continue

                d_cent = float(np.linalg.norm(np.array(ex_obj.centroid, dtype=np.float32) - np.array(det_centroid, dtype=np.float32)))
                d_anch = float(np.linalg.norm(np.array(ex_obj.anchor_centroid, dtype=np.float32) - np.array(det_centroid, dtype=np.float32)))
                best_d = min(d_cent, d_anch)
                box_iou = compute_bbox_iou(getattr(ex_obj, "anchor_bbox", ex_obj.bbox), det_bbox)

                if best_d <= min_existing_dist or (is_det_bag and box_iou >= 0.20):
                    min_existing_dist = best_d
                    matched_existing = ex_obj

            if matched_existing is not None:
                # REUSE EXISTING ACTIVE TRACK ID!
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
                matched_existing.is_occluded = False
                if det_label is not None:
                    matched_existing.class_label = det_label
                if zone_id and zone_id != matched_existing.zone_id:
                    matched_existing.zone_id = zone_id

                # Sticky Stationary logic: continue accumulating dwell time
                is_bag_obj = matched_existing.class_label in ("tas", "backpack", "handbag", "suitcase")
                if is_bag_obj:
                    anchor_d = float(np.linalg.norm(np.array(matched_existing.anchor_centroid, dtype=np.float32) - np.array(matched_existing.centroid, dtype=np.float32)))
                    anchor_box = getattr(matched_existing, "anchor_bbox", matched_existing.bbox)
                    anchor_iou = compute_bbox_iou(anchor_box, det_bbox)
                    is_near_anchor = (anchor_d <= self.anchor_radius_px) or (anchor_iou >= 0.50)

                    if is_near_anchor:
                        matched_existing.moved_confirmation_frames = 0
                        if not matched_existing.is_stationary:
                            matched_existing.is_stationary = True
                            if matched_existing.dwell_duration > 0.0:
                                matched_existing.stationary_start = now - matched_existing.dwell_duration
                    else:
                        if anchor_d > 50.0:
                            matched_existing.moved_confirmation_frames += 1
                        else:
                            matched_existing.moved_confirmation_frames = max(0, matched_existing.moved_confirmation_frames - 1)

                    if matched_existing.is_stationary:
                        if matched_existing.moved_confirmation_frames >= 30:
                            matched_existing.is_stationary = False
                            matched_existing.moved_confirmation_frames = 0
                            matched_existing.last_moved_time = now
                            matched_existing.stationary_start = now
                            matched_existing.anchor_centroid = matched_existing.centroid
                            matched_existing.anchor_bbox = det_bbox
                            matched_existing.dwell_duration = 0.0
                            matched_existing.is_triggered = False
                            matched_existing.alert_sent = False
                            matched_existing.is_warning = False
                            matched_existing.is_pre_alarm = False
                            matched_existing.pre_alarm_alerted = False
                            logger.info(
                                f"[TRACKER] Track ID {matched_existing.track_id} confirmed moved (>50px for 30 frames). Resetting stationary & dwell."
                            )
                        else:
                            if not matched_existing.is_attended:
                                matched_existing.dwell_duration = now - matched_existing.stationary_start
                            else:
                                matched_existing.stationary_start = now - matched_existing.dwell_duration
                            matched_existing.is_warning = (matched_existing.dwell_duration >= matched_existing.dwell_threshold * 0.50)
                            matched_existing.is_pre_alarm = (matched_existing.dwell_duration >= matched_existing.dwell_threshold * 0.85)
                    else:
                        if is_near_anchor:
                            matched_existing.is_stationary = True
                            matched_existing.stationary_start = now
                            matched_existing.dwell_duration = 0.0
                        else:
                            matched_existing.stationary_start = now
                            matched_existing.dwell_duration = 0.0
            else:
                # Check Spatial Memory before registering a brand new ID!
                matched_spatial = self._match_spatial_memory(det_centroid, det_bbox, zone_id, det_label, now)
                if matched_spatial is not None:
                    self._recover_from_spatial_memory(
                        matched_spatial, det_centroid, det_bbox, zone_id, area, det_label, edge_dist, conf, now
                    )
                else:
                    # Spatial Matching Hardening: If there is ANY active bag within 60 px in sterile zone,
                    # FORCE reuse of existing Track ID (do not create a duplicate new ID)
                    if is_det_bag:
                        nearby_active_bag = None
                        for active_obj in self.objects.values():
                            if active_obj.class_label in ("tas", "backpack", "handbag", "suitcase"):
                                d_c = float(np.linalg.norm(np.array(active_obj.centroid, dtype=np.float32) - np.array(det_centroid, dtype=np.float32)))
                                d_a = float(np.linalg.norm(np.array(active_obj.anchor_centroid, dtype=np.float32) - np.array(det_centroid, dtype=np.float32)))
                                b_iou = compute_bbox_iou(active_obj.bbox, det_bbox)
                                if min(d_c, d_a) <= 60.0 or b_iou >= 0.20:
                                    nearby_active_bag = active_obj
                                    break
                        if nearby_active_bag is not None:
                            logger.debug(
                                f"[TRACKER] Suppressed duplicate bag detection within 60px of active Track ID {nearby_active_bag.track_id}"
                            )
                            continue

                    # Only register a new ID if completely outside 60 px of ANY known object
                    self._register(det_centroid, det_bbox, zone_id, area, now, label=det_label, edge_dist=edge_dist, conf=conf)

        # 5. Universal owner proximity across all zones
        self.evaluate_owner_proximity(now)

        # Anti-memory leak: purge tracks unobserved for > max_disappeared_sec (4.5s) or > max_age_frames (45 frames)
        purged = self._purge_stale_objects(now)

        return list(self.objects.values()), purged

    def _purge_stale_objects(self, now: float) -> List[TrackedObject]:
        """Deregister objects that disappeared longer than max_disappeared_sec and max_age_frames.
        Stationary bags are archived into spatial_memory cache before purge."""
        stale_ids = []
        for obj_id, obj in self.objects.items():
            is_bag = obj.class_label in ("tas", "backpack", "handbag", "suitcase")
            if is_bag and obj.is_stationary:
                # Stationary Bag Latching: hold confirmed stationary bag for up to 600 frames (~45s)
                max_frames = max(getattr(self, "stationary_max_age_frames", 600), self.max_age_frames)
                max_sec = max(getattr(self, "stationary_max_disappeared_sec", 45.0), self.max_disappeared_sec)
            elif obj.is_stationary or is_bag:
                extra_frames = 150 if getattr(obj, "is_occluded", False) else 0
                max_frames = max(150 + extra_frames, self.max_age_frames)
                max_sec = max(12.0, self.max_disappeared_sec)
            else:
                max_frames = self.max_age_frames
                max_sec = self.max_disappeared_sec

            if (now - obj.last_seen) > max_sec and obj.missed_frames > max_frames:
                stale_ids.append(obj_id)

        purged_objects: List[TrackedObject] = []
        for obj_id in stale_ids:
            obj = self.objects[obj_id]
            is_bag = obj.class_label in ("tas", "backpack", "handbag", "suitcase")
            if is_bag and (obj.is_stationary or obj.dwell_duration > 0.0):
                # Archive stationary bag to Spatial Memory cache
                self.spatial_memory[obj_id] = SpatialMemoryEntry(
                    track_id=obj.track_id,
                    class_label=obj.class_label,
                    zone_id=obj.zone_id,
                    anchor_centroid=obj.anchor_centroid,
                    last_centroid=obj.centroid,
                    last_bbox=obj.bbox,
                    first_seen=obj.first_seen,
                    stationary_start=obj.stationary_start,
                    accumulated_dwell=obj.dwell_duration,
                    last_seen=obj.last_seen,
                    deregistered_at=now,
                    occluded_by_person=getattr(obj, "is_occluded", False),
                    contour_area=obj.contour_area,
                    confidence=obj.confidence,
                    dwell_threshold=obj.dwell_threshold,
                    last_owner_info=getattr(obj, "last_owner_info", None),
                    anchor_bbox=getattr(obj, "anchor_bbox", obj.bbox),
                )
            purged_objects.append(obj)
            del self.objects[obj_id]

        # Purge expired entries from spatial memory cache (TTL exceeded)
        expired_cache_ids = [
            cid for cid, entry in self.spatial_memory.items()
            if (now - entry.deregistered_at) > self.spatial_memory_ttl_sec
        ]
        for cid in expired_cache_ids:
            del self.spatial_memory[cid]

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
                if getattr(bag, "is_occluded", False):
                    # Occluded by passerby or person in front -> freeze/pause dwell, do not zero
                    bag.is_attended = False
                    bag.attended_start = None
                    bag.attended_duration = 0.0
                    bag.stationary_start = now - bag.dwell_duration
                else:
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
