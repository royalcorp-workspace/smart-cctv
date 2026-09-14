"""Tripwire (Line Crossing Detection) Engine for Smart CCTV 2.0."""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

from engine.logger import logger


@dataclass
class TripwireLine:
    """Virtual tripwire line representation."""
    line_id: str
    name: str
    p1: Tuple[int, int]
    p2: Tuple[int, int]
    direction: str = "both"  # "both", "A_to_B", "B_to_A"
    target_classes: List[str] = field(default_factory=lambda: ["person", "car", "bus", "truck"])


@dataclass
class LineCrossingEvent:
    """Event emitted when an object crosses a virtual tripwire."""
    track: object
    line_id: str
    line_name: str
    direction: str
    timestamp: float


def cross_product_2d(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Compute 2D cross product of vectors (B - A) and (C - A).

    Positive: C lies on the left side of vector AB (Counter-Clockwise).
    Negative: C lies on the right side of vector AB (Clockwise).
    Zero: Collinear.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Determine if line segment p1-p2 intersects line segment p3-p4.

    Uses 2D orientation test (Counter-Clockwise sign comparison).
    """
    def ccw(a, b, c) -> bool:
        return cross_product_2d(a, b, c) > 0.0

    cp1 = cross_product_2d(p3, p4, p1)
    cp2 = cross_product_2d(p3, p4, p2)
    cp3 = cross_product_2d(p1, p2, p3)
    cp4 = cross_product_2d(p1, p2, p4)

    # Segments intersect if endpoints of each segment are on opposite sides of the other
    straddles_1 = (cp1 > 0 and cp2 < 0) or (cp1 < 0 and cp2 > 0)
    straddles_2 = (cp3 > 0 and cp4 < 0) or (cp3 < 0 and cp4 > 0)

    if straddles_1 and straddles_2:
        return True

    # Handle boundary touching (one endpoint lies exactly on the line segment)
    def on_segment(p, a, b):
        return (
            min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
            and abs(cross_product_2d(a, b, p)) < 1e-5
        )

    if on_segment(p1, p3, p4) or on_segment(p2, p3, p4) or on_segment(p3, p1, p2) or on_segment(p4, p1, p2):
        return True

    return False


def compute_line_box_overlap(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    bbox: Union[Tuple[int, int, int, int], List[int]],
) -> float:
    """Compute parametric overlap fraction [0.0, 1.0] of line segment p1-p2 inside bbox.

    Uses Liang-Barsky parametric line clipping algorithm.
    Returns the fraction of the line segment length that lies inside the axis-aligned bounding box.
    """
    if not bbox or len(bbox) < 4:
        return 0.0

    x_min = float(bbox[0])
    y_min = float(bbox[1])
    x_max = float(bbox[0] + bbox[2])
    y_max = float(bbox[1] + bbox[3])

    dx = float(p2[0] - p1[0])
    dy = float(p2[1] - p1[1])

    # Zero length segment handling
    if dx == 0.0 and dy == 0.0:
        if x_min <= p1[0] <= x_max and y_min <= p1[1] <= y_max:
            return 1.0
        return 0.0

    t0 = 0.0
    t1 = 1.0

    p = [-dx, dx, -dy, dy]
    q = [p1[0] - x_min, x_max - p1[0], p1[1] - y_min, y_max - p1[1]]

    for pk, qk in zip(p, q):
        if pk == 0.0:
            if qk < 0.0:
                # Parallel and outside
                return 0.0
        else:
            r = qk / pk
            if pk < 0.0:
                if r > t1:
                    return 0.0
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return 0.0
                if r < t1:
                    t1 = r

    if t0 > t1:
        return 0.0

    overlap = max(0.0, min(1.0, t1 - t0))
    return overlap


class TripwireEngine:
    """Real-time tripwire line crossing detector with trajectory tracking, occlusion guard, and cooldown."""

    def __init__(
        self,
        lines: Optional[Dict[str, dict]] = None,
        cooldown_sec: float = 5.0,
        occlusion_overlap_threshold: float = 0.50,
        occlusion_duration_sec: float = 15.0,
        min_track_frames: int = 4,
        min_confidence: float = 0.40,
        min_bbox_area: float = 400.0,
        max_step_px: float = 65.0,
    ) -> None:
        self.lines: Dict[str, TripwireLine] = {}
        self.cooldown_sec: float = cooldown_sec
        self.occlusion_overlap_threshold: float = occlusion_overlap_threshold
        self.occlusion_duration_sec: float = occlusion_duration_sec
        self.min_track_frames: int = min_track_frames
        self.min_confidence: float = min_confidence
        self.min_bbox_area: float = min_bbox_area
        self.max_step_px: float = max_step_px
        self._prev_positions: Dict[int, Tuple[float, float]] = {}
        self._last_triggered: Dict[Tuple[int, str], float] = {}
        self._recent_crossings: Dict[str, float] = {}  # line_id -> timestamp for visual flash
        self._occlusion_start: Dict[Tuple[str, int], float] = {}  # (line_id, track_id) -> start timestamp
        self._occluded_lines: Set[str] = set()
        self._vehicle_anchors: Dict[int, Tuple[float, float]] = {}  # track_id -> initial stationary centroid

        if lines:
            self.load_lines(lines)

    def load_lines(self, lines_dict: Dict[str, dict]) -> None:
        """Load or update tripwire lines from configuration."""
        self.lines.clear()
        for line_id, l_cfg in lines_dict.items():
            if not isinstance(l_cfg, dict):
                continue
            p1_raw = l_cfg.get("p1")
            p2_raw = l_cfg.get("p2")
            if not p1_raw or not p2_raw or len(p1_raw) < 2 or len(p2_raw) < 2:
                continue

            p1 = (int(round(p1_raw[0])), int(round(p1_raw[1])))
            p2 = (int(round(p2_raw[0])), int(round(p2_raw[1])))
            name = l_cfg.get("name", line_id)
            direction = str(l_cfg.get("direction", "both")).lower()
            if direction not in ("both", "a_to_b", "b_to_a"):
                direction = "both"
            target_classes = l_cfg.get("target_classes", ["person", "car", "bus", "truck"])

            self.lines[line_id] = TripwireLine(
                line_id=line_id,
                name=name,
                p1=p1,
                p2=p2,
                direction=direction,
                target_classes=target_classes,
            )
        logger.info(f"[TripwireEngine] Loaded {len(self.lines)} tripwire lines.")

    def has_lines(self) -> bool:
        """Return True if at least one active tripwire line is configured."""
        return len(self.lines) > 0

    def get_recent_flash(self, line_id: str, now: Optional[float] = None) -> bool:
        """Return True if line was crossed within 2.0s (for visual pulse/flash)."""
        curr_time = now if now is not None else time.time()
        last_t = self._recent_crossings.get(line_id, 0.0)
        return (curr_time - last_t) <= 2.0

    def get_flashing_lines(self, now: Optional[float] = None) -> set:
        """Return set of line_ids that were crossed within 2.0s."""
        curr_time = now if now is not None else time.time()
        return {
            line_id for line_id, last_t in self._recent_crossings.items()
            if (curr_time - last_t) <= 2.0
        }

    def get_occluded_lines(self) -> Set[str]:
        """Return set of line_ids currently occluded by stationary vehicles."""
        return set(self._occluded_lines)

    def update(
        self,
        active_tracks: list,
        timestamp: Optional[float] = None,
    ) -> List[LineCrossingEvent]:
        """Evaluate active tracked objects against virtual tripwires.

        Incorporates Liang-Barsky line occlusion guard and foot ground filtering.
        Returns list of LineCrossingEvent for objects crossing lines in valid direction.
        """
        if not self.lines:
            return []

        now = timestamp if timestamp is not None else time.time()

        if not active_tracks:
            # When active_tracks is empty (all objects have left the frame),
            # clear occlusion states, vehicle anchors, and previous track positions
            self._occlusion_start.clear()
            self._occluded_lines.clear()
            self._vehicle_anchors.clear()
            self._prev_positions.clear()
            return []

        events: List[LineCrossingEvent] = []
        current_track_ids = set()

        # 1. Identify stationary trucks and buses for line occlusion evaluation and foot ground filtering
        stationary_vehicles = []
        active_veh_ids = set()
        for track in active_tracks:
            t_id = getattr(track, "track_id", None)
            if t_id is not None:
                current_track_ids.add(t_id)

            c_label = getattr(track, "class_label", "")
            if c_label in ("truck", "bus"):
                if t_id is not None:
                    active_veh_ids.add(t_id)
                is_stat = getattr(track, "is_stationary", False)
                if is_stat and hasattr(track, "bbox") and track.bbox and len(track.bbox) == 4:
                    # Verify spatial movement threshold (shift <= 15.0 px)
                    t_cent = getattr(
                        track,
                        "centroid",
                        (track.bbox[0] + track.bbox[2] / 2.0, track.bbox[1] + track.bbox[3] / 2.0),
                    )
                    anchor = self._vehicle_anchors.setdefault(t_id, t_cent)
                    shift = math.hypot(t_cent[0] - anchor[0], t_cent[1] - anchor[1])
                    if shift <= 15.0:
                        stationary_vehicles.append(track)
                    else:
                        # Vehicle shifted > 15px, update anchor and consider non-stationary
                        self._vehicle_anchors[t_id] = t_cent

        # Purge stale vehicle anchors
        stale_anchors = [vid for vid in self._vehicle_anchors if vid not in active_veh_ids]
        for vid in stale_anchors:
            del self._vehicle_anchors[vid]

        # 2. Evaluate Line Occlusion States
        current_occluding_pairs = set()
        for line_id, line in self.lines.items():
            l_p1 = (float(line.p1[0]), float(line.p1[1]))
            l_p2 = (float(line.p2[0]), float(line.p2[1]))
            for veh in stationary_vehicles:
                v_id = getattr(veh, "track_id", None)
                if v_id is None:
                    continue
                overlap = compute_line_box_overlap(l_p1, l_p2, veh.bbox)
                if overlap >= self.occlusion_overlap_threshold:
                    current_occluding_pairs.add((line_id, v_id))

        # Register or maintain occlusion start timers
        for pair in current_occluding_pairs:
            if pair not in self._occlusion_start:
                self._occlusion_start[pair] = now

        # Purge pairs that are no longer occluding (vehicle moved > 15px, left frame, or overlap < 50%)
        stale_pairs = [pair for pair in self._occlusion_start if pair not in current_occluding_pairs]
        for pair in stale_pairs:
            del self._occlusion_start[pair]

        # Update active occluded lines (overlap >= 50% for >= 15.0s)
        new_occluded_lines = set()
        for (l_id, _), start_t in self._occlusion_start.items():
            if (now - start_t) >= self.occlusion_duration_sec:
                new_occluded_lines.add(l_id)
        self._occluded_lines = new_occluded_lines

        # 3. Evaluate Crossings
        for track in active_tracks:
            track_id = getattr(track, "track_id", None)
            if track_id is None:
                continue

            class_label = getattr(track, "class_label", "object")

            # Reference point: ground contact point (feet) for person, bottom wheel contact for vehicles
            # Strictly ignore centroid, head, shoulders, upper bounding box to prevent false alarm
            # when person is leaning or waving arms across the virtual line.
            if class_label == "person" and hasattr(track, "bbox") and track.bbox and len(track.bbox) == 4:
                bx, by, bw, bh = track.bbox
                cx = bx + bw // 2
                foot_y = (by + bh) - 2
                curr_pt = (float(cx), float(foot_y))
            elif class_label in ("car", "bus", "truck") and hasattr(track, "bbox") and track.bbox and len(track.bbox) == 4:
                bx, by, bw, bh = track.bbox
                cx = bx + bw // 2
                wheel_y = (by + bh) - 5
                curr_pt = (float(cx), float(wheel_y))
            elif hasattr(track, "bbox") and track.bbox and len(track.bbox) == 4:
                bx, by, bw, bh = track.bbox
                cx = bx + bw // 2
                ground_y = (by + bh) - 2
                curr_pt = (float(cx), float(ground_y))
            else:
                curr_pt = (float(track.centroid[0]), float(track.centroid[1]))

            prev_pt = self._prev_positions.get(track_id)
            self._prev_positions[track_id] = curr_pt

            # Skip first frame for newly registered track (needs at least 2 points to form a vector)
            if prev_pt is None:
                continue

            # Skip if object is stationary / hasn't moved
            if prev_pt[0] == curr_pt[0] and prev_pt[1] == curr_pt[1]:
                continue

            # Anti-Ghost Gate 1: Track must be actively detected in the current frame (not coasting)
            missed = getattr(track, "missed_frames", 0)
            is_active = getattr(track, "is_active_this_frame", True)
            if missed > 0 or not is_active:
                continue

            # Anti-Ghost Gate 2: Track maturity check (must be confirmed across multiple frames)
            frame_cnt = getattr(track, "frame_count", 1)
            if frame_cnt < self.min_track_frames:
                continue

            # Anti-Ghost Gate 3: Pedestrian Confidence & Micro-Bbox Guard
            if class_label == "person":
                track_conf = getattr(track, "confidence", 1.0)
                if track_conf < self.min_confidence:
                    continue

                if hasattr(track, "bbox") and track.bbox and len(track.bbox) == 4:
                    box_area = float(track.bbox[2] * track.bbox[3])
                    if box_area < self.min_bbox_area:
                        continue

            # Anti-Ghost Gate 4: Bound maximum single-frame step displacement (suppress tracker ID/noise jumps)
            step_dx = curr_pt[0] - prev_pt[0]
            step_dy = curr_pt[1] - prev_pt[1]
            step_dist = math.hypot(step_dx, step_dy)
            if step_dist > self.max_step_px:
                logger.debug(
                    f"[TripwireEngine] Suppressed jump crossing for Track #{track_id} ({class_label}): "
                    f"step={step_dist:.1f}px > max={self.max_step_px}px"
                )
                continue

            # Foot Ground Filter: ignore pedestrian crossings if foot_pt falls within
            # the upper/middle body of a stationary truck (by <= foot_y <= by + 0.85 * bh)
            # and (bx <= foot_x <= bx + bw)
            if class_label == "person" and stationary_vehicles:
                is_phantom = False
                for veh in stationary_vehicles:
                    v_bx, v_by, v_bw, v_bh = veh.bbox
                    if (v_bx <= curr_pt[0] <= v_bx + v_bw) and (v_by <= curr_pt[1] <= v_by + 0.85 * v_bh):
                        is_phantom = True
                        break
                if is_phantom:
                    logger.debug(
                        f"[TripwireEngine] Suppressed phantom pedestrian #{track_id}: "
                        f"foot within stationary vehicle body."
                    )
                    continue

            # Evaluate against all tripwire lines
            for line_id, line in self.lines.items():
                # Suppress crossing trigger events for lines currently in OCCLUDED state
                if line_id in self._occluded_lines:
                    continue

                # Filter by target class
                if line.target_classes and class_label not in line.target_classes:
                    continue

                # Anti-flapping: check cooldown per (track_id, line_id)
                last_trig = self._last_triggered.get((track_id, line_id), 0.0)
                if (now - last_trig) < self.cooldown_sec:
                    continue

                # Check 2D segment intersection
                l_p1 = (float(line.p1[0]), float(line.p1[1]))
                l_p2 = (float(line.p2[0]), float(line.p2[1]))

                if segments_intersect(prev_pt, curr_pt, l_p1, l_p2):
                    # Evaluate crossing direction relative to vector P1 -> P2
                    s_prev = cross_product_2d(l_p1, l_p2, prev_pt)
                    s_curr = cross_product_2d(l_p1, l_p2, curr_pt)

                    if s_prev > 0 and s_curr <= 0:
                        detected_dir = "A_to_B"
                    elif s_prev < 0 and s_curr >= 0:
                        detected_dir = "B_to_A"
                    else:
                        detected_dir = "both"

                    # Check direction compliance
                    dir_match = (
                        line.direction == "both"
                        or line.direction.lower() == detected_dir.lower()
                    )

                    if dir_match:
                        self._last_triggered[(track_id, line_id)] = now
                        self._recent_crossings[line_id] = now
                        event = LineCrossingEvent(
                            track=track,
                            line_id=line_id,
                            line_name=line.name,
                            direction=detected_dir,
                            timestamp=now,
                        )
                        events.append(event)
                        logger.warning(
                            f"[TripwireEngine] Line Crossing Detected! Line='{line.name}' ({line_id}), "
                            f"Track #{track_id} ({class_label}), Dir={detected_dir}"
                        )

        # Purge stale track history
        stale_tracks = set(self._prev_positions.keys()) - current_track_ids
        for st in stale_tracks:
            del self._prev_positions[st]

        # Purge stale cooldown entries older than 30s
        stale_triggers = [
            k for k, v in self._last_triggered.items() if (now - v) > 30.0
        ]
        for sk in stale_triggers:
            del self._last_triggered[sk]

        return events

