"""Tripwire (Line Crossing Detection) Engine for Smart CCTV 2.0."""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple, Union

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


class TripwireEngine:
    """Real-time tripwire line crossing detector with trajectory tracking and cooldown."""

    def __init__(
        self,
        lines: Optional[Dict[str, dict]] = None,
        cooldown_sec: float = 5.0,
    ) -> None:
        self.lines: Dict[str, TripwireLine] = {}
        self.cooldown_sec: float = cooldown_sec
        self._prev_positions: Dict[int, Tuple[float, float]] = {}
        self._last_triggered: Dict[Tuple[int, str], float] = {}
        self._recent_crossings: Dict[str, float] = {}  # line_id -> timestamp for visual flash

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

    def update(
        self,
        active_tracks: list,
        timestamp: Optional[float] = None,
    ) -> List[LineCrossingEvent]:
        """Evaluate active tracked objects against virtual tripwires.

        Returns list of LineCrossingEvent for objects crossing lines in valid direction.
        """
        if not self.lines or not active_tracks:
            return []

        now = timestamp if timestamp is not None else time.time()
        events: List[LineCrossingEvent] = []
        current_track_ids = set()

        for track in active_tracks:
            track_id = getattr(track, "track_id", None)
            if track_id is None:
                continue
            current_track_ids.add(track_id)

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

            # Evaluate against all tripwire lines
            for line_id, line in self.lines.items():
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
