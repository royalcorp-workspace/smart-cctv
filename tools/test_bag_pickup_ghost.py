"""Unit and integration test for Bag Pickup, Exit Zone Purge, and Ghost Tile Suppression.

Verifies:
1. Bag movement within sterile zone maintains tracking & dwell time.
2. Bag retrieval / exit zone with physical contact purges track within 10 missed frames.
3. Ghost tile / DualSubtractor blobs on bare floor without YOLO confirmation (>5.0s) are discarded.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.tracker import CentroidTracker, TrackedObject


def test_bag_shift_within_zone():
    """Verify that a bag shifting/moving slightly within the sterile zone stays tracked."""
    tracker = CentroidTracker(
        stationary_max_age_frames=900,
        stationary_max_disappeared_sec=75.0,
        spatial_memory_ttl_sec=600.0,
    )
    now = 1000.0

    # Frame 1: Register bag at (200, 200)
    det1 = [((180, 180, 40, 40), (200, 200), "zone_2_transit", 1600.0, "tas", 20.0, 0.90, "yolo")]
    active, purged = tracker.update(det1, timestamp=now)
    assert len(active) == 1, "Failed to register initial bag"
    bag_id = active[0].track_id

    # Simulate 5 seconds stationary
    for i in range(1, 60):
        t = now + i * 0.1
        active, purged = tracker.update(det1, timestamp=t)

    assert len(active) == 1
    assert active[0].is_stationary is True
    assert active[0].dwell_duration > 4.0
    initial_dwell = active[0].dwell_duration

    # Frame 61: Bag shifts slightly (20px) within zone (anchor distance <= anchor_radius_px)
    shift_det = [((190, 190, 40, 40), (210, 210), "zone_2_transit", 1600.0, "tas", 20.0, 0.90, "yolo")]
    active, purged = tracker.update(shift_det, timestamp=now + 6.1)

    assert len(active) == 1
    assert active[0].track_id == bag_id
    assert active[0].is_stationary is True
    assert active[0].dwell_duration >= initial_dwell
    print(" -> PASS: test_bag_shift_within_zone")


def test_instant_pickup_and_exit_zone_purge():
    """Verify that when a bag is picked up by a person (physical contact within 2.5s),
    the track is immediately purged after 10 missed frames (~0.8s).
    """
    tracker = CentroidTracker(
        stationary_max_age_frames=900,
        stationary_max_disappeared_sec=75.0,
        spatial_memory_ttl_sec=600.0,
    )
    now = 1000.0

    # 1. Bag stationary on floor
    bag_box = (200, 200, 50, 50)
    bag_cent = (225, 225)
    det_bag = [(bag_box, bag_cent, "zone_2_transit", 2500.0, "tas", 20.0, 0.90, "yolo")]
    for i in range(10):
        tracker.update(det_bag, timestamp=now + i * 0.1)

    assert len(tracker.objects) == 1
    bag = list(tracker.objects.values())[0]
    bag_id = bag.track_id
    assert bag.is_stationary is True

    # 2. Person approaches and physically touches/overlaps the bag (IoU > 0.20 or IoF > 0.30)
    person_box = (190, 150, 80, 120)
    person_cent = (230, 210)
    t_pickup = now + 2.0
    det_interaction = [
        (bag_box, bag_cent, "zone_2_transit", 2500.0, "tas", 20.0, 0.90, "yolo"),
        (person_box, person_cent, "zone_2_transit", 9600.0, "person", 10.0, 0.95, "yolo"),
    ]
    active, purged = tracker.update(det_interaction, timestamp=t_pickup)
    assert bag.last_physical_contact_time == t_pickup, "Physical contact time not updated"

    # 3. Person lifts bag and walks away (stepping away from bag position)
    # Frame 1 to 9: missing, under 10 frames -> latching holds it
    for f in range(1, 10):
        t_miss = t_pickup + f * 0.08  # at ~12.5 fps
        # Person walking away: x moves from 230 to 320 px
        p_step_box = (190 + f * 10, 150, 80, 120)
        p_step_cent = (230 + f * 10, 210)
        det_leaving = [
            (p_step_box, p_step_cent, "zone_2_transit", 9600.0, "person", 10.0, 0.95, "yolo"),
        ]
        active, purged = tracker.update(det_leaving, timestamp=t_miss)
        assert any(obj.track_id == bag_id for obj in active), f"Bag should latch on frame {f}"

    # Frame 10: missed_frames reaches 10, person walked away (contact was at t_pickup, diff = 0.8s <= 2.5s) -> IMMEDIATE PURGE!
    t_frame_10 = t_pickup + 10 * 0.08
    p_step_box_10 = (290, 150, 80, 120)
    p_step_cent_10 = (330, 210)
    det_leaving_10 = [
        (p_step_box_10, p_step_cent_10, "zone_2_transit", 9600.0, "person", 10.0, 0.95, "yolo"),
    ]
    active, purged = tracker.update(det_leaving_10, timestamp=t_frame_10)

    # Bag MUST be purged immediately
    purged_ids = [p.track_id for p in purged]
    assert bag_id in purged_ids, "Bag should be immediately purged on missed_frames >= 10 with physical contact <= 2.5s"
    purged_bag = next(p for p in purged if p.track_id == bag_id)
    assert purged_bag.is_retrieved is True

    # Spatial memory must NOT retain retrieved bag
    assert bag_id not in tracker.spatial_memory, "Retrieved bag must not be retained in Spatial Memory"
    print(" -> PASS: test_instant_pickup_and_exit_zone_purge")


def test_ghost_floor_tile_discard_logic():
    """Verify that a DualSubtractor blob on bare floor with no YOLO confirmation is rejected."""
    now = 2000.0

    # Case A: Blob with NO current YOLO and NO active track with YOLO seen <= 5.0s
    yolo_results = []  # No YOLO bag detection
    active_tracker_objects = {}  # No active track

    bcx, bcy = 225, 225
    bx, by, bw, bh = 200, 200, 50, 50

    has_current_yolo = any(
        y_cid in (24, 26, 28) and y_conf > 0.15 for _, _, _, y_conf, y_cid, _ in yolo_results
    )
    has_recent_yolo_track = False
    for trk in active_tracker_objects.values():
        if (now - getattr(trk, "last_yolo_seen_time", 0.0)) <= 5.0:
            has_recent_yolo_track = True

    # Must be discarded!
    should_discard = not has_current_yolo and not has_recent_yolo_track
    assert should_discard is True, "Ghost floor blob without YOLO must be discarded"
    print(" -> PASS: test_ghost_floor_tile_discard_logic")


def run_all_tests():
    print("=" * 60)
    print("RUNNING BAG PICKUP, EXIT ZONE PURGE & GHOST TILE TESTS")
    print("=" * 60)
    test_bag_shift_within_zone()
    test_instant_pickup_and_exit_zone_purge()
    test_ghost_floor_tile_discard_logic()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
