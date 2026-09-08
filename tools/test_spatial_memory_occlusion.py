"""Focused test suite for Spatial Memory and Occlusion Recovery in Smart CCTV 2.0."""

import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.tracker import CentroidTracker, TrackedObject, SpatialMemoryEntry


def test_short_occlusion_person_passerby() -> bool:
    """Test 5-10 seconds occlusion by passing person.
    
    Verifies that:
    1. Dwell time is frozen/held during occlusion (not reset to 0).
    2. Track ID is preserved when person moves away.
    3. is_attended does not zero the dwell timer during occlusion.
    """
    print("\n--- [SCENARIO 1] 5-10s Occlusion by Passing Person ---")
    tracker = CentroidTracker(
        max_distance_px=50.0,
        anchor_radius_px=40.0,
        spatial_memory_ttl_sec=180.0,
        spatial_match_distance_px=45.0,
        stationary_max_age_frames=600,
        stationary_max_disappeared_sec=45.0,
    )

    t0 = 500.0
    bag_bbox = (200, 200, 50, 50)
    bag_centroid = (225, 225)
    zone_id = "zone_2_transit"

    # Step 1: Bag stationary for 25 seconds
    tracker.update([(bag_bbox, bag_centroid, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=t0)
    active, _ = tracker.update([(bag_bbox, bag_centroid, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=t0 + 25.0)
    assert len(active) == 1
    assert active[0].track_id == 1
    assert abs(active[0].dwell_duration - 25.0) < 0.2
    print(f" - Initial bag state: Track ID={active[0].track_id}, Dwell={active[0].dwell_duration:.1f}s")

    # Step 2: Person walks over and blocks the bag for 8 seconds (t = 525s to t = 533s)
    # The bag is completely occluded: only person bbox is detected
    person_bbox = (200, 150, 60, 140)  # Covers bag at (200, 200)
    person_centroid = (230, 220)
    person_dets = [(person_bbox, person_centroid, zone_id, 8400.0, "person", 20.0, 0.95)]

    for step in range(1, 9):
        active, _ = tracker.update(person_dets, timestamp=t0 + 25.0 + step)
        bag = tracker.objects.get(1)
        assert bag is not None, f"Bag track 1 must remain active during grace period (step {step})"
        assert bag.is_occluded is True, f"Bag must have is_occluded=True (step {step})"
        assert bag.is_attended is False, f"Occluded bag must not be marked attended (step {step})"
        assert abs(bag.dwell_duration - 25.0) < 0.2, f"Dwell must remain held at 25s, got {bag.dwell_duration}"

    print(" - During 8s occlusion: Bag is_occluded=True, dwell held strictly at 25.0s, is_attended=False.")

    # Step 3: Person walks away, bag is visible again at same location
    active, _ = tracker.update([(bag_bbox, (226, 225), zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=t0 + 34.0)
    bag_resumed = tracker.objects.get(1)
    assert bag_resumed is not None, "Track ID 1 must be preserved!"
    assert bag_resumed.is_active_this_frame is True, "Bag must be active in this frame"
    assert bag_resumed.is_occluded is False, "is_occluded must be cleared when visible"
    assert bag_resumed.dwell_duration >= 25.0, f"Dwell must not reset to 0! Got {bag_resumed.dwell_duration}"
    print(f" - After occlusion: Track ID={bag_resumed.track_id} preserved, Dwell smoothly resumed at {bag_resumed.dwell_duration:.1f}s.")
    print(" -> PASS: Scenario 1 succeeded.")
    return True


def test_extended_occlusion_buffer_purge_recovery() -> bool:
    """Test extended occlusion where bag is latched up to 600 frames, gets purged after 600 frames, and recovers via Spatial Memory."""
    print("\n--- [SCENARIO 2] Extended Occlusion & Spatial Memory Cache Recovery ---")
    tracker = CentroidTracker(
        max_distance_px=50.0,
        anchor_radius_px=40.0,
        spatial_memory_ttl_sec=180.0,
        spatial_match_distance_px=45.0,
        stationary_max_age_frames=600,
        stationary_max_disappeared_sec=45.0,
    )

    t0 = 1000.0
    bag_bbox = (400, 300, 70, 70)
    bag_centroid = (435, 335)
    zone_id = "zone_2_transit"

    # Step 1: Bag stationary for 40 seconds
    tracker.update([(bag_bbox, bag_centroid, zone_id, 4900.0, "backpack", 20.0, 0.9)], timestamp=t0)
    active, _ = tracker.update([(bag_bbox, bag_centroid, zone_id, 4900.0, "backpack", 20.0, 0.9)], timestamp=t0 + 40.0)
    assert active[0].track_id == 1
    assert abs(active[0].dwell_duration - 40.0) < 0.2
    print(f" - Bag initial state: Track ID={active[0].track_id}, Dwell={active[0].dwell_duration:.1f}s")

    # Step 2a: Latching period (180 frames / 18s) - bag MUST NOT be purged and MUST stay rendered
    for f in range(180):
        tracker.update([], timestamp=t0 + 40.0 + f * 0.1)
    assert 1 in tracker.objects, "Bag MUST remain latched in active tracker during 600-frame latching window"
    assert tracker.objects[1].should_render is True, "Bag must stay renderable (should_render=True) to prevent visual flapping"
    print(" - Stationary Latching check: Bag latched in active tracking and rendered during temporary drop.")

    # Step 2b: Extended disappearance exceeding 600 frames (> 45s) -> purged to spatial_memory
    for f in range(180, 620):
        tracker.update([], timestamp=t0 + 40.0 + f * 0.1)

    assert 1 not in tracker.objects, "Bag must be purged from active objects after 620 frames (> 45s)"
    assert 1 in tracker.spatial_memory, "Bag MUST be preserved in spatial_memory cache"
    cached = tracker.spatial_memory[1]
    assert abs(cached.accumulated_dwell - 40.0) < 0.2, f"Cached dwell must be ~40s, got {cached.accumulated_dwell}"
    print(f" - After >45s absence: ID=1 archived in spatial_memory with preserved dwell={cached.accumulated_dwell:.1f}s.")

    # Step 3: Bag re-emerges (within TTL 180s)
    t_reappear = t0 + 40.0 + 80.0
    active, _ = tracker.update([(bag_bbox, (436, 335), zone_id, 4900.0, "backpack", 20.0, 0.89)], timestamp=t_reappear)
    assert len(active) == 1
    recovered = active[0]
    assert recovered.track_id == 1, f"Track ID must be RECOVERED as 1, got {recovered.track_id}"
    assert abs(recovered.dwell_duration - 40.0) < 0.5, f"Dwell must resume from ~40s, got {recovered.dwell_duration}"
    assert 1 not in tracker.spatial_memory, "Recovered track must be removed from spatial_memory"
    print(f" - Re-identification: Track ID={recovered.track_id} revived from Spatial Memory, Dwell={recovered.dwell_duration:.1f}s.")
    print(" -> PASS: Scenario 2 succeeded.")
    return True


def test_moved_bag_vs_spatial_memory() -> bool:
    """Test that a bag genuinely moved to a distant location (> 45px) does not falsely link to old cache."""
    print("\n--- [SCENARIO 3] Genuinely Moved Bag (> 45px) Does Not False-Match ---")
    tracker = CentroidTracker(
        max_distance_px=50.0,
        anchor_radius_px=40.0,
        spatial_memory_ttl_sec=180.0,
        spatial_match_distance_px=45.0,
        stationary_max_age_frames=600,
        stationary_max_disappeared_sec=45.0,
    )

    t0 = 2000.0
    bag_bbox1 = (100, 100, 50, 50)
    bag_centroid1 = (125, 125)
    zone_id = "zone_2_transit"

    # Bag 1 at (125, 125) for 30s
    tracker.update([(bag_bbox1, bag_centroid1, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=t0)
    tracker.update([(bag_bbox1, bag_centroid1, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=t0 + 30.0)

    # Purge bag 1 to spatial memory (exceeding 600 frames / 45s)
    for f in range(620):
        tracker.update([], timestamp=t0 + 30.0 + f * 0.1)

    assert 1 in tracker.spatial_memory

    # A different bag or the same bag moved 150px away to (280, 280)
    bag_bbox2 = (255, 255, 50, 50)
    bag_centroid2 = (280, 280)
    active, _ = tracker.update([(bag_bbox2, bag_centroid2, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=t0 + 100.0)

    assert len(active) == 1
    new_bag = active[0]
    # Must NOT link to ID 1 because distance (125, 125) to (280, 280) is ~219px (> 45px)
    assert new_bag.track_id != 1, f"Moved bag must get a NEW ID, but got {new_bag.track_id}"
    assert new_bag.track_id == 2
    assert new_bag.dwell_duration == 0.0, "New bag position must start dwell from 0.0s"
    print(f" - Distant bag check: Position (280, 280) assigned new Track ID={new_bag.track_id}, Dwell=0.0s.")
    print(" -> PASS: Scenario 3 succeeded.")
    return True


def test_ttl_expiration_cleanup() -> bool:
    """Test that spatial memory entries strictly expire and purge after 180s TTL."""
    print("\n--- [SCENARIO 4] Spatial Memory TTL (180s) Expiration ---")
    tracker = CentroidTracker(
        max_distance_px=50.0,
        anchor_radius_px=40.0,
        spatial_memory_ttl_sec=180.0,
        spatial_match_distance_px=45.0,
        stationary_max_age_frames=600,
        stationary_max_disappeared_sec=45.0,
    )

    t0 = 3000.0
    bag_bbox = (150, 150, 50, 50)
    bag_centroid = (175, 175)
    zone_id = "zone_2_transit"

    # Register and purge bag (exceeding 600 frames / 45s)
    tracker.update([(bag_bbox, bag_centroid, zone_id, 2500.0, "tas", 20.0, 0.9)], timestamp=t0)
    for f in range(620):
        tracker.update([], timestamp=t0 + 5.0 + f * 0.1)

    assert 1 in tracker.spatial_memory, "Should be in cache right after purge"
    dereg_time = tracker.spatial_memory[1].deregistered_at

    # Fast forward 195 seconds (> 180s TTL)
    tracker.update([], timestamp=dereg_time + 195.0)
    assert 1 not in tracker.spatial_memory, "Must be cleanly purged after exceeding TTL 180s"
    assert len(tracker.spatial_memory) == 0
    print(" - TTL expiration check: Entry cleanly purged after 195s (> 180s TTL).")
    print(" -> PASS: Scenario 4 succeeded.")
    return True


def run_all_scenarios():
    print("=" * 65)
    print("RUNNING DEDICATED SPATIAL MEMORY & OCCLUSION RECOVERY TEST SUITE")
    print("=" * 65)

    assert test_short_occlusion_person_passerby()
    assert test_extended_occlusion_buffer_purge_recovery()
    assert test_moved_bag_vs_spatial_memory()
    assert test_ttl_expiration_cleanup()

    print("\n" + "=" * 65)
    print("ALL 4 SCENARIOS PASSED PERFECTLY! (100% SUCCESSFUL)")
    print("=" * 65)


if __name__ == "__main__":
    run_all_scenarios()
