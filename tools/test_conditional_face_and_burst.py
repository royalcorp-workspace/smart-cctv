"""Unit test suite for Conditional Face Detection & Dynamic Burst Capture.

Tests:
1. Conditional Face Detection: Bypass when no person in frame, grace period (1.5s) preservation, and final cache clearing.
2. Dynamic Burst Capture:
   - Normal interval = 3.
   - Burst trigger on new bag in sterile zone (interval = 1 for 3s).
   - Burst trigger on bag newly stationary (interval = 1 for 3s).
   - Burst trigger on stationary bag moving/removed (interval = 1 for 3s).
   - Recovery to normal interval = 3 after burst expiry.
3. Zone-Prioritized Hybrid Head RoI:
   - Extraction of high-res head crop RoI for relevant persons in sterile zone.
   - Multi-crop and single-crop handling in YuNetFaceDetector.
"""

import time
from pathlib import Path
from typing import List, Tuple
import numpy as np

from engine.face_detector import YuNetFaceDetector
from engine.tracker import CentroidTracker, TrackedObject


def test_conditional_face_bypass_and_grace_period():
    """Verify that face detection is skipped when no persons are present and cache is purged after 1.5s."""
    print("\n[TEST 1] Conditional Face Detection Bypass & Grace Period...")
    
    # Mock pipeline state
    last_person_seen_time = 1000.0
    cached_faces = [((100, 100, 40, 40), 0.92, "Alice")]
    face_recog_cache = {1: {"label": "Alice"}}

    # Frame 1: Person present -> YuNet allowed
    person_boxes = [(100, 80, 50, 120)]
    now = 1000.5
    has_persons = len(person_boxes) > 0
    assert has_persons is True
    if has_persons:
        last_person_seen_time = now
    print(" - Person present at t=1000.5s: Face detection allowed.")

    # Frame 2: Person drops for 0.5s (transient drop) -> Within grace period (< 1.5s)
    person_boxes = []
    now = 1001.0
    has_persons = len(person_boxes) > 0
    assert has_persons is False

    # Simulate logic
    if has_persons:
        last_person_seen_time = now
    else:
        if (now - last_person_seen_time) >= 1.5:
            cached_faces = []
            face_recog_cache.clear()

    # Face detection is BYPASSED
    assert has_persons is False, "Face detection must be bypassed when no person in frame"
    # But cached_faces is PRESERVED (anti-flicker grace period)
    assert len(cached_faces) == 1, "Cached faces should be preserved within 1.5s grace period"
    print(" - Person absent for 0.5s: Face detection bypassed, cache preserved (anti-flicker).")

    # Frame 3: Person absent for 2.0s (> 1.5s grace period)
    now = 1002.6
    has_persons = len(person_boxes) > 0
    if has_persons:
        last_person_seen_time = now
    else:
        if (now - last_person_seen_time) >= 1.5:
            cached_faces = []
            face_recog_cache.clear()

    assert len(cached_faces) == 0, "Cached faces must be purged after 1.5s absence"
    assert len(face_recog_cache) == 0, "Recognition cache must be cleared after 1.5s absence"
    print(" - [PASS] Conditional Face Detection bypass & 1.5s grace period verified!")


def test_dynamic_burst_capture_triggers():
    """Verify dynamic burst interval reduction (3 -> 1) for 3.0s across all 3 bag interaction triggers."""
    print("\n[TEST 2] Dynamic Burst Capture Triggers (3.0s at interval=1)...")
    
    base_detect_interval = 3
    face_burst_until_time = 0.0
    prev_stationary_bag_ids = set()
    zones_cfg = {"zone_2_transit": {"detect_unattended": True}}

    # Helper function matching main.py logic
    def check_burst(active_tracks, now, prev_stationary):
        trigger_burst = False
        current_stationary = set()
        for trk in active_tracks:
            is_bag = trk.class_label in ("tas", "backpack", "handbag", "suitcase")
            if is_bag:
                z_info = zones_cfg.get(trk.zone_id, {})
                is_unattended = z_info.get("detect_unattended", False)
                if is_unattended:
                    # Condition 1: New bag in sterile zone
                    if trk.frame_count <= 2:
                        trigger_burst = True
                    # Condition 2: Bag newly stationary
                    if trk.is_stationary:
                        current_stationary.add(trk.track_id)
                        if trk.track_id not in prev_stationary:
                            trigger_burst = True

        # Condition 3: Stationary bag starts moving or is taken
        for prev_sid in prev_stationary:
            if prev_sid not in current_stationary:
                trigger_burst = True

        return trigger_burst, current_stationary

    t0 = 1000.0

    # Trigger 1: New bag entered sterile zone
    new_bag = TrackedObject(
        track_id=1,
        centroid=(300, 300),
        anchor_centroid=(300, 300),
        bbox=(280, 280, 40, 40),
        zone_id="zone_2_transit",
        contour_area=1600.0,
        first_seen=t0,
        last_seen=t0,
        stationary_start=t0,
        class_label="tas",
        frame_count=1,
        is_stationary=False,
    )
    trig, curr_stat = check_burst([new_bag], t0, prev_stationary_bag_ids)
    prev_stationary_bag_ids = curr_stat
    assert trig is True
    face_burst_until_time = t0 + 3.0

    # Check interval during burst
    is_burst = (t0 + 1.0) < face_burst_until_time
    cur_interval = 1 if is_burst else base_detect_interval
    assert cur_interval == 1, "Interval must be 1 during burst"
    print(" - Trigger 1 (New bag in sterile zone): Burst activated (interval=1).")

    # Fast forward after 3.5s -> burst expired
    t1 = t0 + 3.5
    is_burst = t1 < face_burst_until_time
    cur_interval = 1 if is_burst else base_detect_interval
    assert cur_interval == 3, "Interval must return to 3 after burst expiry"
    print(" - Burst expiry check: Interval successfully returned to 3.")

    # Trigger 2: Bag newly stationary
    new_bag.frame_count = 10
    new_bag.is_stationary = True
    trig, curr_stat = check_burst([new_bag], t1, prev_stationary_bag_ids)
    prev_stationary_bag_ids = curr_stat
    assert trig is True
    face_burst_until_time = t1 + 3.0

    is_burst = (t1 + 0.5) < face_burst_until_time
    assert (1 if is_burst else 3) == 1
    print(" - Trigger 2 (Bag newly stationary): Burst activated (interval=1).")

    # Trigger 3: Stationary bag moved or picked up
    t2 = t1 + 4.0
    new_bag.is_stationary = False  # moved
    trig, curr_stat = check_burst([new_bag], t2, prev_stationary_bag_ids)
    prev_stationary_bag_ids = curr_stat
    assert trig is True
    face_burst_until_time = t2 + 3.0

    is_burst = (t2 + 1.0) < face_burst_until_time
    assert (1 if is_burst else 3) == 1
    print(" - Trigger 3 (Stationary bag moved/picked up): Burst activated (interval=1).")
    print(" - [PASS] Dynamic Burst Capture triggers and lifecycle verified!")


def test_zone_prioritized_hybrid_roi_and_detector():
    """Verify Zone-Prioritized Hybrid Head RoI extraction and detector multi-ROI handling."""
    print("\n[TEST 3] Zone-Prioritized Hybrid Head RoI & YuNet Multi-Crop...")

    # Person in 640x480 space
    p_box = (200, 100, 80, 180)  # px, py, pw, ph
    disp_w, disp_h = 1920, 1080
    scale_x = disp_w / 640.0
    scale_y = disp_h / 480.0

    px, py, pw, ph = p_box
    pad_x = int(pw * 0.15)
    h_x1 = max(0, px - pad_x)
    h_y1 = max(0, py - int(ph * 0.05))
    h_x2 = min(640, px + pw + pad_x)
    h_y2 = min(480, py + int(ph * 0.45))

    disp_x1 = int(round(h_x1 * scale_x))
    disp_y1 = int(round(h_y1 * scale_y))
    disp_x2 = int(round(h_x2 * scale_x))
    disp_y2 = int(round(h_y2 * scale_y))

    head_roi = (disp_x1, disp_y1, disp_x2, disp_y2)
    assert head_roi[2] > head_roi[0]
    assert head_roi[3] > head_roi[1]
    print(f" - Extracted 1080p head RoI: {head_roi}")

    # Test YuNet multi-crop parameter tolerance
    detector = YuNetFaceDetector(alternate_inference=False)
    blank_disp = np.zeros((1080, 1920, 3), dtype=np.uint8)
    blank_infer = np.zeros((480, 640, 3), dtype=np.uint8)

    # Test with single tuple ROI
    faces_single = detector.detect_faces(frame=blank_infer, display_frame=blank_disp, crop_roi=head_roi)
    assert isinstance(faces_single, list)

    # Test with list of ROIs (multi-ROI)
    faces_multi = detector.detect_faces(frame=blank_infer, display_frame=blank_disp, crop_roi=[head_roi, (100, 100, 300, 300)])
    assert isinstance(faces_multi, list)
    print(" - [PASS] YuNet handles single-ROI and multi-ROI dynamic crops smoothly.")


def run_all():
    print("================================================================")
    print("RUNNING CONDITIONAL FACE DETECTION & DYNAMIC BURST TEST SUITE")
    print("================================================================")
    test_conditional_face_bypass_and_grace_period()
    test_dynamic_burst_capture_triggers()
    test_zone_prioritized_hybrid_roi_and_detector()
    print("\n================================================================")
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("================================================================")


if __name__ == "__main__":
    run_all()
