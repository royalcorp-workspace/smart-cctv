"""Universal 3-Panel Telegram Composite Evidence Card Builder for Smart CCTV 2.0.

Generates sleek, high-resolution (1500x720) dark-mode visual cards combining:
- Panel 1 (Kiri): "AREA STERIL" -> Contextual macro camera view (wide angle).
- Panel 2 (Tengah): "OBJEK TERLENGKAP" -> Close-up crop of the violating physical object (20% padding).
- Panel 3 (Kanan): "TERDUGA PELAKU" -> High-res face crop (YuNet/SFace), body fallback, or placeholder.
- Footer Banner: 3-column structured metadata telemetry and stage escalation.
"""

import datetime
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np


def _fit_image_into_box(
    img: np.ndarray,
    max_w: int,
    max_h: int,
    bg_color: Tuple[int, int, int] = (20, 15, 12),
) -> np.ndarray:
    """Scale and center an image into a fixed container box preserving aspect ratio."""
    canvas = np.full((max_h, max_w, 3), bg_color, dtype=np.uint8)
    if img is None or img.size == 0:
        return canvas

    ih, iw = img.shape[:2]
    if ih <= 0 or iw <= 0:
        return canvas

    scale = min(max_w / float(iw), max_h / float(ih))
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    # Center in container
    x_offset = (max_w - new_w) // 2
    y_offset = (max_h - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized
    return canvas


def generate_composite_evidence(
    object_track: Any,
    current_frame: np.ndarray,
    stage_name: str = "BREACH",
    zone_name: Optional[str] = None,
    camera_id: str = "cam_01",
    timestamp_str: Optional[str] = None,
) -> np.ndarray:
    """Construct a premium 3-panel side-by-side composite evidence card (1500x720).

    Args:
        object_track: TrackedObject instance or duck-typed object.
        current_frame: Full-resolution clean or display frame (e.g. 1080p).
        stage_name: "WARNING" (50%), "PRE_ALARM" (85%), "BREACH" (100%), or "RESOLVED".
        zone_name: Human-readable zone name.
        camera_id: Camera identifier string.
        timestamp_str: Formatted timestamp string (optional, defaults to now).

    Returns:
        np.ndarray: BGR image of shape (720, 1500, 3).
    """
    card_w = 1500
    card_h = 720

    # Base canvas: Dark charcoal slate (#0d131f)
    canvas = np.full((card_h, card_w, 3), (28, 17, 11), dtype=np.uint8)

    # Timestamp normalization
    if not timestamp_str:
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Object properties
    track_id = getattr(object_track, "track_id", 0)
    class_label = getattr(object_track, "class_label", "object")
    dwell_duration = float(getattr(object_track, "dwell_duration", 0.0))
    dwell_threshold = float(getattr(object_track, "dwell_threshold", 3600.0))
    zone_id = getattr(object_track, "zone_id", "sterile_zone")
    display_zone = zone_name or zone_id or "Zona Transit"

    # Actor properties
    actor_meta = getattr(object_track, "associated_face_meta", None) or getattr(object_track, "last_owner_info", None) or {}
    actor_name = actor_meta.get("name", "Tidak Teridentifikasi")
    actor_conf = float(actor_meta.get("confidence", 0.0))
    face_crop = getattr(object_track, "associated_face_crop", None)
    if face_crop is None:
        face_crop = actor_meta.get("face_crop")
    person_crop = getattr(object_track, "associated_person_crop", None)
    if person_crop is None:
        person_crop = actor_meta.get("person_crop")

    # 1. Header Bar (Y: 0 to 75)
    cv2.rectangle(canvas, (0, 0), (card_w, 75), (45, 30, 22), -1)
    cv2.line(canvas, (0, 75), (card_w, 75), (75, 55, 42), 1, cv2.LINE_AA)

    # Left: Camera Badge & Brand Title
    cam_badge = f"[{camera_id.upper()}]"
    cv2.putText(canvas, cam_badge, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 220, 100), 2, cv2.LINE_AA)
    title_text = "SMART CCTV 2.0 | COMPOSITE EVIDENCE CARD (3-PANEL)"
    cv2.putText(canvas, title_text, (140, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

    # Right: Stage Pill Badge
    stage_upper = stage_name.upper()
    if "WARN" in stage_upper:
        stage_text = "STAGE 1: WARNING (50% DWELL)"
        stage_bg = (25, 140, 240)    # Amber / Orange
        stage_fg = (255, 255, 255)
    elif "PRE" in stage_upper:
        stage_text = "STAGE 2: PRE-ALARM (85% DWELL)"
        stage_bg = (20, 90, 245)     # Coral / Orange-Red
        stage_fg = (255, 255, 255)
    elif "RESOLV" in stage_upper:
        stage_text = "STAGE 4: RESOLVED"
        stage_bg = (60, 180, 50)     # Emerald Green
        stage_fg = (255, 255, 255)
    else:
        stage_text = "STAGE 3: CLEAR AREA BREACH"
        stage_bg = (35, 35, 225)     # Crimson Red
        stage_fg = (255, 255, 255)

    (st_w, st_h), _ = cv2.getTextSize(stage_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    pill_pad_x, pill_pad_y = 16, 8
    pill_x2 = card_w - 24
    pill_x1 = pill_x2 - (st_w + pill_pad_x * 2)
    pill_y1 = 18
    pill_y2 = pill_y1 + st_h + pill_pad_y * 2
    cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), stage_bg, -1)
    cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, stage_text, (pill_x1 + pill_pad_x, pill_y2 - pill_pad_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, stage_fg, 2, cv2.LINE_AA)

    # 2. 3-Panel Main Layout (Y: 88 to 573, Height: 485 px)
    panel_y1 = 88
    panel_h = 485
    panel_y2 = panel_y1 + panel_h

    # 3 Balanced Columns (~468px width each, 20px gap)
    p_w = 468
    p1_x1, p1_x2 = 24, 24 + p_w
    p2_x1, p2_x2 = 516, 516 + p_w
    p3_x1, p3_x2 = 1008, 1008 + p_w

    inner_box_w = p_w - 24    # 444 px
    inner_box_h = panel_h - 85 # 400 px
    left_foot_y = panel_y2 - 26

    # ==========================================================
    # --- PANEL 1 (KIRI): AREA STERIL (KONTEKS MAKRO WIDE) ---
    # ==========================================================
    cv2.rectangle(canvas, (p1_x1, panel_y1), (p1_x2, panel_y2), (38, 26, 18), -1)
    cv2.rectangle(canvas, (p1_x1, panel_y1), (p1_x2, panel_y2), (75, 55, 42), 1, cv2.LINE_AA)

    # Panel 1 Header Strip
    cv2.rectangle(canvas, (p1_x1, panel_y1), (p1_x2, panel_y1 + 38), (52, 36, 26), -1)
    cv2.circle(canvas, (p1_x1 + 18, panel_y1 + 19), 5, (60, 200, 50), -1, cv2.LINE_AA)
    p1_title = "AREA STERIL (KONTEKS MAKRO)"
    cv2.putText(canvas, p1_title, (p1_x1 + 32, panel_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Macro Frame (Wide-Angle overview with object location highlight)
    raw_bbox = getattr(object_track, "bbox", (0, 0, 0, 0))
    bx, by, bw, bh = raw_bbox
    fh, fw = current_frame.shape[:2] if (current_frame is not None and current_frame.size > 0) else (1080, 1920)

    # Coordinate space scaling if bbox is in 640x480 space
    if (fw, fh) != (640, 480) and (bx < 640 and by < 480 and bw <= 640 and bh <= 480):
        scale_x = fw / 640.0
        scale_y = fh / 480.0
    else:
        scale_x = 1.0
        scale_y = 1.0

    sbx = int(round(bx * scale_x))
    sby = int(round(by * scale_y))
    sbw = max(1, int(round(bw * scale_x)))
    sbh = max(1, int(round(bh * scale_y)))

    if current_frame is not None and current_frame.size > 0:
        macro_view = current_frame.copy()
        # Highlight object in the macro scene
        cv2.rectangle(macro_view, (sbx, sby), (sbx + sbw, sby + sbh), (0, 220, 255), 3, cv2.LINE_AA)
        cv2.circle(macro_view, (sbx + sbw // 2, sby + sbh // 2), 8, (0, 220, 255), -1, cv2.LINE_AA)
    else:
        macro_view = np.full((360, 640, 3), (35, 25, 18), dtype=np.uint8)
        cv2.putText(macro_view, "Tampilan Makro Tidak Tersedia", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    p1_display = _fit_image_into_box(macro_view, inner_box_w, inner_box_h, bg_color=(24, 16, 12))
    canvas[panel_y1 + 42 : panel_y1 + 42 + inner_box_h, p1_x1 + 12 : p1_x1 + 12 + inner_box_w] = p1_display

    p1_foot_text = f"Sudut Kamera Penuh  |  Zona: {display_zone}"
    cv2.putText(canvas, p1_foot_text, (p1_x1 + 16, left_foot_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 180, 160), 1, cv2.LINE_AA)

    # ==========================================================
    # --- PANEL 2 (TENGAH): OBJEK TERLENGKAP (CROP DETAIL) ---
    # ==========================================================
    cv2.rectangle(canvas, (p2_x1, panel_y1), (p2_x2, panel_y2), (38, 26, 18), -1)
    cv2.rectangle(canvas, (p2_x1, panel_y1), (p2_x2, panel_y2), (75, 55, 42), 1, cv2.LINE_AA)

    # Panel 2 Header Strip
    cv2.rectangle(canvas, (p2_x1, panel_y1), (p2_x2, panel_y1 + 38), (52, 36, 26), -1)
    cv2.circle(canvas, (p2_x1 + 18, panel_y1 + 19), 5, (255, 200, 0), -1, cv2.LINE_AA)
    p2_title = f"OBJEK TERLENGKAP : {class_label.upper()} #{track_id}"
    cv2.putText(canvas, p2_title, (p2_x1 + 32, panel_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Extract close-up crop with 20% margin padding
    pad_w = int(round(sbw * 0.20))
    pad_h = int(round(sbh * 0.20))
    ox1 = max(0, sbx - pad_w)
    oy1 = max(0, sby - pad_h)
    ox2 = min(fw, sbx + sbw + pad_w)
    oy2 = min(fh, sby + sbh + pad_h)

    if current_frame is not None and current_frame.size > 0 and (ox2 > ox1 and oy2 > oy1):
        object_crop = current_frame[oy1:oy2, ox1:ox2].copy()
        # Highlight object within its cropped frame
        rel_x = sbx - ox1
        rel_y = sby - oy1
        cv2.rectangle(object_crop, (rel_x, rel_y), (rel_x + sbw, rel_y + sbh), (0, 220, 255), 2, cv2.LINE_AA)
    else:
        object_crop = np.full((300, 400, 3), (35, 25, 18), dtype=np.uint8)
        cv2.putText(object_crop, "Crop Objek Tidak Tersedia", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    p2_display = _fit_image_into_box(object_crop, inner_box_w, inner_box_h, bg_color=(24, 16, 12))
    canvas[panel_y1 + 42 : panel_y1 + 42 + inner_box_h, p2_x1 + 12 : p2_x1 + 12 + inner_box_w] = p2_display

    p2_foot_text = f"Ukuran: {sbw}x{sbh} px  |  Margin: 20%"
    cv2.putText(canvas, p2_foot_text, (p2_x1 + 16, left_foot_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 180, 160), 1, cv2.LINE_AA)

    # ==========================================================
    # --- PANEL 3 (KANAN): TERDUGA PELAKU / PEMILIK ---
    # ==========================================================
    cv2.rectangle(canvas, (p3_x1, panel_y1), (p3_x2, panel_y2), (38, 26, 18), -1)
    cv2.rectangle(canvas, (p3_x1, panel_y1), (p3_x2, panel_y2), (75, 55, 42), 1, cv2.LINE_AA)

    # Panel 3 Header Strip
    cv2.rectangle(canvas, (p3_x1, panel_y1), (p3_x2, panel_y1 + 38), (52, 36, 26), -1)
    cv2.circle(canvas, (p3_x1 + 18, panel_y1 + 19), 5, (0, 200, 255), -1, cv2.LINE_AA)
    p3_title = "TERDUGA PELAKU / PEMILIK"
    cv2.putText(canvas, p3_title, (p3_x1 + 32, panel_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Select actor visual representation
    if face_crop is not None and face_crop.size > 0:
        actor_img = face_crop
        actor_badge = "WAJAH TERDETEKSI (YuNet + SFace)"
        badge_bg = (40, 130, 30)  # Green
    elif person_crop is not None and person_crop.size > 0:
        actor_img = person_crop
        actor_badge = "SOSOK TUBUH (Wajah Tidak Terdeteksi)"
        badge_bg = (20, 120, 200)  # Orange
    else:
        actor_img = None
        actor_badge = None
        badge_bg = None

    if actor_img is not None:
        p3_display = _fit_image_into_box(actor_img, inner_box_w, inner_box_h, bg_color=(24, 16, 12))
        if actor_badge:
            (bw_t, bh_t), _ = cv2.getTextSize(actor_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(p3_display, (8, 8), (8 + bw_t + 16, 8 + bh_t + 12), badge_bg, -1)
            cv2.rectangle(p3_display, (8, 8), (8 + bw_t + 16, 8 + bh_t + 12), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(p3_display, actor_badge, (16, 8 + bh_t + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        # Elegant Dark Card Placeholder
        p3_display = np.full((inner_box_h, inner_box_w, 3), (24, 16, 12), dtype=np.uint8)
        cv2.rectangle(p3_display, (20, 30), (inner_box_w - 20, inner_box_h - 30), (50, 35, 26), 2, cv2.LINE_AA)
        cv2.putText(p3_display, "[ WAJAH / ORANG TIDAK TERDETEKSI ]", (30, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (140, 140, 140), 2, cv2.LINE_AA)
        cv2.putText(p3_display, "Tidak ada person berada di dekat objek saat peletakan.", (28, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(p3_display, "Area di luar jangkauan detektor wajah atau pencahayaan minim.", (16, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (95, 95, 95), 1, cv2.LINE_AA)

    canvas[panel_y1 + 42 : panel_y1 + 42 + inner_box_h, p3_x1 + 12 : p3_x1 + 12 + inner_box_w] = p3_display

    conf_str = f"{int(round(actor_conf * 100))}%" if actor_conf > 0 else "-"
    p3_foot_text = f"Identitas: {actor_name}  |  Skor: {conf_str}"
    cv2.putText(canvas, p3_foot_text, (p3_x1 + 16, left_foot_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 180, 160), 1, cv2.LINE_AA)

    # ==========================================================
    # --- 3. FOOTER METADATA BANNER (Y: 585 to 705, Height: 120 px) ---
    # ==========================================================
    foot_y1 = 585
    foot_y2 = 705
    cv2.rectangle(canvas, (24, foot_y1), (card_w - 24, foot_y2), (38, 26, 18), -1)
    cv2.rectangle(canvas, (24, foot_y1), (card_w - 24, foot_y2), (75, 55, 42), 1, cv2.LINE_AA)

    dwell_m = dwell_duration / 60.0
    thresh_m = dwell_threshold / 60.0

    # Column 1
    c1_x = 45
    cv2.putText(canvas, f"ID OBJEK : {class_label.upper()} #{track_id}", (c1_x, foot_y1 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"ZONA     : {display_zone}", (c1_x, foot_y1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 200, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"KAMERA   : {camera_id.upper()}", (c1_x, foot_y1 + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 140, 130), 1, cv2.LINE_AA)

    # Column 2
    c2_x = 535
    cv2.putText(canvas, f"DWELL    : {dwell_m:.1f}m / {thresh_m:.1f}m", (c2_x, foot_y1 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"ESKALASI : {stage_text}", (c2_x, foot_y1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, stage_bg, 1, cv2.LINE_AA)
    dwell_percent = min(100, int(round((dwell_duration / max(1.0, dwell_threshold)) * 100.0)))
    cv2.putText(canvas, f"PROGRES  : {dwell_percent}% Dari Ambang Batas", (c2_x, foot_y1 + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 140, 130), 1, cv2.LINE_AA)

    # Column 3
    c3_x = 1035
    cv2.putText(canvas, f"WAKTU    : {timestamp_str}", (c3_x, foot_y1 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"PELAKU   : {actor_name}", (c3_x, foot_y1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 200, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"KEMIRIPAN: {conf_str}", (c3_x, foot_y1 + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 140, 130), 1, cv2.LINE_AA)

    return canvas


def encode_composite_jpg(composite_img: np.ndarray, quality: int = 90) -> Optional[bytes]:
    """Encode composite image array to JPEG bytes."""
    if composite_img is None or composite_img.size == 0:
        return None
    success, enc = cv2.imencode(".jpg", composite_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if success:
        return enc.tobytes()
    return None
