"""
Interactive HUD utilities

Provides:
  - Feature 1: Click-to-select bounding boxes, with frame-to-frame tracking
    so the selection follows the same ship even as detections refresh.
  - Feature 2: A radar/sonar-style aesthetic display showing the selected
    ship's distance (and, once merged, heading/bearing).

"""

import cv2
import math
import time

# ---------------------------------------------------------------------------
# Colours (BGR order, since OpenCV)
# ---------------------------------------------------------------------------
COLOR_DEFAULT_BOX  = (255, 128, 0)   # blue — matches your existing box colour
COLOR_SELECTED_BOX = (0, 165, 255)   # orange highlight
COLOR_TEXT         = (255, 255, 255)
COLOR_PANEL_BG     = (20, 20, 20)
COLOR_RING         = (0, 200, 80)
COLOR_SWEEP        = (0, 255, 120)
COLOR_BLIP_NEAR    = (0, 0, 255)     # red   - close
COLOR_BLIP_MID     = (0, 220, 255)   # amber - medium
COLOR_BLIP_FAR     = (0, 255, 0)     # green - far

# Distance thresholds (metres) for blip colour coding — tune these to
# whatever counts as "near/mid/far" for your use case.
NEAR_THRESHOLD_M = 30
MID_THRESHOLD_M = 60

# Radar display range in metres — anything beyond this is pinned to the
# outer ring rather than drawn off-panel.
RADAR_MAX_RANGE_M = 100

# How close (in pixels) a detection's centre must be to the previously
# selected ship's centre to still count as "the same ship" between frames.
TRACKING_MATCH_RADIUS_PX = 120


class SelectionState:
    """Tracks which detected ship is currently selected, across frames."""

    def __init__(self):
        self.selected_center = None   # (x, y) of the currently selected ship
        self.click_point = None       # set by the mouse callback, consumed once per frame
        self.sweep_angle = 0.0        # radar sweep animation state (degrees)
        self.settings_button_rect = None  # (x1, y1, x2, y2), set when the button is drawn
        self.settings_requested = False   # set when the Settings button is clicked


def mouse_callback(event, x, y, flags, state: "SelectionState"):
    """Pass a SelectionState instance as the `param` to cv2.setMouseCallback."""
    if event == cv2.EVENT_LBUTTONDOWN:
        rect = state.settings_button_rect
        if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
            state.settings_requested = True   # clicked the Settings button, not a ship
            return
        state.click_point = (x, y)


def _box_center(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _point_in_box(pt, box_xyxy):
    x, y = pt
    x1, y1, x2, y2 = box_xyxy
    return x1 <= x <= x2 and y1 <= y <= y2


def _dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def update_selection(state: SelectionState, boxes_xyxy):
    """
    Resolve any pending click, then re-match the selected ship to this
    frame's detections so the highlight tracks it smoothly.

    Args:
        state: the shared SelectionState.
        boxes_xyxy: list of (x1, y1, x2, y2) for this frame's detections.

    Returns:
        Index into boxes_xyxy of the currently selected ship, or None if
        nothing is selected (or the selection couldn't be re-matched
        this particular frame).
    """
    # 1) Handle a pending click
    if state.click_point is not None:
        clicked_idx = None
        for i, box in enumerate(boxes_xyxy):
            if _point_in_box(state.click_point, box):
                clicked_idx = i
                break

        if clicked_idx is not None:
            clicked_center = _box_center(boxes_xyxy[clicked_idx])
            already_selected = (
                state.selected_center is not None
                and _dist(clicked_center, state.selected_center) < 20
            )
            # Clicking the already-selected ship deselects it (toggle)
            state.selected_center = None if already_selected else clicked_center
        else:
            # Clicked empty space -> deselect
            state.selected_center = None

        state.click_point = None

    # 2) Re-match the selection to this frame's boxes (tracking)
    if state.selected_center is None:
        return None

    best_idx, best_dist = None, TRACKING_MATCH_RADIUS_PX
    for i, box in enumerate(boxes_xyxy):
        d = _dist(_box_center(box), state.selected_center)
        if d < best_dist:
            best_dist, best_idx = d, i

    if best_idx is not None:
        state.selected_center = _box_center(boxes_xyxy[best_idx])  # follow movement
        return best_idx

    # Selection temporarily lost (e.g. no detection this frame) — keep the
    # last known position so it can be re-acquired once detection resumes.
    return None


def draw_detections(frame, boxes_xyxy, labels, confidences, selected_idx):
    """Draw all bounding boxes as usual, highlighting the selected one."""
    for i, (box, label, conf) in enumerate(zip(boxes_xyxy, labels, confidences)):
        x1, y1, x2, y2 = [int(v) for v in box]
        is_selected = (i == selected_idx)
        color = COLOR_SELECTED_BOX if is_selected else COLOR_DEFAULT_BOX
        thickness = 3 if is_selected else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        tag = f'{label} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(frame, tag, (x1 + 5, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if is_selected:
            cv2.putText(frame, 'SELECTED (click again to deselect)', (x1, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_info_callout(frame, box_xyxy, distance_m, heading_text='pending', source_label=None):
    """Small distance/heading readout above the selected ship's box."""
    x1, y1, _, _ = [int(v) for v in box_xyxy]
    if distance_m is not None:
        dist_text = f'Dist: {distance_m:.1f}m'
        if source_label:
            dist_text += f'  [{source_label}]'
    else:
        dist_text = 'Dist: N/A'
    hdg_text = f'Heading: {heading_text}'

    cv2.putText(frame, dist_text, (x1, max(20, y1 - 44)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(frame, hdg_text, (x1, max(40, y1 - 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def draw_settings_button(frame, state: SelectionState, label='Settings'):
    """Clickable button in the top-right corner. Its position is stored on
    the state so mouse_callback can tell a button click from a ship click."""
    h, w = frame.shape[:2]
    bw, bh, margin = 130, 34, 16
    x1, y1 = w - bw - margin, margin
    x2, y2 = x1 + bw, y1 + bh
    state.settings_button_rect = (x1, y1, x2, y2)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TEXT, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(frame, label, (x1 + (bw - tw) // 2, y1 + (bh + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)


def draw_horizon_line(frame, horizon_row, confidence):
    """Draw a faint dashed line at the detected horizon row, for visual
    feedback while testing/tuning the horizon-dip ranging method."""
    if horizon_row is None:
        return
    h, w = frame.shape[:2]
    y = int(horizon_row)
    dash_len, gap_len = 18, 10
    x = 0
    while x < w:
        cv2.line(frame, (x, y), (min(x + dash_len, w), y), (0, 255, 255), 1, cv2.LINE_AA)
        x += dash_len + gap_len
    cv2.putText(frame, f'HORIZON ({confidence:.0%})', (10, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


def draw_radar_panel(frame, state: SelectionState, distance_m, has_target,
                      bearing_deg=0.0, radius=90, panel_margin=40):
    """
    Draw a radar/sonar-style HUD panel in the bottom-right corner showing
    the selected ship's distance as a blip on rotating range rings.

    bearing_deg: relative bearing in degrees, 0 = straight ahead, positive
    = to the right. Currently derived as a placeholder from the bounding
    box's horizontal offset (see live_detect.py) until real heading data
    from the team's heading-estimation work is merged in.
    """
    h, w = frame.shape[:2]
    cx = w - radius - panel_margin
    cy = h - radius - panel_margin

    # Semi-transparent panel background
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), radius + 14, COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    # Range rings + crosshair
    for frac in (0.25, 0.5, 0.75, 1.0):
        cv2.circle(frame, (cx, cy), int(radius * frac), COLOR_RING, 1, cv2.LINE_AA)
    cv2.line(frame, (cx - radius, cy), (cx + radius, cy), COLOR_RING, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - radius), (cx, cy + radius), COLOR_RING, 1, cv2.LINE_AA)

    for frac in (0.5, 1.0):
        label_val = int(RADAR_MAX_RANGE_M * frac)
        cv2.putText(frame, f'{label_val}m', (cx + int(radius * frac) - 16, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1, cv2.LINE_AA)

    # Rotating sweep line with a short fading trail
    for trail_i in range(6):
        angle = math.radians(state.sweep_angle - trail_i * 8)
        alpha = 0.5 * (1 - trail_i / 6)
        end = (int(cx + radius * math.sin(angle)), int(cy - radius * math.cos(angle)))
        sweep_overlay = frame.copy()
        cv2.line(sweep_overlay, (cx, cy), end, COLOR_SWEEP, 2, cv2.LINE_AA)
        cv2.addWeighted(sweep_overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    state.sweep_angle = (state.sweep_angle + 6) % 360

    # Blip for the selected/tracked ship
    if has_target and distance_m is not None:
        clamped = min(distance_m, RADAR_MAX_RANGE_M)
        r = (clamped / RADAR_MAX_RANGE_M) * radius
        angle = math.radians(bearing_deg)
        bx = int(cx + r * math.sin(angle))
        by = int(cy - r * math.cos(angle))

        if distance_m < NEAR_THRESHOLD_M:
            blip_color = COLOR_BLIP_NEAR
        elif distance_m < MID_THRESHOLD_M:
            blip_color = COLOR_BLIP_MID
        else:
            blip_color = COLOR_BLIP_FAR

        pulse = int(2 * abs(math.sin(time.time() * 4)))
        cv2.circle(frame, (bx, by), 5 + pulse, blip_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), 9 + pulse, blip_color, 1, cv2.LINE_AA)

        cv2.putText(frame, f'{distance_m:.0f}m', (cx - radius, cy - radius - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, blip_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'NO TARGET', (cx - radius, cy - radius - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    cv2.putText(frame, 'DISTANCE RADAR', (cx - radius, cy - radius - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)