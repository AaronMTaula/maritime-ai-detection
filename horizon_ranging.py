"""
Horizon-dip based distance refinement (Layer 2).

This is an optional accuracy booster layered on top of the always-
available focal-length-based estimate (distance_estimation.py +
camera_profiles.py). It implements the classic maritime "vertical
sextant angle" / dip-of-horizon ranging technique -- with one important
correction from the first version of this idea, noted below.

IMPORTANT -- why this needs Layer 1's focal length too:
It's tempting to think the horizon's dip angle alone could "self
calibrate" the camera's degrees-per-pixel, with no focal length needed
at all. It can't, in practice: at a realistic camera height (e.g. 1.5m),
the dip angle is only ~0.03-0.05 degrees, which works out to well under
one pixel on a typical 720p frame. That's far smaller than the noise in
horizon detection itself, so it can't be used to derive angular
resolution from scratch. Instead, Layer 2 uses Layer 1's known focal
length (from camera_profiles.py) to convert pixel offsets to angles
correctly, and uses the live-detected horizon purely as a REFERENCE
LEVEL LINE -- which is still genuinely useful, because it automatically
cancels out camera pitch/tilt error every frame (a boat-mounted or
handheld camera won't stay perfectly level), without needing a gyro or
IMU. The dip angle is then a small, honest correction on top of that.

How it works:
  1. Detect the horizon row in the current frame.
  2. Convert both the horizon's and the target's pixel row to an angle
     relative to the camera's boresight, using the known focal length
     (proper arctan/pinhole projection, not a linear approximation):
         angle(row) = atan((row - principal_row) / focal_length_px)
  3. Because both angles are measured the same way, any unknown camera
     tilt cancels out in the difference between them -- this is the
     self-leveling property.
  4. Add the known dip angle (geometry only, no measurement needed):
         dip (degrees) = 0.0293 * sqrt(height in metres)
     to get the target's true depression angle below level.
  5. Solve: distance = height / tan(depression angle).

WHEN TO USE THIS: open water with a clear, level sea horizon (ideally a
fixed mount). It is OFF by default and switched on per configuration in
the Settings window. In harbours the visible "horizon" is usually a
skyline of hills, buildings or trees, which sits well ABOVE the true sea
horizon -- using it as the level reference would add a large angular
error, so a more forgiving detector would make results worse, not
better. The width-based estimate (distance_estimation.py) does not
depend on the horizon or on camera tilt at all, so it carries the
system everywhere this method can't.

Even when enabled, this method reports "unavailable" (returns None)
whenever no horizon is confidently detected, and the caller falls back
to the width-based estimate.

CAMERA_HEIGHT_ABOVE_WATER_M is the one remaining physical constant this
method needs -- set once at installation time (measured from the camera
mount down to the waterline), not something the end user configures
interactively at runtime.
"""

import cv2
import numpy as np
import math

from app_config import get_active_settings

# Developer-only fallback if the Settings UI has genuinely never been
# run. Clients set the real value through that UI (camera_setup_ui.py),
# not by editing this file.
DEFAULT_CAMERA_HEIGHT_M = 1.5


def get_camera_height_m():
    """Camera height above the waterline, metres -- from the Settings window."""
    return get_active_settings().get("camera_height_m") or DEFAULT_CAMERA_HEIGHT_M


def horizon_assist_enabled():
    """True only if the user turned on horizon-assisted ranging in Settings."""
    return bool(get_active_settings().get("horizon_assist"))


# Kept for any external code/backwards compatibility that reads this
# directly; prefer get_camera_height_m() since it reflects live config.
CAMERA_HEIGHT_ABOVE_WATER_M = DEFAULT_CAMERA_HEIGHT_M

# Only search for the horizon within this vertical band of the frame
# (as a fraction of frame height), to avoid false positives from sky
# clouds or foreground clutter near the top/bottom edges.
HORIZON_SEARCH_BAND = (0.25, 0.75)

# Minimum edge-strength confidence (0-1) for a detected horizon to be
# trusted. Tune this against real footage -- raise it if false positives
# occur (e.g. wharf edges or wave crests mistaken for the horizon).
MIN_HORIZON_CONFIDENCE = 0.35


def detect_horizon_row(frame):
    """
    Estimate the pixel row of the sea horizon using the strongest
    horizontal brightness transition within the search band.

    Returns (row_px, confidence) where confidence is 0-1, or (None, conf)
    if no horizon could be confidently detected this frame.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    row_start = int(h * HORIZON_SEARCH_BAND[0])
    row_end = int(h * HORIZON_SEARCH_BAND[1])

    row_means = blurred[row_start:row_end, :].mean(axis=1)
    if len(row_means) < 3:
        return None, 0.0

    gradients = np.abs(np.diff(row_means))
    best_idx = int(np.argmax(gradients))
    best_gradient = float(gradients[best_idx])

    # Normalise gradient strength into a rough 0-1 confidence score,
    # relative to the overall brightness contrast present in the band.
    band_contrast = float(row_means.max() - row_means.min()) + 1e-6
    confidence = min(best_gradient / band_contrast, 1.0)

    if confidence < MIN_HORIZON_CONFIDENCE:
        return None, confidence

    horizon_row = row_start + best_idx
    return horizon_row, confidence


def _dip_angle_deg(height_m):
    """Standard navigational dip-of-horizon approximation."""
    return 0.0293 * math.sqrt(max(height_m, 0.0))


def _pixel_row_to_angle_deg(row_px, principal_row_px, focal_length_px):
    """Proper pinhole-model angle (degrees) of a pixel row relative to
    the camera's boresight. Positive = below the boresight."""
    return math.degrees(math.atan((row_px - principal_row_px) / focal_length_px))


def estimate_distance_via_horizon(frame, target_waterline_row_px, focal_length_px,
                                   camera_height_m=None):
    """
    Attempt to estimate distance to a target using the horizon-dip method.

    Args:
        frame: current BGR frame (used to detect the horizon).
        target_waterline_row_px: pixel row of the target's waterline
            (typically the bottom edge of its bounding box).
        focal_length_px: focal length in pixels, from Layer 1
            (camera_profiles.get_focal_length_px) -- needed to convert
            pixel offsets to angles correctly. See module docstring for
            why the dip angle alone can't provide this.
        camera_height_m: camera height above the waterline, metres. If
            omitted, uses the value set via the Settings UI.

    Returns:
        (distance_m, debug_info). distance_m is None if the method is
        unavailable this frame. debug_info includes 'horizon_row' and
        'confidence' for optional on-screen display / troubleshooting.
    """
    if camera_height_m is None:
        camera_height_m = get_camera_height_m()

    h, w = frame.shape[:2]
    horizon_row, confidence = detect_horizon_row(frame)
    debug_info = {"horizon_row": horizon_row, "confidence": confidence}

    if horizon_row is None:
        return None, debug_info

    principal_row = h / 2.0
    horizon_angle_deg = _pixel_row_to_angle_deg(horizon_row, principal_row, focal_length_px)
    target_angle_deg = _pixel_row_to_angle_deg(target_waterline_row_px, principal_row, focal_length_px)

    # Any unknown camera tilt affects both angles equally, so it cancels
    # out here -- this is what makes the method self-leveling frame to
    # frame without needing a gyro/IMU.
    dip_deg = _dip_angle_deg(camera_height_m)
    target_depression_deg = dip_deg + (target_angle_deg - horizon_angle_deg)

    if target_depression_deg <= 0:
        # Target's waterline is at or above true level -- not
        # geometrically valid (detection noise, or the box's bottom
        # edge isn't really sitting on the waterline).
        return None, debug_info

    distance_m = camera_height_m / math.tan(math.radians(target_depression_deg))

    # Sanity bound -- discard implausible results rather than displaying
    # something misleading.
    if distance_m <= 0 or distance_m > 20000:
        return None, debug_info

    debug_info.update({
        "dip_deg": dip_deg,
        "horizon_angle_deg": horizon_angle_deg,
        "target_angle_deg": target_angle_deg,
        "target_depression_deg": target_depression_deg,
    })
    return distance_m, debug_info