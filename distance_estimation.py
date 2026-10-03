"""
Monocular distance estimation for maritime vessel detection (Layer 1).

Uses the pinhole camera model:

    distance_m = (real_world_width_m * focal_length_px) / bbox_width_px

Focal length is now derived automatically from known camera hardware
specs (see camera_profiles.py) rather than requiring a manual,
interactive calibration step. This is the always-available default path;
horizon_ranging.py provides an optional, more accurate refinement on top
of this when a clean sea horizon is visible.

Accuracy notes:
  - This method assumes the camera is looking roughly perpendicular to
    the ferry's beam (i.e. broadside-on). If the ferry is angled
    toward/away from the camera, the apparent width shrinks and the
    estimate will read "further away than it really is" -- a known
    limitation of single-camera geometric methods, worth noting as
    future work (e.g. combining with heading/orientation once merged).
  - Distance accuracy also degrades at long range, since small pixel
    errors in the bounding box translate into larger distance errors.
"""

from camera_profiles import get_focal_length_px, get_horizontal_fov_deg

# Reference real-world width of the vessel class being detected (metres).
# Default: ~10m beam, based on Fullers360 harbour ferry class (e.g. Kea).
# Update this if you confirm the exact vessel class/spec you're training on.
KNOWN_WIDTH_M = 10.0


def load_focal_length(image_width_px):
    """
    Zero-calibration focal length (pixels), computed from known camera
    hardware specs for the current frame width. No manual calibration
    step required -- see camera_profiles.py.
    """
    return get_focal_length_px(image_width_px)


def load_horizontal_fov_deg():
    """Approximate horizontal field of view for the active camera profile."""
    return get_horizontal_fov_deg()


def estimate_distance(bbox_width_px, focal_length_px, known_width_m=KNOWN_WIDTH_M):
    """
    Estimate distance to an object using its detected bounding box width.

    Args:
        bbox_width_px: width of the YOLO bounding box in pixels (x2 - x1)
        focal_length_px: focal length in pixels (see load_focal_length)
        known_width_m: real-world width of the object class (metres)

    Returns:
        Estimated distance in metres, or None if bbox_width_px is invalid.
    """
    if bbox_width_px <= 0:
        return None
    return (known_width_m * focal_length_px) / bbox_width_px


if __name__ == "__main__":
    # Quick manual test -- no calibration file needed anymore
    frame_width = 1280
    f_px = load_focal_length(frame_width)
    test_bbox_width = 220  # example pixel width
    d = estimate_distance(test_bbox_width, f_px)
    print(f"Focal length @ {frame_width}px width: {f_px:.1f}px")
    print(f"Horizontal FOV: {load_horizontal_fov_deg():.1f} deg")
    print(f"Test bbox width {test_bbox_width}px -> estimated distance: {d:.1f}m")