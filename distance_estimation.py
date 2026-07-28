"""
Monocular distance estimation for maritime vessel detection.

Uses the pinhole camera model:

    distance_m = (real_world_width_m * focal_length_px) / bbox_width_px


"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "camera_config.json")

# Reference real-world width of the vessel class being detected (metres).
# Default: ~10m beam, based on Fullers360 harbour ferry class (e.g. Kea).
# Update this if you confirm the exact vessel class/spec you're training on.
KNOWN_WIDTH_M = 10.0


def load_focal_length():
    """Load the calibrated focal length (in pixels) from camera_config.json."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(
            f"No camera_config.json found at {CONFIG_PATH}.\n"
            "Run calibrate_focal_length.py first to generate it."
        )
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    return config["focal_length_px"]


def estimate_distance(bbox_width_px, focal_length_px, known_width_m=KNOWN_WIDTH_M):
    """
    Estimate distance to an object using its detected bounding box width.

    Args:
        bbox_width_px: width of the YOLO bounding box in pixels (x2 - x1)
        focal_length_px: calibrated camera focal length in pixels
        known_width_m: real-world width of the object class (metres)

    Returns:
        Estimated distance in metres, or None if bbox_width_px is invalid.
    """
    if bbox_width_px <= 0:
        return None
    return (known_width_m * focal_length_px) / bbox_width_px


if __name__ == "__main__":
    # Quick manual test
    f_px = load_focal_length()
    test_bbox_width = 220  # example pixel width
    d = estimate_distance(test_bbox_width, f_px)
    print(f"Focal length loaded: {f_px:.1f}px")
    print(f"Test bbox width {test_bbox_width}px -> estimated distance: {d:.1f}m")