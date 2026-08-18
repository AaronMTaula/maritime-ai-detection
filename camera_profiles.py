"""
Known camera hardware profiles for automatic, zero-calibration focal
length estimation.

Instead of asking the user to manually calibrate the camera (click on a
reference object at a known distance), we use the camera's OWN known,
fixed specs directly. This works because the deployment camera is a
specific known model (e.g. iPhone 17 Pro), not an arbitrary/unknown
webcam that varies per client machine.

We use each camera's 35mm-equivalent focal length. This value is
specifically designed to be comparable across different physical sensor
sizes -- it's defined as "the focal length that would produce the same
field of view on a full-frame (36mm-wide) sensor." That means we can
convert straight to a pixel focal length without needing the camera's
raw physical sensor dimensions (which are hard to source precisely for
phone cameras, and not something OpenCV can query directly):

    focal_length_px = (equivalent_focal_length_mm / 36.0) * image_width_px

To deploy on a different fixed camera, add a new entry below -- then
select it once through the Settings UI (camera_setup_ui.py). Nothing
here requires a source code edit for the end user; ACTIVE_CAMERA_PROFILE
below is only a same-machine developer fallback (used if the Settings
UI has genuinely never been run), not something a client ever needs to
touch.
"""

import math

from app_config import load_config

# 35mm-equivalent focal length (mm) for each known deployment camera's
# main/wide (1x) lens, since that's what Continuity Camera / default
# capture uses.
CAMERA_PROFILES = {
    # Apple "iPhone 17 Pro" tech specs: 48MP Fusion Main camera,
    # 24mm equivalent focal length, f/1.78.
    "iphone_17_pro_main": {
        "equiv_focal_length_mm": 24.0,
        "label": "iPhone 17 Pro - Main (1x) camera",
    },
    # Placeholder -- replace with real specs if/when Navantia installs
    # dedicated fixed webcam hardware instead of an iPhone.
    "generic_webcam_hd": {
        "equiv_focal_length_mm": 28.0,  # typical laptop/USB webcam ballpark
        "label": "Generic HD webcam (approximate, ~28mm equivalent)",
    },
}

# Developer-only fallback, used solely if the Settings UI has never been
# run at all (shouldn't normally happen -- live_detect.py runs it on
# first launch). Clients configure their camera through that UI, not by
# editing this file.
ACTIVE_CAMERA_PROFILE = "iphone_17_pro_main"

FULL_FRAME_SENSOR_WIDTH_MM = 36.0  # reference width "equivalent" focal lengths are relative to


def _resolve_profile(profile=None):
    if profile:
        return profile
    cfg = load_config()
    return cfg.get("camera_profile") or ACTIVE_CAMERA_PROFILE


def get_focal_length_px(image_width_px, profile=None):
    """Compute focal length in pixels for the configured (or given) camera profile."""
    profile = _resolve_profile(profile)
    if profile not in CAMERA_PROFILES:
        raise KeyError(
            f"Unknown camera profile '{profile}'. "
            f"Available: {list(CAMERA_PROFILES.keys())}"
        )
    equiv_focal_mm = CAMERA_PROFILES[profile]["equiv_focal_length_mm"]
    return (equiv_focal_mm / FULL_FRAME_SENSOR_WIDTH_MM) * image_width_px


def get_horizontal_fov_deg(profile=None):
    """Approximate horizontal field of view (degrees) for the configured camera profile."""
    profile = _resolve_profile(profile)
    equiv_focal_mm = CAMERA_PROFILES[profile]["equiv_focal_length_mm"]
    half_fov = math.atan((FULL_FRAME_SENSOR_WIDTH_MM / 2) / equiv_focal_mm)
    return math.degrees(2 * half_fov)


if __name__ == "__main__":
    fpx = get_focal_length_px(1280)
    hfov = get_horizontal_fov_deg()
    print(f"Profile: {CAMERA_PROFILES[ACTIVE_CAMERA_PROFILE]['label']}")
    print(f"Focal length @1280px width: {fpx:.1f}px")
    print(f"Approx horizontal FOV: {hfov:.1f} deg")