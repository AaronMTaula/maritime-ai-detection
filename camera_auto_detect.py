"""
Best-effort automatic camera detection.

The goal is to let the Settings UI pre-fill (or in the best case, skip
entirely) the camera selection step by detecting what's actually
connected, e.g. an iPhone via Continuity Camera. This is explicitly
"best effort": if it can't confidently detect anything, it returns None
and the Settings UI simply falls back to asking the user to pick from a
dropdown -- so the program still never requires a source code edit,
even if detection fails.

NOTE: the macOS parsing below is a heuristic based on the general
structure of `system_profiler SPCameraDataType` output (each connected
camera is listed as a colon-terminated name, e.g. "Naz's iPhone Camera:",
followed by indented properties like Model ID / Unique ID). It hasn't
been validated against every macOS version or camera type, so treat it
as a convenience, not a guarantee -- the manual dropdown in the Settings
UI is the reliable path either way.
"""

import subprocess
import sys


def detect_connected_camera_name():
    """Return a human-readable connected camera name string, or None."""
    try:
        if sys.platform == "darwin":
            return _detect_macos()
    except Exception:
        # Detection is a convenience, not a requirement -- never let a
        # detection failure block the user from using the Settings UI.
        pass
    return None


def _detect_macos():
    result = subprocess.run(
        ["system_profiler", "SPCameraDataType"],
        capture_output=True, text=True, timeout=5
    )
    output = result.stdout
    known_property_keys = {"Model ID", "Unique ID", "Camera"}

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.endswith(":"):
            continue
        candidate = line[:-1].strip()
        if not candidate or candidate in known_property_keys:
            continue
        # Skip the top-level "Camera:" section header itself
        if candidate == "Camera":
            continue
        return candidate
    return None


def match_profile_from_name(camera_name, camera_profiles):
    """
    Match a detected camera name string to a known profile key.
    Returns None if there's no confident match -- the Settings UI will
    then just leave the dropdown for the user to choose manually.
    """
    if not camera_name:
        return None
    name_lower = camera_name.lower()
    if "iphone" in name_lower:
        # Closest/only known iPhone profile right now. Update this
        # matching logic as more iPhone profiles are added.
        if "iphone_17_pro_main" in camera_profiles:
            return "iphone_17_pro_main"
    return None