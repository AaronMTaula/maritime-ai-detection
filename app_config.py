"""
Small persistent config store for the handful of installation-time
settings this program needs (which camera is connected, how high it's
mounted above the waterline, etc).

This is intentionally simple -- a local JSON file -- but the important
part is WHO writes to it: only camera_setup_ui.py, driven by the
Settings window. Nothing in this project should require a client (or
teammate on a fresh machine) to open and edit a .py file to configure
their hardware.
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "app_config.json")

DEFAULTS = {
    "camera_profile": None,   # e.g. "iphone_17_pro_main" -- set via the Settings UI
    "camera_height_m": None,  # camera height above the waterline, metres -- set via the Settings UI
}


def load_config():
    """Load the current config, merged with defaults for any missing keys."""
    if not os.path.exists(CONFIG_PATH):
        return dict(DEFAULTS)
    try:
        with open(CONFIG_PATH, "r") as f:
            data = json.load(f)
        merged = dict(DEFAULTS)
        merged.update(data)
        return merged
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULTS)


def save_config(updates):
    """Merge `updates` into the existing config and persist it."""
    merged = load_config()
    merged.update(updates)
    with open(CONFIG_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    return merged


def is_configured():
    """True once the essential setup fields have been set via the Settings UI."""
    cfg = load_config()
    return cfg.get("camera_profile") is not None and cfg.get("camera_height_m") is not None