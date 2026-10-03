"""
Persistent store for named camera configurations, plus the settings
that are active for the current session.

Everything here is written by the Settings window (camera_setup_ui.py).


Stored file layout (app_config.json):
    {
      "configs": {
        "Viaduct ferry": {"camera_profile": "iphone_17_pro_main",
                          "camera_height_m": 3.0,
                          "horizon_assist": false},
        ...
      },
      "last_used": "Viaduct ferry"
    }

Older single-config files (flat "camera_profile"/"camera_height_m" keys)
are migrated automatically into a config called "Default".
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_config.json")

DEFAULT_SETTINGS = {
    "camera_profile": None,    # key from camera_profiles.CAMERA_PROFILES
    "camera_height_m": None,   # only used by horizon-assisted ranging
    "horizon_assist": False,   # optional open-water refinement, off by default
    "device_index": None,      # physical camera device index (int), chosen in Settings
    "device_backend": None,    # OpenCV backend constant (int) that opened it, or None
    "device_label": None,      # human-readable label shown in Settings, e.g. "Camera 0 -- 1920x1080 (CAP_AVFOUNDATION)"
}

# Settings in use for the current session (set when the user presses
# Start in the Settings window). Kept in memory so per-frame code never
# has to read the file.
_active_settings = None


def _normalise(settings):
    merged = dict(DEFAULT_SETTINGS)
    merged.update({k: settings.get(k) for k in DEFAULT_SETTINGS if k in settings})
    merged["horizon_assist"] = bool(merged["horizon_assist"])
    return merged


def _load_store():
    store = {"configs": {}, "last_used": None}
    if not os.path.exists(CONFIG_PATH):
        return store
    try:
        with open(CONFIG_PATH, "r") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return store

    if "configs" in raw:
        store["configs"] = {name: _normalise(cfg) for name, cfg in raw["configs"].items()}
        store["last_used"] = raw.get("last_used")
    elif raw.get("camera_profile"):
        # Legacy single-config file -> migrate to a named config.
        store["configs"]["Default"] = _normalise({
            "camera_profile": raw.get("camera_profile"),
            "camera_height_m": raw.get("camera_height_m"),
            "horizon_assist": False,
        })
        store["last_used"] = "Default"

    if store["last_used"] not in store["configs"]:
        store["last_used"] = None
    return store


def _write_store(store):
    with open(CONFIG_PATH, "w") as f:
        json.dump(store, f, indent=2)


# ── Saved configurations ──────────────────────────────────────────

def list_configs():
    """Return {name: settings} for every saved configuration."""
    return _load_store()["configs"]


def get_last_used():
    """Name of the most recently used saved configuration, or None."""
    return _load_store()["last_used"]


def save_config(name, settings):
    """Save (or overwrite) a named configuration and mark it as last used."""
    store = _load_store()
    store["configs"][name] = _normalise(settings)
    store["last_used"] = name
    _write_store(store)


def delete_config(name):
    store = _load_store()
    store["configs"].pop(name, None)
    if store["last_used"] == name:
        store["last_used"] = None
    _write_store(store)


def set_last_used(name):
    store = _load_store()
    if name in store["configs"]:
        store["last_used"] = name
        _write_store(store)


# ── Session settings ──────────────────────────────────────────────

def set_active_settings(settings):
    """Use these settings for the rest of this session."""
    global _active_settings
    _active_settings = _normalise(settings)


def get_active_settings():
    """
    Settings for the current session. Falls back to the last-used saved
    configuration, then to defaults, if the Settings window hasn't been
    run yet (e.g. when a module is run on its own for testing).
    """
    if _active_settings is not None:
        return dict(_active_settings)
    store = _load_store()
    if store["last_used"]:
        return dict(store["configs"][store["last_used"]])
    return dict(DEFAULT_SETTINGS)