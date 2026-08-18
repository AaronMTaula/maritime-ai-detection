"""
Camera Settings UI -- the ONLY place camera setup happens for the end
user. No source code edits required, ever.

Flow:
  1. Try to auto-detect the connected camera (best effort).
  2. Pre-fill the dropdown with the best guess if one was found.
  3. Let the user confirm (or pick a different camera from the dropdown
     if detection was wrong / unavailable).
  4. Ask for the camera's mount height above the waterline -- a single
     number, with a plain-English explanation of what it's for.
  5. Save the result via app_config.py.

This appears automatically on first run (see live_detect.py), and can be
reopened at any time from the main program (press 'C' during live
detection) if the hardware ever changes -- still no code editing.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from camera_profiles import CAMERA_PROFILES
from camera_auto_detect import detect_connected_camera_name, match_profile_from_name
from app_config import save_config, load_config


def run_setup_dialog(force=False):
    """
    Show the Settings window.

    Returns the saved config dict once the user confirms, or None if
    they cancelled (the caller should not proceed to launch the camera
    view in that case).

    If force=False and the app is already fully configured, this
    returns the existing config immediately without showing anything.
    """
    existing = load_config()
    if not force and existing.get("camera_profile") and existing.get("camera_height_m") is not None:
        return existing

    result = {}

    root = tk.Tk()
    root.title("Maritime Detection -- Camera Setup")
    root.geometry("480x380")
    root.resizable(False, False)

    detected_name = detect_connected_camera_name()
    guessed_profile = match_profile_from_name(detected_name, CAMERA_PROFILES)

    tk.Label(root, text="Camera Setup", font=("Helvetica", 15, "bold")).pack(pady=(18, 4))
    tk.Label(root, text="This only needs to be set once for this device.",
             fg="#666666").pack(pady=(0, 12))

    if detected_name:
        tk.Label(root, text=f"Detected camera: {detected_name}", fg="#2a7a2a").pack(pady=(0, 10))
    else:
        tk.Label(root, text="Couldn't auto-detect a camera -- please select below.",
                 fg="#a05a00").pack(pady=(0, 10))

    tk.Label(root, text="Camera:").pack(anchor="w", padx=28)
    profile_keys = list(CAMERA_PROFILES.keys())
    profile_labels = [CAMERA_PROFILES[k]["label"] for k in profile_keys]

    current_profile = existing.get("camera_profile") or guessed_profile
    default_idx = profile_keys.index(current_profile) if current_profile in profile_keys else 0

    profile_var = tk.StringVar(value=profile_labels[default_idx])
    profile_dropdown = ttk.Combobox(
        root, textvariable=profile_var, values=profile_labels,
        state="readonly", width=48
    )
    profile_dropdown.pack(padx=28, pady=(2, 16))

    tk.Label(root, text="Camera height above the waterline (metres):").pack(anchor="w", padx=28)
    tk.Label(
        root,
        text="Measure straight down from where the camera is mounted to\n"
             "the water's surface. A rough estimate is fine.",
        fg="#666666", justify="left"
    ).pack(anchor="w", padx=28, pady=(0, 6))

    height_var = tk.StringVar(value=str(existing.get("camera_height_m") or "1.5"))
    height_entry = tk.Entry(root, textvariable=height_var, width=10)
    height_entry.pack(padx=28, pady=(0, 24), anchor="w")

    def on_confirm():
        try:
            height_m = float(height_var.get())
            if height_m <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid height",
                "Please enter a positive number of metres, e.g. 1.5"
            )
            return

        selected_label = profile_var.get()
        selected_key = profile_keys[profile_labels.index(selected_label)]

        cfg = save_config({
            "camera_profile": selected_key,
            "camera_height_m": height_m,
        })
        result.update(cfg)
        root.destroy()

    def on_cancel():
        root.destroy()

    button_frame = tk.Frame(root)
    button_frame.pack(pady=6)
    tk.Button(
        button_frame, text="Confirm & Start", command=on_confirm,
        bg="#2a7a2a", fg="white", padx=14, pady=4
    ).pack(side="left", padx=6)
    tk.Button(button_frame, text="Cancel", command=on_cancel, padx=14, pady=4).pack(side="left", padx=6)

    root.mainloop()
    return result if result else None


if __name__ == "__main__":
    cfg = run_setup_dialog(force=True)
    print(f"Saved config: {cfg}")