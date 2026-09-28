"""
Settings window -- the ONLY place camera setup happens for the end user.
No source code or file edits required, ever.

Shown on every launch, and again any time the user clicks "Settings"
inside the main program. It lets the user:
  * pick from their saved configurations (last used is pre-selected, so
    starting is one click / Enter),
  * create a new configuration, or edit and re-save an existing one,
  * delete configurations they no longer need.

A configuration is: a camera, plus (optionally) horizon-assisted ranging
with the camera's mount height. Horizon assist is off by default and the
height box stays greyed out until it's ticked, so most users only ever
choose a camera.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from camera_profiles import CAMERA_PROFILES
from camera_auto_detect import detect_connected_camera_name, match_profile_from_name
from app_config import (
    list_configs, save_config, delete_config,
    get_last_used, set_last_used, set_active_settings,
)

DEFAULT_HEIGHT_TEXT = "1.5"


def run_setup_dialog():
    """
    Show the Settings window.

    Returns True if the user pressed Start (the chosen settings are then
    active for this session), or False if they closed/cancelled it.
    """
    profile_keys = list(CAMERA_PROFILES.keys())
    profile_labels = [CAMERA_PROFILES[k]["label"] for k in profile_keys]

    detected_name = detect_connected_camera_name()
    guessed_profile = match_profile_from_name(detected_name, CAMERA_PROFILES)
    default_profile = guessed_profile if guessed_profile in profile_keys else profile_keys[0]

    outcome = {"started": False}

    root = tk.Tk()
    root.title("Maritime Detection -- Settings")
    root.geometry("560x560")
    root.resizable(False, False)

    tk.Label(root, text="Camera Settings", font=("Helvetica", 15, "bold")).pack(pady=(14, 2))
    if detected_name:
        tk.Label(root, text=f"Detected camera: {detected_name}", fg="#2a7a2a").pack()
    else:
        tk.Label(root, text="No camera auto-detected -- choose one below.", fg="#a05a00").pack()

    # ── Saved configurations list ─────────────────────────────────
    list_frame = ttk.LabelFrame(root, text="Saved configurations")
    list_frame.pack(fill="x", padx=20, pady=(12, 6))

    listbox = tk.Listbox(list_frame, height=5, exportselection=False, activestyle="none")
    listbox.pack(side="left", fill="x", expand=True, padx=(8, 4), pady=8)

    list_buttons = tk.Frame(list_frame)
    list_buttons.pack(side="right", padx=(4, 8), pady=8)

    # ── Configuration form ────────────────────────────────────────
    form = ttk.LabelFrame(root, text="Configuration")
    form.pack(fill="x", padx=20, pady=6)

    tk.Label(form, text="Name:").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
    name_var = tk.StringVar()
    tk.Entry(form, textvariable=name_var, width=28).grid(row=0, column=1, sticky="w", pady=(8, 2))

    tk.Label(form, text="Camera:").grid(row=1, column=0, sticky="w", padx=8, pady=2)
    profile_var = tk.StringVar(value=CAMERA_PROFILES[default_profile]["label"])
    ttk.Combobox(form, textvariable=profile_var, values=profile_labels,
                 state="readonly", width=32).grid(row=1, column=1, sticky="w", pady=2)

    horizon_var = tk.BooleanVar(value=False)
    tk.Checkbutton(form, text="Use horizon-assisted ranging (open water only)",
                   variable=horizon_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(8, 0))
    tk.Label(form, text="Leave off on harbours and ferries -- distance still works without it.",
             fg="#666666").grid(row=3, column=0, columnspan=2, sticky="w", padx=26)

    tk.Label(form, text="Camera height above water (m):").grid(row=4, column=0, sticky="w", padx=8, pady=(6, 8))
    height_var = tk.StringVar(value=DEFAULT_HEIGHT_TEXT)
    height_entry = tk.Entry(form, textvariable=height_var, width=8, state="disabled")
    height_entry.grid(row=4, column=1, sticky="w", pady=(6, 8))

    def sync_height_state(*_):
        height_entry.configure(state="normal" if horizon_var.get() else "disabled")
    horizon_var.trace_add("write", sync_height_state)

    status_var = tk.StringVar(value="")
    tk.Label(root, textvariable=status_var, fg="#2a7a2a").pack(pady=(2, 0))

    # ── Helpers ───────────────────────────────────────────────────
    def refresh_list(select_name=None):
        listbox.delete(0, "end")
        names = sorted(list_configs().keys(), key=str.lower)
        for n in names:
            listbox.insert("end", n)
        if select_name in names:
            idx = names.index(select_name)
            listbox.selection_set(idx)
            listbox.see(idx)

    def selected_name():
        sel = listbox.curselection()
        return listbox.get(sel[0]) if sel else None

    def load_into_form(name):
        cfg = list_configs().get(name)
        if not cfg:
            return
        name_var.set(name)
        key = cfg.get("camera_profile")
        if key in profile_keys:
            profile_var.set(CAMERA_PROFILES[key]["label"])
        horizon_var.set(bool(cfg.get("horizon_assist")))
        height = cfg.get("camera_height_m")
        height_var.set(str(height) if height else DEFAULT_HEIGHT_TEXT)
        sync_height_state()

    def clear_form():
        name_var.set("")
        profile_var.set(CAMERA_PROFILES[default_profile]["label"])
        horizon_var.set(False)
        height_var.set(DEFAULT_HEIGHT_TEXT)
        sync_height_state()

    def collect_settings():
        """Read the form; returns a settings dict, or None after showing an error."""
        key = profile_keys[profile_labels.index(profile_var.get())]
        height = None
        if horizon_var.get():
            try:
                height = float(height_var.get())
                if height <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid height",
                                     "Please enter the camera height above the water in metres, e.g. 1.5")
                return None
        return {"camera_profile": key, "camera_height_m": height,
                "horizon_assist": horizon_var.get()}

    # ── Button handlers ───────────────────────────────────────────
    def on_select(_event=None):
        name = selected_name()
        if name:
            load_into_form(name)
            status_var.set("")

    def on_new():
        listbox.selection_clear(0, "end")
        clear_form()
        status_var.set("")

    def on_delete():
        name = selected_name()
        if not name:
            messagebox.showinfo("Delete", "Select a saved configuration to delete.")
            return
        if messagebox.askyesno("Delete configuration", f"Delete '{name}'? This can't be undone."):
            delete_config(name)
            refresh_list()
            clear_form()
            status_var.set(f"Deleted '{name}'.")

    def on_save():
        name = name_var.get().strip()
        if not name:
            messagebox.showerror("Name needed", "Give this configuration a name so you can find it later.")
            return
        settings = collect_settings()
        if settings is None:
            return
        if name in list_configs() and name != selected_name():
            if not messagebox.askyesno("Overwrite?", f"'{name}' already exists. Replace it?"):
                return
        save_config(name, settings)
        refresh_list(select_name=name)
        status_var.set(f"Saved '{name}'.")

    def on_start():
        settings = collect_settings()
        if settings is None:
            return
        set_active_settings(settings)
        name = selected_name()
        if name:
            set_last_used(name)
        outcome["started"] = True
        root.destroy()

    def on_cancel():
        root.destroy()

    tk.Button(list_buttons, text="New", width=8, command=on_new).pack(pady=2)
    tk.Button(list_buttons, text="Delete", width=8, command=on_delete).pack(pady=2)
    listbox.bind("<<ListboxSelect>>", on_select)

    tk.Label(root, text="Start uses what's shown above. Click Save to keep it for next time.",
             fg="#666666").pack(pady=(4, 0))

    buttons = tk.Frame(root)
    buttons.pack(pady=12)
    tk.Button(buttons, text="Save configuration", command=on_save, padx=10, pady=4).pack(side="left", padx=6)
    tk.Button(buttons, text="Start", command=on_start, bg="#2a7a2a", fg="white",
              padx=22, pady=4).pack(side="left", padx=6)
    tk.Button(buttons, text="Cancel", command=on_cancel, padx=10, pady=4).pack(side="left", padx=6)

    root.bind("<Return>", lambda _e: on_start())
    root.protocol("WM_DELETE_WINDOW", on_cancel)

    # ── Initial state: pre-select the last used configuration ─────
    last_used = get_last_used()
    refresh_list(select_name=last_used)
    if last_used:
        load_into_form(last_used)
    else:
        clear_form()
        if not list_configs():
            name_var.set("Default")

    # Bring the window to the front (Tk windows can open behind the terminal on macOS)
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))

    root.mainloop()
    return outcome["started"]


if __name__ == "__main__":
    print("Started" if run_setup_dialog() else "Cancelled")