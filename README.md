# Maritime AI Detection

### Auckland University of Technology | Navantia Australia Collaboration Project

## Project Status

**Complete.** All feature branches (classification, distance estimation, and
heading/orientation groundwork) have been merged into `main` and field-tested
on the water at Auckland's Viaduct Harbour.

## Overview

A YOLOv8-based computer vision system for detecting, classifying, and
estimating the position and heading of maritime vessels from a live camera
feed. The program runs entirely through its own UI — launching it (`python
live_detect.py`) is the only step that touches a terminal; camera selection,
distance-estimation settings, and reconfiguration all happen in on-screen
windows, with nothing requiring a source code or config file edit.

## Current Model Performance (v3, `ferry_v3_map0.924_recall0.889`)

| Metric | Result |
|---|---|
| Precision | ~0.88 |
| Recall | 0.889 |
| mAP50 | 0.924 |

Trained with `train_ferry_model.py` (see **Training** below). The exported
model folder name records its own headline recall/mAP50 figures.

## Features

- **Live ship detection** (YOLOv8, PyTorch on macOS / OpenVINO INT8 on
  Windows) with on-screen bounding boxes, class label, and confidence.
- **Click-to-select**: click a ship's bounding box to highlight it (colour
  change) and see its distance and relative bearing. Click again to
  deselect. Selection tracks the same ship across frames.
- **Distance estimation** with zero manual calibration — focal length is
  derived automatically from the camera's known hardware profile (see
  `camera_profiles.py`). An optional **horizon-assisted** mode refines
  this further using the sea horizon, for open-water use only (off by
  default — see the Settings window).
- **Radar-style HUD panel** showing the selected ship's distance as a
  rotating-sweep blip, colour-coded by proximity.
- **Compass** showing the selected ship's position relative to the
  camera's view (left/right/up/down of frame centre). This is
  screen-relative, not a true magnetic heading — see **Known
  limitations** below.
- **Settings window** (Tkinter): pick a physical camera device, a camera
  profile, and (optionally) horizon-assisted ranging with mount height.
  Configurations can be named, saved, reloaded, and deleted, so
  switching between e.g. an iPhone and a laptop webcam is a dropdown
  choice, not a code edit. Reachable any time via the on-screen
  **Settings** button or the **C** key.

## Team

- Aaron Taula — Software Developer / Tech Lead
- Naz Janif — Team Lead / Software Developer
- Tyler Halliday — In Loving Memory
- Zhiyuan Zhang — AI Lead

## Supervisor

Kien Tran — Auckland University of Technology

## Client

Navantia Australia

---

## Quick Start

Both Winodows and MacOS platforms run the program the same way once set up:

```bash
python live_detect.py
```

That's it — no camera index, model path, or other setup needs to be typed
in. The Settings window handles all of that on first launch (and again any
time you want to change it).

The two platforms differ only in how the Python environment is created —
see below.

## Setup — macOS

- macOS uses the included PyTorch (`.pt`) model by default — this is
  chosen automatically (see the platform note at the top of
  `live_detect.py`); you don't need to change anything.
- `requirements-openvino.txt` is **optional** on macOS (only needed if
  you want to export models to OpenVINO format yourself, e.g. via
  `train_ferry_model.py`).
- The first time you run it, grant your terminal/VS Code camera access
  if macOS prompts for it (System Settings → Privacy & Security →
  Camera).

## Setup — Windows

- Windows uses the included OpenVINO INT8 model by default, again
  chosen automatically — no path to edit.
- `requirements-openvino.txt` is **required** here (not optional): the
  `openvino` package is needed at *run time* to load and run that model,
  not just to export one. Skipping this install will cause the model
  load to fail on Windows.
- The first time you run it, allow camera access if Windows prompts for
  it (Settings → Privacy & security → Camera).

### Nothing else needs platform-specific edits

Camera backend selection (`CAP_DSHOW`/`CAP_MSMF` on Windows vs
`CAP_AVFOUNDATION` on macOS) and the model path are both detected
automatically at runtime — see `camera_device_scan.py` and the platform
note at the top of `live_detect.py`. If you ever do need to override the
model path manually (e.g. to test a different export), set the
`MARITIME_MODEL_PATH` environment variable rather than editing the script.

---

## Using the program

On launch, the **Settings** window appears:

- Pick (or rescan for) your camera device, choose the matching camera
  profile, and optionally enable horizon-assisted ranging with your
  camera's height above the waterline.
- **Save configuration** names and stores the current setup so it's
  listed next time — handy if you switch between devices (e.g. an
  iPhone vs a laptop webcam) regularly.
- **Start** launches the live view using whatever's currently shown,
  saved or not.

Once running:

| Key / Action | Effect |
|---|---|
| Click a ship's bounding box | Select it (shows distance, bearing, compass) |
| Click the same box again | Deselect |
| **Settings** button (top-right) or **C** | Reopen Settings — switching camera device here takes effect immediately |
| **S** | Save a screenshot |
| **Q** | Quit |

## Known limitations

- **Compass is screen-relative, not true heading.** It shows whether the
  selected ship is towards the left/right/top/bottom of the camera's
  current view, not a magnetic/world bearing. True heading estimation
  (OBB-based) is a separate, not-yet-merged workstream — once available,
  it's a drop-in replacement for the `direction` value `live_detect.py`
  passes to the compass (see `ui_hud.py`'s `draw_compass`).
- **Horizon-assisted ranging needs open water.** In enclosed harbours
  (hills, buildings, wharves in view), there's no true sea horizon to
  detect, so this mode is off by default — the always-available
  width-based estimate carries distance accuracy everywhere else.
- **Detection range**: field testing from a handheld phone on a ferry
  reliably detected vessels out to ~144m. This is mostly limited by the
  model's inference input size (`imgsz=320` on the INT8 export); a
  640-sized export and/or more long-range training images would likely
  extend this.

## Training

The dataset configuration (`data.yaml`) uses repository-relative paths, so
training works the same on both platforms:

```bash
python train_ferry_model.py --epochs 80 --imgsz 640 --batch 8 --device cpu
```

If OpenVINO and NNCF are installed, the script also exports an OpenVINO
model after training; otherwise it prints that the export was skipped.

## Repository housekeeping

- `app_config.json` (per-machine Settings — camera device, profile, mount
  height) and `camera_config.json` (output of the old, no-longer-used
  manual calibration tool) are personal/local and now in `.gitignore`.
  If either is already tracked in the repo from before, remove it from
  version control with `git rm --cached app_config.json
  camera_config.json` so each teammate's own settings don't overwrite
  anyone else's.

