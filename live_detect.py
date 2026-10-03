import os
import cv2
import threading
from collections import deque

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from ultralytics import YOLO
from camera_setup_ui import run_setup_dialog
from camera_device_scan import open_selected_camera, get_backend_name
from app_config import get_active_settings
from distance_estimation import estimate_distance, load_focal_length, load_horizontal_fov_deg, KNOWN_WIDTH_M
from horizon_ranging import estimate_distance_via_horizon, detect_horizon_row, get_camera_height_m, horizon_assist_enabled
from ui_hud import (
    SelectionState, mouse_callback, update_selection,
    draw_detections, draw_info_callout, draw_radar_panel, draw_horizon_line,
    draw_settings_button, draw_compass, compute_relative_direction
)

# ── Platform note ───────────────────────────────────────────────────
# This file is cross platform: it detects Windows vs macOS
# automatically via os.name (see DEFAULT_MODEL_PATH below and
# camera_device_scan.get_platform_backends()). There is nothing in THIS
# file that needs manual editing to run on Windows vs macOS. The only
# per-platform steps anyone needs are in README.md (creating the virtual
# environment and installing requirements) -- see that file for more details


def get_model_imgsz(model_path):
    """Read the exported model's expected image size from metadata.

    Only OpenVINO export folders carry a metadata.yaml; a raw .pt
    checkpoint never has one, so that case isn't a warning-worthy
    problem -- just silently use the standard 640 default.
    """
    if model_path.lower().endswith(".pt"):
        return 640

    if yaml is None:
        print("WARNING: PyYAML is unavailable; using model input size 640.")
        return 640

    metadata_path = os.path.join(model_path, "metadata.yaml")
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = yaml.safe_load(metadata_file) or {}
    except (FileNotFoundError, NotADirectoryError):
        print(f"WARNING: Model metadata unavailable at {metadata_path}; using imgsz=640.")
        return 640
    except yaml.YAMLError as error:
        print(f"WARNING: Could not parse {metadata_path}: {error}; using imgsz=640.")
        return 640

    imgsz = metadata.get("imgsz")
    if isinstance(imgsz, (list, tuple)):
        return int(imgsz[0]) if imgsz else 640
    if isinstance(imgsz, str):
        try:
            return int(imgsz)
        except ValueError:
            print(f"WARNING: Invalid model imgsz {imgsz!r}; using imgsz=640.")
            return 640
    if isinstance(imgsz, int):
        return imgsz
    print("WARNING: Model metadata has no valid imgsz; using imgsz=640.")
    return 640



# ── Camera settings ───────────────────────────────────────────────
if not run_setup_dialog():
    print("Setup was cancelled. Exiting.")
    exit()

DEFAULT_MODEL_PATH = (
    "runs/detect/train_v3_more_data/ferry_v3_map0.924_recall0.889.pt"
    if os.name == "posix"
    else "runs/detect/train_v3_more_data/"
         "ferry_v3_map0.924_recall0.889_int8_openvino_model/"
)
MODEL_PATH = os.environ.get("MARITIME_MODEL_PATH", DEFAULT_MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Detection model not found: {MODEL_PATH}. "
        "Set MARITIME_MODEL_PATH to a compatible local model."
    )
MODEL_IMGSZ = get_model_imgsz(MODEL_PATH)
MODEL_BACKEND_NAME = (
    "PyTorch" if MODEL_PATH.lower().endswith(".pt")
    else "OpenVINO INT8" if "openvino" in MODEL_PATH.lower()
    else "YOLO"
)

print(f"Running script: {os.path.abspath(__file__)}")
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print(f"Model loaded successfully. Expected input size: {MODEL_IMGSZ}x{MODEL_IMGSZ}.")

try:
    renamed = {
        idx: "Ferry" for idx, name in model.names.items()
        if str(name).strip().lower() == "item"
    }
    if renamed:
        model.names.update(renamed)
except Exception as exc:  # pragma: no cover - defensive, should not happen
    print(f"WARNING: Could not rename placeholder class names: {exc}")

print("Zero-calibration mode: focal length derived from the camera profile "
      "chosen in Settings. No manual code edits needed.")
print(f"Reference vessel width: {KNOWN_WIDTH_M}m | "
      f"Camera height above waterline: {get_camera_height_m()}m | "
      f"Horizon-assisted ranging: {'on' if horizon_assist_enabled() else 'off'}")

# ── Open the physical camera chosen in Settings ─────────────────────
# Which device to open was already decided in the Settings window (no
# terminal prompt here) - see camera_setup_ui.py's "Camera device"
# dropdown. If it fails to open (e.g. unplugged since Settings closed),
# we reopen Settings rather than falling back to a terminal input.
active_settings = get_active_settings()
camera_index = active_settings.get("device_index")
camera_backend = active_settings.get("device_backend")

cap = None
while cap is None:
    if camera_index is None:
        print("No camera device is configured.")
    else:
        print(f"\nOpening camera {camera_index} "
              f"(backend: {get_backend_name(camera_backend)})...")
        cap, opened_backend = open_selected_camera(camera_index, camera_backend)
        if cap is not None:
            break
        print(f"Could not open camera {camera_index}. It may be disconnected "
              "or in use by another application.")

    print("Opening Settings so you can choose a different camera...")
    if not run_setup_dialog():
        print("Setup was cancelled. Exiting.")
        exit()
    active_settings = get_active_settings()
    camera_index = active_settings.get("device_index")
    camera_backend = active_settings.get("device_backend")

requested_width, requested_height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera opened successfully at {actual_width}x{actual_height} "
      f"using {get_backend_name(opened_backend)}.")
if (actual_width, actual_height) != (requested_width, requested_height):
    print(f"Note: requested {requested_width}x{requested_height}; continuing with the "
          f"camera's actual resolution {actual_width}x{actual_height}.")

print("Live detection running. Compass shows relative bearing based on "
      "screen position (not true heading until the team's heading work is merged).")
print("Press Q to quit.")
print("Press S to save a screenshot.")
print("Press C to reopen Settings (e.g. to change camera or camera device).")
print("Click a ship's bounding box to select it and see distance/heading.")
print("Click the Settings button (top-right) to do the same with the mouse.")

# Shared state between threads
latest_frame = None
latest_results = None
frame_lock = threading.Lock()
results_lock = threading.Lock()
running = True
screenshot_count = 0

# Click-to-select state (Feature 1 + 2)
selection_state = SelectionState()
direction_history = deque(maxlen=5)
current_display_direction = "center"


# ── Inference thread ──────────────────────────────────────────────
def inference_thread():
    global latest_results, running
    frame_count = 0

    while running:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame_count += 1

        if frame_count % 5 == 0:
            results = model(
                frame,
                conf=0.6,
                verbose=False,
                imgsz=MODEL_IMGSZ,
                half=not MODEL_PATH.lower().endswith(".pt")
            )
            with results_lock:
                latest_results = results


thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

# ── Window + mouse callback setup ─────────────────────────────────
window_name = 'Maritime Ferry Detection - Live'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1500, 900)
cv2.setMouseCallback(window_name, mouse_callback, selection_state)


def open_settings():
    """Reopen the Settings window from inside the live view.

    If the user picks a different camera device while here, actually
    switch to it (release the old capture, open the new one) -- not just
    redraw the window with the old camera still running.
    """
    global cap, camera_index, camera_backend, actual_width, actual_height

    cv2.destroyAllWindows()
    for _ in range(5):
        cv2.waitKey(1)

    if run_setup_dialog():
        new_settings = get_active_settings()
        new_index = new_settings.get("device_index")
        new_backend = new_settings.get("device_backend")
        if (new_index, new_backend) != (camera_index, camera_backend):
            print(f"Switching to camera {new_index}...")
            new_cap, _opened_backend = open_selected_camera(new_index, new_backend)
            if new_cap is not None:
                cap.release()
                cap = new_cap
                camera_index, camera_backend = new_index, new_backend
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Now using camera {camera_index} at {actual_width}x{actual_height}.")
            else:
                print(f"Could not open camera {new_index}; keeping the current camera.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1500, 900)
    cv2.setMouseCallback(window_name, mouse_callback, selection_state)
    selection_state.settings_requested = False
    selection_state.click_point = None


# ── Main display loop ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    with frame_lock:
        latest_frame = frame.copy()

    with results_lock:
        current_results = latest_results

    frame_width = frame.shape[1]
    focal_length_px = load_focal_length(frame_width)
    hfov_deg = load_horizontal_fov_deg()

    use_horizon = horizon_assist_enabled()
    if use_horizon:
        horizon_row, horizon_confidence = detect_horizon_row(frame)
    else:
        horizon_row, horizon_confidence = None, 0.0

    if current_results is not None:
        boxes = current_results[0].boxes
        boxes_xyxy = [b.xyxy[0].tolist() for b in boxes]
        confidences = [float(b.conf[0]) for b in boxes]
        labels = [model.names[int(b.cls[0])].capitalize() for b in boxes]

        annotated_frame = frame.copy()
        draw_horizon_line(annotated_frame, horizon_row, horizon_confidence)

        selected_idx = update_selection(selection_state, boxes_xyxy)
        draw_detections(annotated_frame, boxes_xyxy, labels, confidences, selected_idx)

        detections = len(boxes_xyxy)
        color = (0, 255, 0) if detections > 0 else (0, 0, 255)
        cv2.putText(annotated_frame, f'Ferries detected: {detections}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        distance_m = None
        distance_source = None
        bearing_deg = 0.0
        has_target = selected_idx is not None

        if has_target:
            sel_box = boxes_xyxy[selected_idx]
            bbox_width_px = sel_box[2] - sel_box[0]
            waterline_row_px = sel_box[3]
            raw_direction = compute_relative_direction(
                sel_box, frame_width, frame.shape[0]
            )
            direction_history.append(raw_direction)
            most_common_direction = max(
                set(direction_history), key=direction_history.count
            )
            if direction_history.count(most_common_direction) >= 3:
                current_display_direction = most_common_direction

            if use_horizon and horizon_row is not None:
                horizon_distance_m, _debug = estimate_distance_via_horizon(
                    frame, waterline_row_px, focal_length_px
                )
                if horizon_distance_m is not None:
                    distance_m = horizon_distance_m
                    distance_source = 'Horizon'

            if distance_m is None:
                distance_m = estimate_distance(bbox_width_px, focal_length_px)
                distance_source = 'Width-based'

            box_cx = (sel_box[0] + sel_box[2]) / 2
            offset_frac = (box_cx - frame_width / 2) / (frame_width / 2)
            bearing_deg = offset_frac * (hfov_deg / 2)

            draw_info_callout(annotated_frame, sel_box, distance_m,
                              heading_text=f'Screen-relative: {current_display_direction}',
                              source_label=distance_source)

        draw_radar_panel(annotated_frame, selection_state, distance_m,
                          has_target, bearing_deg=bearing_deg)

        # Compass only shows a direction while a ship is actually
        # selected; otherwise it's reset so it doesn't keep pointing at
        # whatever was last selected.
        compass_direction = current_display_direction if has_target else None

    else:
        annotated_frame = frame.copy()
        draw_horizon_line(annotated_frame, horizon_row, horizon_confidence)
        cv2.putText(annotated_frame, 'Initialising...', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        compass_direction = None

    # ── HUD layout: each element has its own corner/zone so nothing
    # overlaps -- settings top-right, compass top-left (below the
    # detection count), radar bottom-right (drawn above), status bar
    # along the bottom.
    draw_settings_button(annotated_frame, selection_state)
    draw_compass(annotated_frame, (130, 190), 70, direction=compass_direction)

    cv2.putText(annotated_frame,
                f'Camera: {camera_index} | {MODEL_BACKEND_NAME} | '
                f'{actual_width}x{actual_height} | '
                f'imgsz={MODEL_IMGSZ} | conf=0.6 | Click a ship to select',
                (20, annotated_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    cv2.imshow(window_name, annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        running = False
        break

    if key == ord('s'):
        screenshot_path = f'screenshot_{screenshot_count}.jpg'
        cv2.imwrite(screenshot_path, annotated_frame)
        print(f"Screenshot saved: {screenshot_path}")
        screenshot_count += 1

    if key == ord('c') or selection_state.settings_requested:
        print("Opening Settings...")
        open_settings()

# Cleanup
running = False
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")