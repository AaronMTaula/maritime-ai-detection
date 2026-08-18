import sys
import cv2
import threading
from ultralytics import YOLO
from app_config import is_configured
from camera_setup_ui import run_setup_dialog
from distance_estimation import estimate_distance, load_focal_length, load_horizontal_fov_deg, KNOWN_WIDTH_M
from horizon_ranging import estimate_distance_via_horizon, detect_horizon_row, get_camera_height_m
from ui_hud import (
    SelectionState, mouse_callback, update_selection,
    draw_detections, draw_info_callout, draw_radar_panel, draw_horizon_line
)

<<<<<<< HEAD

def find_camera_index():
    print("Scanning for available cameras...")
    found = []

    for index in range(10):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"  [{index}] Camera found — Resolution: {int(width)}x{int(height)}")
                found.append(index)
            cap.release()
        else:
            print(f"  [{index}] No camera found")

    if not found:
        print("No cameras detected. Please connect a camera and try again.")
        raise RuntimeError("No camera available")

    print(f"\nAvailable camera indices: {found}")

    if len(found) == 1:
        camera_index = found[0]
        print(f"Only one camera found — using index {camera_index} automatically.")
        return camera_index

    while True:
        selection = input(f"\nSelect camera index {found}: ").strip()
        if selection == "":
            camera_index = found[0]
            print(f"No selection made — defaulting to index {camera_index}.")
            return camera_index
        if selection.isdigit() and int(selection) in found:
            return int(selection)
        print(f"Invalid selection. Please enter one of {found}.")


# ── First-run setup (or if the config is somehow incomplete) ──────
if not is_configured():
    print("First run detected -- opening Camera Setup window...")
    cfg = run_setup_dialog()
    if cfg is None:
        print("Setup was cancelled. Exiting.")
        exit()

=======
# ── First-run setup (or if the config is somehow incomplete) ──────
# This is the ONLY place camera setup happens for the end user -- a
# small window, not a code edit. Press 'C' during live detection to
# reopen this later if the hardware changes.
if not is_configured():
    print("First run detected -- opening Camera Setup window...")
    cfg = run_setup_dialog()
    if cfg is None:
        print("Setup was cancelled. Exiting.")
        exit()

>>>>>>> 618dd2e (Refine classification detection workflow)
# Load INT8 OpenVINO model
model = YOLO('runs/detect/train_v2_151img/weights/best_int8_openvino_model/')

print("Zero-calibration mode: focal length derived from the camera profile "
      "chosen in Camera Setup. No manual code edits needed.")
print(f"Reference vessel width: {KNOWN_WIDTH_M}m | "
      f"Camera height above waterline: {get_camera_height_m()}m")

<<<<<<< HEAD
CAMERA_INDEX = find_camera_index()
print(f"\nUsing camera index {CAMERA_INDEX}")

# Open webcam
if sys.platform.startswith("win"):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(CAMERA_INDEX)
=======
# Open webcam (use cv2.VideoCapture(0, cv2.CAP_DSHOW) instead on Windows)
cap = cv2.VideoCapture(0)
>>>>>>> 618dd2e (Refine classification detection workflow)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Live detection running.")
print("Press Q to quit.")
print("Press S to save a screenshot.")
print("Press C to reopen Camera Setup (e.g. if the camera hardware changes).")
print("Click a ship's bounding box to select it and see distance/heading.")

# Shared state between threads
latest_frame = None
latest_results = None
frame_lock = threading.Lock()
results_lock = threading.Lock()
running = True
screenshot_count = 0

# Click-to-select state (Feature 1 + 2)
selection_state = SelectionState()

<<<<<<< HEAD

=======
>>>>>>> 618dd2e (Refine classification detection workflow)
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

        # Run detection every 5th frame
        if frame_count % 5 == 0:
            results = model(
                frame,
                conf=0.6,
                verbose=False,
                imgsz=320,
                half=True
            )
            with results_lock:
                latest_results = results

<<<<<<< HEAD

=======
>>>>>>> 618dd2e (Refine classification detection workflow)
# ── Start inference thread ────────────────────────────────────────
thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

# ── Window + mouse callback setup ─────────────────────────────────
window_name = 'Maritime Ferry Detection - Live'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback, selection_state)

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

    # Focal length + FOV, computed automatically each frame from the
    # active camera profile (no calibration step, no config file to load)
    frame_width = frame.shape[1]
    focal_length_px = load_focal_length(frame_width)
    hfov_deg = load_horizontal_fov_deg()

    # Horizon detection runs every frame (cheap -- no neural net involved),
    # independent of ship detection. Used for the visual overlay and as
    # an input to the Layer 2 distance refinement below.
    horizon_row, horizon_confidence = detect_horizon_row(frame)

    if current_results is not None:
        boxes = current_results[0].boxes
        boxes_xyxy = [b.xyxy[0].tolist() for b in boxes]
        confidences = [float(b.conf[0]) for b in boxes]
<<<<<<< HEAD
=======
        # model.names supports the full taxonomy (ferry/crane/bird/port_structure)
        # even though only 'ferry' is trained on right now
>>>>>>> 618dd2e (Refine classification detection workflow)
        labels = [model.names[int(b.cls[0])].capitalize() for b in boxes]

        annotated_frame = frame.copy()

        draw_horizon_line(annotated_frame, horizon_row, horizon_confidence)

<<<<<<< HEAD
        selected_idx = update_selection(selection_state, boxes_xyxy)
        draw_detections(annotated_frame, boxes_xyxy, labels, confidences, selected_idx)

=======
        # Resolve clicks + frame-to-frame tracking of the selected ship
        selected_idx = update_selection(selection_state, boxes_xyxy)

        # Draw all boxes, highlighting the selected one
        draw_detections(annotated_frame, boxes_xyxy, labels, confidences, selected_idx)

        # Detection count banner
>>>>>>> 618dd2e (Refine classification detection workflow)
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
<<<<<<< HEAD
            waterline_row_px = sel_box[3]

=======
            waterline_row_px = sel_box[3]  # bottom edge of the bounding box

            # --- Layer 2: try the horizon-dip refinement first ---
>>>>>>> 618dd2e (Refine classification detection workflow)
            if horizon_row is not None:
                horizon_distance_m, _debug = estimate_distance_via_horizon(
                    frame, waterline_row_px, focal_length_px
                )
                if horizon_distance_m is not None:
                    distance_m = horizon_distance_m
                    distance_source = 'Horizon'

<<<<<<< HEAD
=======
            # --- Layer 1: fall back to the always-available width-based estimate ---
>>>>>>> 618dd2e (Refine classification detection workflow)
            if distance_m is None:
                distance_m = estimate_distance(bbox_width_px, focal_length_px)
                distance_source = 'Width-based'

<<<<<<< HEAD
=======
            # Relative bearing from horizontal box offset -- placeholder
            # until real heading data from the team is merged in.
>>>>>>> 618dd2e (Refine classification detection workflow)
            box_cx = (sel_box[0] + sel_box[2]) / 2
            offset_frac = (box_cx - frame_width / 2) / (frame_width / 2)
            bearing_deg = offset_frac * (hfov_deg / 2)

            draw_info_callout(annotated_frame, sel_box, distance_m,
<<<<<<< HEAD
                              heading_text='pending (team merge)',
                              source_label=distance_source)

=======
                               heading_text='pending (team merge)',
                               source_label=distance_source)

        # Radar HUD panel (Feature 2)
>>>>>>> 618dd2e (Refine classification detection workflow)
        draw_radar_panel(annotated_frame, selection_state, distance_m,
                          has_target, bearing_deg=bearing_deg)

    else:
        annotated_frame = frame.copy()
        draw_horizon_line(annotated_frame, horizon_row, horizon_confidence)
        cv2.putText(annotated_frame, 'Initialising...', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Settings info at bottom of frame
    cv2.putText(annotated_frame,
                'OpenVINO INT8 | Threaded | imgsz=320 | conf=0.6 | Click a ship to select',
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

    if key == ord('c'):
<<<<<<< HEAD
=======
        # Reopen Camera Setup mid-session -- still no code edits, just a
        # small window. The main loop pauses while it's open; focal
        # length and camera height are re-read from config automatically
        # on the next frame once it closes.
>>>>>>> 618dd2e (Refine classification detection workflow)
        print("Opening Camera Setup...")
        run_setup_dialog(force=True)

# Cleanup
running = False
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")