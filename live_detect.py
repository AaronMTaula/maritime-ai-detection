import contextlib
import io
import os
import cv2
import threading

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from ultralytics import YOLO

# Distance estimation (from v2) — requires camera_config.json produced by
# calibrate_focal_length.py. If that hasn't been run yet, distance display
# is disabled but detection still works.
try:
    from distance_estimation import estimate_distance, load_focal_length, KNOWN_WIDTH_M
    DISTANCE_MODULE_AVAILABLE = True
except ImportError:
    DISTANCE_MODULE_AVAILABLE = False

# ── Suppress OpenCV backend warnings ─────────────────────────────
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)

# ── Camera Backend Utilities ──────────────────────────────────────
def get_backend_name(backend):
    if backend is None:
        return "default"
    backend_names = {
        cv2.CAP_ANY: "CAP_ANY",
        cv2.CAP_DSHOW: "CAP_DSHOW",
        cv2.CAP_MSMF: "CAP_MSMF",
        cv2.CAP_VFW: "CAP_VFW",
    }
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backend_names[cv2.CAP_AVFOUNDATION] = "CAP_AVFOUNDATION"
    return backend_names.get(backend, str(backend))


def get_platform_backends():
    """Return backends appropriate for the current OS."""
    backends = [None]
    if os.name == "nt":
        backends += [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif hasattr(cv2, "CAP_AVFOUNDATION"):
        backends += [cv2.CAP_AVFOUNDATION]
    else:
        backends += [cv2.CAP_ANY]
    return backends


def try_open_camera(index, backend=None):
    """Attempt to open a camera silently. Returns (cap, frame) or None."""
    with contextlib.redirect_stderr(io.StringIO()):
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None
    return cap, frame


def find_available_cameras(max_devices=10):
    """Scan for available cameras across all platform backends."""
    backends = get_platform_backends()
    discovered = []
    seen = set()

    for index in range(max_devices):
        for backend in backends:
            result = try_open_camera(index, backend)
            if result is None:
                continue
            cap, frame = result
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            key = (index, backend)
            if key not in seen:
                seen.add(key)
                discovered.append((index, backend, width, height))
            break

    return discovered


def open_selected_camera(index, preferred_backend=None):
    """Open a camera with fallback across backends."""
    backends = []
    if preferred_backend is not None:
        backends.append(preferred_backend)
    for b in get_platform_backends():
        if b not in backends:
            backends.append(b)

    for backend in backends:
        result = try_open_camera(index, backend)
        if result is not None:
            cap, _ = result
            return cap, backend

    return None, None


# ── Model Configuration ─────────────────────────────────────────────
def get_model_imgsz(model_path):
    """Read the exported model metadata and return the expected input size."""
    metadata_path = os.path.join(model_path, "metadata.yaml")
    if not os.path.exists(metadata_path):
        return 640

    try:
        if yaml is None:
            raise ImportError("PyYAML is not available")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f) or {}
    except Exception:
        return 640

    imgsz = metadata.get("imgsz")
    if isinstance(imgsz, (list, tuple)):
        if not imgsz:
            return 640
        return int(imgsz[0])
    if isinstance(imgsz, str):
        try:
            return int(imgsz)
        except ValueError:
            return 640
    if isinstance(imgsz, int):
        return imgsz
    return 640


MODEL_PATH = os.environ.get(
    "MARITIME_MODEL_PATH",
    "runs/detect/train_v3_more_data/ferry_v3_map0.924_recall0.889_int8_openvino_model/"
)
MODEL_IMGSZ = get_model_imgsz(MODEL_PATH)

print(f"Loading model from: {MODEL_PATH}")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    model = YOLO(MODEL_PATH)
print(f"Model loaded successfully. Expected input size: {MODEL_IMGSZ}x{MODEL_IMGSZ}.")

# ── Distance Estimation Setup ──────────────────────────────────────
FOCAL_LENGTH_PX = None
if DISTANCE_MODULE_AVAILABLE:
    try:
        FOCAL_LENGTH_PX = load_focal_length()
        print(f"Loaded focal length: {FOCAL_LENGTH_PX:.1f}px "
              f"(reference width: {KNOWN_WIDTH_M}m)")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Distance estimation will be disabled until calibration is run.")
else:
    print("WARNING: distance_estimation module not found — distance display disabled.")

# NOTE: focal length calibration is tied to the resolution it was calibrated
# at. If the camera you select below ends up running at a different
# resolution than calibration used, distances will be skewed. A mismatch
# warning is printed once the actual capture resolution is known.

# ── Camera Scan and Selection ─────────────────────────────────────
print("\nScanning for available cameras...")
available_cameras = find_available_cameras()

if not available_cameras:
    print("Error: No cameras detected. Please connect a camera and try again.")
    exit()

print("Available cameras:")
for i, (index, backend, width, height) in enumerate(available_cameras, 1):
    print(f"  [{i}] Camera index {index} — {width}x{height} — backend: {get_backend_name(backend)}")

if len(available_cameras) == 1:
    selected_index = 0
    print("\nOnly one camera detected — using it automatically.")
else:
    while True:
        selection = input(f"\nSelect camera [1-{len(available_cameras)}]: ").strip()
        if selection == "":
            selected_index = 0
            print("No selection — defaulting to camera 1.")
            break
        try:
            selected_index = int(selection) - 1
            if 0 <= selected_index < len(available_cameras):
                break
            print(f"Invalid selection. Please enter a number between 1 and {len(available_cameras)}.")
        except ValueError:
            print(f"Invalid input. Please enter a number between 1 and {len(available_cameras)}.")

camera_index, camera_backend, cam_width, cam_height = available_cameras[selected_index]
print(f"\nOpening camera {camera_index} ({cam_width}x{cam_height})...")

cap, opened_backend = open_selected_camera(camera_index, camera_backend)

if cap is None:
    print(f"Error: Could not open camera {camera_index}.")
    exit()

# Request a resolution, but don't assume the camera honours it — read back
# whatever it actually delivers and use THAT everywhere downstream
# (display, distance calc). This is what makes the script resolution-agnostic
# across different cameras rather than hardcoding 1280x720.
REQUESTED_WIDTH, REQUESTED_HEIGHT = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)

ACTUAL_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ACTUAL_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera opened successfully. Running at {ACTUAL_WIDTH}x{ACTUAL_HEIGHT}.")
if (ACTUAL_WIDTH, ACTUAL_HEIGHT) != (REQUESTED_WIDTH, REQUESTED_HEIGHT):
    print(f"Note: requested {REQUESTED_WIDTH}x{REQUESTED_HEIGHT} but camera returned "
          f"{ACTUAL_WIDTH}x{ACTUAL_HEIGHT}. Continuing with the actual resolution.")
if FOCAL_LENGTH_PX is not None:
    print("Note: distance estimates assume the capture resolution matches what "
          "calibrate_focal_length.py was run at. Re-calibrate if resolutions differ.")

print("Live detection running.")
print("Press Q to quit.")
print("Press S to save a screenshot.")

# ── Shared State ──────────────────────────────────────────────────
latest_frame = None
latest_results = None
frame_lock = threading.Lock()
results_lock = threading.Lock()
running = True
screenshot_count = 0

# ── TODO: Orientation feature hook ─────────────────────────────────
# Placeholder for the upcoming orientation-estimation feature. When ready,
# add an `estimate_orientation(box, frame)` import similarly to
# `estimate_distance` above, call it inside the detection loop below where
# distance is computed, and append its label to the per-box overlay text.

# ── Inference Thread ──────────────────────────────────────────────
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
                half=True
            )
            with results_lock:
                latest_results = results

# ── Start Inference Thread ────────────────────────────────────────
thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()

# ── Main Display Loop ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    with frame_lock:
        latest_frame = frame.copy()

    with results_lock:
        current_results = latest_results

    if current_results is not None:
        annotated_frame = current_results[0].plot()
        detections = len(current_results[0].boxes)
        color = (0, 255, 0) if detections > 0 else (0, 0, 255)
        cv2.putText(annotated_frame,
                    f'Ferries detected: {detections}',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

        if detections > 0:
            for i, box in enumerate(current_results[0].boxes):
                conf = float(box.conf[0])

                # Bounding box width in pixels (x2 - x1) — used for distance
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox_width_px = x2 - x1

                if FOCAL_LENGTH_PX is not None:
                    distance_m = estimate_distance(bbox_width_px, FOCAL_LENGTH_PX)
                    label = f'Conf: {conf:.2f} | Dist: {distance_m:.1f}m'
                else:
                    label = f'Conf: {conf:.2f} | Dist: N/A (not calibrated)'

                # TODO: append orientation label here once that feature lands
                # e.g. label += f' | Heading: {orientation_deg:.0f} deg'

                cv2.putText(annotated_frame,
                           label,
                           (20, 80 + (i * 40)),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (255, 255, 0), 2)
    else:
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame,
                    'Initialising...',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2)

    cv2.putText(annotated_frame,
                f'Camera: {camera_index} | OpenVINO INT8 | {ACTUAL_WIDTH}x{ACTUAL_HEIGHT} | '
                f'imgsz={MODEL_IMGSZ} | conf=0.6 | Every 5th frame',
                (20, annotated_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

    cv2.imshow('Maritime Ferry Detection - Live', annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        running = False
        break

    if key == ord('s'):
        screenshot_path = f'screenshot_{screenshot_count}.jpg'
        cv2.imwrite(screenshot_path, annotated_frame)
        print(f"Screenshot saved: {screenshot_path}")
        screenshot_count += 1

# ── Cleanup ───────────────────────────────────────────────────────
running = False
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")