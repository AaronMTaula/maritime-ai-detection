import contextlib
import io
import os
import cv2
import threading
from ultralytics import YOLO

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
        # Windows
        backends += [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    elif hasattr(cv2, "CAP_AVFOUNDATION"):
        # macOS
        backends += [cv2.CAP_AVFOUNDATION]
    else:
        # Linux
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
            break  # Found a working backend for this index — move to next

    return discovered


<<<<<<< HEAD
# Load INT8 OpenVINO model
model = YOLO('runs/detect/train_v2_151img/weights/best_int8_openvino_model/')
print("Model loaded successfully.")
=======
def open_selected_camera(index, preferred_backend=None):
    """Open a camera with fallback across backends."""
    backends = []
    if preferred_backend is not None:
        backends.append(preferred_backend)
    for b in get_platform_backends():
        if b not in backends:
            backends.append(b)
>>>>>>> main

    for backend in backends:
        result = try_open_camera(index, backend)
        if result is not None:
            cap, _ = result
            return cap, backend

    return None, None


# ── Load Model ────────────────────────────────────────────────────
print("Loading model...")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    model = YOLO('runs/detect/train_v2_151img/weights/best_int8_openvino_model/')
print("Model loaded successfully.")

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
    print(f"\nOnly one camera detected — using it automatically.")
else:
    while True:
        selection = input(f"\nSelect camera [1-{len(available_cameras)}]: ").strip()
        if selection == "":
            selected_index = 0
            print(f"No selection — defaulting to camera 1.")
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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
<<<<<<< HEAD
=======

>>>>>>> main
print("Camera opened successfully.")
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
                imgsz=320,
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
                cv2.putText(annotated_frame,
                           f'Conf: {conf:.2f}',
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
<<<<<<< HEAD
                f'Camera: {CAMERA_INDEX} | OpenVINO INT8 | imgsz=320 | conf=0.6 | Every 5th frame',
=======
                f'Camera: {camera_index} | OpenVINO INT8 | imgsz=320 | conf=0.6 | Every 5th frame',
>>>>>>> main
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