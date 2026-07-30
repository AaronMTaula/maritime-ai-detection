import cv2
import threading
from ultralytics import YOLO


def get_backend_name(backend):
    if backend is None:
        return "default"
    backend_names = {
        cv2.CAP_ANY: "CAP_ANY",
        cv2.CAP_DSHOW: "CAP_DSHOW",
        cv2.CAP_MSMF: "CAP_MSMF",
        cv2.CAP_VFW: "CAP_VFW",
        cv2.CAP_AVFOUNDATION: "CAP_AVFOUNDATION",
    }
    return backend_names.get(backend, str(backend))


def find_available_cameras(max_devices=10):
    backends = [None]
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_AVFOUNDATION]:
        if hasattr(cv2, "CAP_DSHOW") and backend is not None:
            backends.append(backend)

    discovered = []
    seen = set()

    for backend in backends:
        for index in range(max_devices):
            if backend is None:
                cap = cv2.VideoCapture(index)
            else:
                cap = cv2.VideoCapture(index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            ret, _ = cap.read()
            cap.release()

            if ret:
                key = (index, backend)
                if key not in seen:
                    seen.add(key)
                    discovered.append((index, backend))

    return discovered


# Load INT8 OpenVINO model
model = YOLO('runs/detect/train_v2_151img/weights/best_int8_openvino_model/')
print("Model loaded successfully.")

# Discover and let the user choose a camera
available_cameras = find_available_cameras()

if not available_cameras:
    print("Error: No camera devices were detected.")
    exit()

print("Available cameras:")
for i, (index, backend) in enumerate(available_cameras, 1):
    print(f"  {i}. Camera index {index} using {get_backend_name(backend)}")

if len(available_cameras) == 1:
    selected_index = 0
else:
    selection = input("Select a camera number: ").strip()
    try:
        selected_index = int(selection) - 1
        if selected_index < 0 or selected_index >= len(available_cameras):
            raise ValueError
    except ValueError:
        print("Invalid selection. Using the first detected camera.")
        selected_index = 0

camera_index, camera_backend = available_cameras[selected_index]

if camera_backend is None:
    cap = cv2.VideoCapture(camera_index)
else:
    cap = cv2.VideoCapture(camera_index, camera_backend)

if not cap.isOpened():
    print("Error: Could not open the selected camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
                f'Camera: {CAMERA_INDEX} | OpenVINO INT8 | imgsz=320 | conf=0.6 | Every 5th frame',
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