import cv2
import threading
from ultralytics import YOLO

# ── Camera Scan and Selection ─────────────────────────────────────
print("Scanning for available cameras...")
found = []

for index in range(10):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"  [{index}] Camera found — Resolution: {int(width)}x{int(height)}")
            found.append(index)
        cap.release()
        cv2.waitKey(100)
    else:
        print(f"  [{index}] No camera found")
        cap.release()

if not found:
    print("No cameras detected. Please connect a camera and try again.")
    exit()

print(f"\nAvailable camera indices: {found}")

if len(found) == 1:
    CAMERA_INDEX = found[0]
    print(f"Only one camera found — using index {CAMERA_INDEX} automatically.")
else:
    while True:
        selection = input(f"\nSelect camera index {found}: ").strip()
        if selection == "":
            CAMERA_INDEX = found[0]
            print(f"No selection made — defaulting to index {CAMERA_INDEX}.")
            break
        elif selection.isdigit() and int(selection) in found:
            CAMERA_INDEX = int(selection)
            break
        else:
            print(f"Invalid selection. Please enter one of {found}.")

print(f"\nUsing camera index {CAMERA_INDEX}")

# ── Load Model ────────────────────────────────────────────────────
print("Loading model...")
model = YOLO('runs/detect/train_v2_151img/weights/best_int8_openvino_model/')
print("Model loaded successfully.")

# ── Open Webcam ───────────────────────────────────────────────────
print(f"Opening camera index {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"Error: Could not open camera {CAMERA_INDEX}.")
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