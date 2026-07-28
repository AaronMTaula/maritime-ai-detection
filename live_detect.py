import cv2
import threading
from ultralytics import YOLO
from distance_estimation import estimate_distance, load_focal_length, KNOWN_WIDTH_M
 
# Load INT8 OpenVINO model
model = YOLO('runs/detect/train_v2_151img/weights/best_int8_openvino_model/')
 
# Load calibrated focal length for distance estimation.
# Run calibrate_focal_length.py first if camera_config.json doesn't exist yet.
try:
    FOCAL_LENGTH_PX = load_focal_length()
    print(f"Loaded focal length: {FOCAL_LENGTH_PX:.1f}px "
          f"(reference width: {KNOWN_WIDTH_M}m)")
except FileNotFoundError as e:
    FOCAL_LENGTH_PX = None
    print(f"WARNING: {e}")
    print("Distance estimation will be disabled until calibration is run.")
 
# Open webcam using DirectShow backend
cap = cv2.VideoCapture(0)
 
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
 
# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
print("Live detection running.")
print("Press Q to quit.")
print("Press S to save a screenshot.")
 
# Shared state between threads
latest_frame = None
latest_results = None
frame_lock = threading.Lock()
results_lock = threading.Lock()
running = True
screenshot_count = 0
 
# ── Inference thread ──────────────────────────────────────────────
def inference_thread():
    global latest_results, running
    frame_count = 0
 
    while running:
        # Safely grab the latest frame
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
            # Safely store results
            with results_lock:
                latest_results = results
 
# ── Start inference thread ────────────────────────────────────────
thread = threading.Thread(target=inference_thread, daemon=True)
thread.start()
 
# ── Main display loop ─────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
 
    # Safely update the shared frame for inference thread
    with frame_lock:
        latest_frame = frame.copy()
 
    # Safely read latest results for display
    with results_lock:
        current_results = latest_results
 
    # Draw results on frame
    if current_results is not None:
        annotated_frame = current_results[0].plot()
 
        # Count detections
        detections = len(current_results[0].boxes)
 
        # Display detection count
        color = (0, 255, 0) if detections > 0 else (0, 0, 255)
        cv2.putText(annotated_frame,
                    f'Ferries detected: {detections}',
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)
 
        # Display confidence scores and distance estimates
        if detections > 0:
            for i, box in enumerate(current_results[0].boxes):
                conf = float(box.conf[0])
 
                # Bounding box width in pixels (x2 - x1)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox_width_px = x2 - x1
 
                if FOCAL_LENGTH_PX is not None:
                    distance_m = estimate_distance(bbox_width_px, FOCAL_LENGTH_PX)
                    label = f'Conf: {conf:.2f} | Dist: {distance_m:.1f}m'
                else:
                    label = f'Conf: {conf:.2f} | Dist: N/A (not calibrated)'
 
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
 
    # Display settings info at bottom of frame
    cv2.putText(annotated_frame,
                'OpenVINO INT8 | Threaded | imgsz=320 | conf=0.6 | Every 5th frame',
                (20, annotated_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)
 
    cv2.imshow('Maritime Ferry Detection - Live', annotated_frame)
 
    key = cv2.waitKey(1) & 0xFF
 
    # Q to quit
    if key == ord('q'):
        running = False
        break
 
    # S to save screenshot
    if key == ord('s'):
        screenshot_path = f'screenshot_{screenshot_count}.jpg'
        cv2.imwrite(screenshot_path, annotated_frame)
        print(f"Screenshot saved: {screenshot_path}")
        screenshot_count += 1
 
# Cleanup
running = False
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")