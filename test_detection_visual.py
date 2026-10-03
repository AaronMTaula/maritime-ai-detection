"""
Interactive visual test script for maritime ferry detection
Displays detection results on images with bounding boxes
"""

import cv2
import glob
from pathlib import Path
from ultralytics import YOLO

# Load the model
print("Loading model...")
model = YOLO('runs/detect/train_v3_more_data/ferry_v3_map0.924_recall0.889.pt')
print("Model loaded successfully!\n")

# Get test images
image_dir = Path('dataset/train/images')
image_files = sorted(glob.glob(str(image_dir / '*.jpg')))

if not image_files:
    print("No images found in dataset/train/images/")
    exit(1)

print(f"Found {len(image_files)} images")
print("Controls: SPACE=next | S=save | Q=quit\n")

current_idx = 0

while True:
    if current_idx >= len(image_files):
        print("\nReached end of images")
        break

    img_path = image_files[current_idx]
    print(f"[{current_idx+1}/{len(image_files)}] Testing: {Path(img_path).name}")

    frame = cv2.imread(img_path)
    if frame is None:
        print("Failed to load image, skipping...\n")
        current_idx += 1
        continue

    # Run detection
    results = model(
        frame,
        conf=0.6,
        verbose=False,
        imgsz=640,
        half=False
    )

    # Draw annotations
    annotated_frame = results[0].plot()
    num_detections = len(results[0].boxes)

    # Add detection info
    cv2.putText(
        annotated_frame,
        f'Detections: {num_detections}',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if num_detections > 0 else (0, 0, 255),
        2
    )

    # Show image
    cv2.imshow('Ferry Detection Test', annotated_frame)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):  # Quit
        running = False
        break
    elif key == ord('s'):  # Save
        save_path = f'test_result_{current_idx}.jpg'
        cv2.imwrite(save_path, annotated_frame)
        print(f"Saved: {save_path}")
    elif key == ord(' '):  # Next (space)
        pass

    print(f"  → {num_detections} detection(s)")
    if num_detections > 0:
        for box in results[0].boxes:
            conf = float(box.conf[0])
            print(f"    Confidence: {conf:.2%}")

    current_idx += 1
    print()

cv2.destroyAllWindows()
print("Test completed!")
