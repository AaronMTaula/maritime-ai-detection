"""
Test script for maritime ferry detection model
Tests the model on images from the dataset without requiring a camera
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
image_files = sorted(glob.glob(str(image_dir / '*.jpg')))[:10]  # First 10 images

if not image_files:
    print("No images found in dataset/train/images/")
    exit(1)

print(f"Found {len(image_files)} images to test\n")

# Test on each image
total_detections = 0
images_with_detections = 0

for img_path in image_files:
    print(f"Testing: {Path(img_path).name}")

    frame = cv2.imread(img_path)
    if frame is None:
        print(f"  ❌ Failed to load image\n")
        continue

    # Run detection
    results = model(
        frame,
        conf=0.6,
        verbose=False,
        imgsz=640,
        half=False
    )

    num_detections = len(results[0].boxes)
    total_detections += num_detections

    if num_detections > 0:
        images_with_detections += 1
        print(f"  ✓ {num_detections} detection(s):")
        for i, box in enumerate(results[0].boxes):
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            print(f"    - Detection {i+1}: Confidence {conf:.2%}, Class ID: {class_id}")
    else:
        print(f"  ⚠ No detections")

    print()

# Summary
print("=" * 50)
print("TEST SUMMARY")
print("=" * 50)
print(f"Images tested: {len(image_files)}")
print(f"Images with detections: {images_with_detections}")
print(f"Total detections: {total_detections}")
print(f"Average detections per image: {total_detections / len(image_files):.2f}")
print("\n✓ Test completed successfully!")
