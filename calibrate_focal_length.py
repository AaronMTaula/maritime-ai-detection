"""
One-time camera calibration for monocular distance estimation.

HOW TO USE:
  1. Pick a reference object with a known width — e.g. a laptop (34cm),
     a piece of A4 paper held sideways (29.7cm), a car (approx. 1.8m), etc.
     A ferry photo you already have (with a known real distance) works too,
     if you know how far away it was when the photo was taken.
  2. Place the object at a known, measured distance from your MacBook camera
     (or use an existing photo where you know that distance).
  3. Run this script. It opens your camera (or loads an image if you pass
     --image), shows the frame, and asks you to click the LEFT edge then
     the RIGHT edge of the reference object.
  4. Enter the known distance and known width when prompted.
  5. The script computes focal_length_px and saves it to camera_config.json,
     which distance_estimation.py and live_detect.py will then use.

Formula used (inverse of the distance formula):
    focal_length_px = (pixel_width * known_distance_m) / known_width_m
"""

import cv2
import json
import os
import argparse

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "camera_config.json")

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 2:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)} recorded: ({x}, {y})")


def get_reference_frame(image_path=None):
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        return frame

    # Fall back to live webcam capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. On macOS, check camera "
                            "permissions in System Settings > Privacy & Security.")

    print("Position your reference object in view, then press SPACE to capture a frame.")
    frame = None
    while True:
        ret, live_frame = cap.read()
        if not ret:
            break
        cv2.imshow("Calibration - press SPACE to capture", live_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            frame = live_frame.copy()
            break
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyWindow("Calibration - press SPACE to capture")
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                         help="Optional path to an existing image instead of using the webcam")
    args = parser.parse_args()

    frame = get_reference_frame(args.image)
    if frame is None:
        print("No frame captured. Exiting.")
        return

    print("\nClick the LEFT edge of the reference object, then the RIGHT edge.")
    print("Press 'r' to reset points, 'q' when done.\n")

    window_name = "Click LEFT edge then RIGHT edge of reference object"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = frame.copy()
        for i, pt in enumerate(clicked_points):
            cv2.circle(display, pt, 6, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (pt[0] + 10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(clicked_points) == 2:
            cv2.line(display, clicked_points[0], clicked_points[1], (0, 255, 0), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            clicked_points.clear()
        if key == ord('q') and len(clicked_points) == 2:
            break

    cv2.destroyAllWindows()

    pixel_width = abs(clicked_points[1][0] - clicked_points[0][0])
    print(f"\nMeasured pixel width: {pixel_width}px")

    known_distance_m = float(input("Enter the KNOWN distance to the object (metres): "))
    known_width_m = float(input("Enter the KNOWN real-world width of the object (metres): "))

    focal_length_px = (pixel_width * known_distance_m) / known_width_m

    config = {
        "focal_length_px": focal_length_px,
        "calibration_pixel_width": pixel_width,
        "calibration_distance_m": known_distance_m,
        "calibration_object_width_m": known_width_m,
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nCalibration complete. Focal length: {focal_length_px:.2f}px")
    print(f"Saved to {CONFIG_PATH}")


if __name__ == "__main__":
    main()