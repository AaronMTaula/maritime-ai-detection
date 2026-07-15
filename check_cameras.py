import cv2

print("Scanning for available cameras...")
found = []

for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera found at index {index}")
            found.append(index)
        cap.release()
    else:
        print(f"No camera at index {index}")

if not found:
    print("\nNo cameras detected by OpenCV.")
else:
    print(f"\nAvailable camera indices: {found}")