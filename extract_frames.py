import cv2
import os

# ── Config ────────────────────────────────────────────────────────
# Base folder containing dated subfolders of field collection media
base_folder = r"C:\Users\Aaron\OneDrive\Desktop\maritime_ai\images\NZ_AT\raw_video_clips\Photos and Videos"
output_folder = r"C:\Users\Aaron\OneDrive\Desktop\maritime_ai\images\NZ_AT\AT_Hop_Ferry"

# Frame interval — 30 = 1 frame per second on 30fps footage
# Maximises variety by avoiding near-identical consecutive frames
frame_interval = 30

os.makedirs(output_folder, exist_ok=True)

# ── Find all videos recursively across all dated subfolders ───────
video_files = []
for root, dirs, files in os.walk(base_folder):
    for f in files:
        if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            video_files.append(os.path.join(root, f))

print(f"Found {len(video_files)} video files across all subfolders")
print(f"Output folder: {output_folder}")
print(f"Frame interval: every {frame_interval}th frame\n")

total_saved = 0

# ── Process each video ────────────────────────────────────────────
for video_path in video_files:
    video_file = os.path.basename(video_path)
    subfolder = os.path.basename(os.path.dirname(video_path))
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Processing: [{subfolder}] {video_file}")
    print(f"FPS: {fps:.1f} | Frames: {total_frames} | Duration: {duration:.1f}s")

    count = 0
    saved = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            # Include subfolder date in filename to avoid collisions
            safe_subfolder = subfolder.replace(" ", "_")
            filename = f"{safe_subfolder}_{os.path.splitext(video_file)[0]}_frame{count:05d}.jpg"
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, frame)
            saved += 1
            total_saved += 1
        count += 1

    cap.release()
    print(f"Saved {saved} frames from {video_file}\n")

print(f"Done. Total frames saved: {total_saved}")
print(f"Output folder: {output_folder}")