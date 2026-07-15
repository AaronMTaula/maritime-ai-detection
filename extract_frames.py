import cv2
import os

# Config
input_folder = r"C:\Users\Aaron\OneDrive\Desktop\maritime_ai\images\NZ_AT\raw_video_clips"
output_folder = r"C:\Users\Aaron\OneDrive\Desktop\maritime_ai\images\NZ_AT\AT_Hop_Ferry"
frame_interval = 15  # extract every 15th frame

os.makedirs(output_folder, exist_ok=True)

video_files = [f for f in os.listdir(input_folder) 
               if f.endswith(('.mp4', '.mov', '.avi', '.mkv'))]

total_saved = 0

for video_file in video_files:
    video_path = os.path.join(input_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nProcessing: {video_file}")
    print(f"FPS: {fps:.1f} | Frames: {total_frames} | Duration: {duration:.1f}s")
    
    count = 0
    saved = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = f"{os.path.splitext(video_file)[0]}_frame{count:05d}.jpg"
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, frame)
            saved += 1
            total_saved += 1
        count += 1
    
    cap.release()
    print(f"Saved {saved} frames from {video_file}")

print(f"\nDone. Total frames saved: {total_saved}")
print(f"Output folder: {output_folder}")