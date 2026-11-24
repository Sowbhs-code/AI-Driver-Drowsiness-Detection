"""
extract_frames.py - turn your own webcam/phone videos into labelled images.
Usage (from project root):
  python training/extract_frames.py my_video.mp4 dataset/eval/glasses/sunglasses --every 15
  --every 15  => keep 1 frame out of every 15 (about 2 per second at 30 fps)
Record one video per class (eyes open, eyes closed, yawning, sunglasses on, sunglasses off ...),
then run this once per video into the matching folder.
"""
import argparse, os, cv2
ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("out_dir"); ap.add_argument("--every", type=int, default=15)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
cap = cv2.VideoCapture(a.video)
if not cap.isOpened(): raise SystemExit(f"Cannot open {a.video}")
tag = os.path.splitext(os.path.basename(a.video))[0]
i = saved = 0
while True:
    ok, frame = cap.read()
    if not ok: break
    if i % a.every == 0:
        cv2.imwrite(os.path.join(a.out_dir, f"{tag}_{i:05d}.jpg"), frame); saved += 1
    i += 1
print(f"Saved {saved} frames to {a.out_dir}")
