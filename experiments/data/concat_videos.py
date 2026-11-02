import cv2
from pathlib import Path

video_dir = Path("/data/alenachao/LEAP/data/videos/jnc7/train")
videos = sorted(video_dir.glob("*.mp4"))

# Get first video's properties
cap = cv2.VideoCapture(str(videos[0]))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/data/alenachao/LEAP/data/videos/jnc7/concat/train/jnc7_train.mp4', fourcc, fps, (width, height))

for v in videos:
    cap = cv2.VideoCapture(str(v))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()

out.release()
