import cv2
import numpy as np
import os
import random
from pathlib import Path
import argparse

object_type = {2:'car',5:'bus',7:'truck'}
mask_base = "./masks"

def get_labels(label_file):
    if not os.path.exists(label_file):
        print(f"Warning: Label file not found: {label_file}")
        return {}, {}
        
    with open(label_file, "r") as f:
        contents = f.readlines()

    parsed_result = {}
    parsed_tuple = {}

    for line in contents:
        content = line.strip().split(",")
        frame_id = int(content[0])
        car_id = int(content[1])
        x_min = max(0, int(float(content[2])))
        y_min = max(0, int(float(content[3])))
        x_max = x_min + int(float(content[4]))
        y_max = y_min + int(float(content[5]))
        conf = float(content[6])
        if int(float(content[7])) not in object_type:
            continue
        object_class = object_type[int(float(content[7]))]

        parsed_result.setdefault(frame_id, []).append(
            [car_id, x_min, y_min, x_max, y_max, conf, object_class]
        )

        if car_id not in parsed_tuple:
            parsed_tuple[car_id] = [frame_id, frame_id]
        else:
            parsed_tuple[car_id][1] = frame_id

    return parsed_result, parsed_tuple


def generate_train_dataset_single_clip(
    video_path,
    label_file,
    cam_id,
    car_id_offset,
    save_dir,
    skip_frame=5
):
    parsed_result, parsed_tuple = get_labels(label_file)
    if not parsed_result:
        return

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    os.makedirs(save_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % skip_frame == 0 and frame_id in parsed_result:
            for car_id, x1, y1, x2, y2, conf, cls in parsed_result[frame_id]:
                duration = parsed_tuple[car_id][1] - parsed_tuple[car_id][0]
                if 15 < duration < 900:
                    crop = frame[y1:y2, x1:x2]
                    # Apply offset to ensure global uniqueness
                    global_car_id = car_id + car_id_offset
                    fname = f"{cam_id}_{global_car_id}_{cls}_{x1}_{y1}_{x2}_{y2}_{frame_id}.jpg"
                    cv2.imwrite(os.path.join(save_dir, fname), crop)

        frame_id += 1

    cap.release()


def generate_test_query_dataset_single_clip(
    video_path,
    label_file,
    cam_id,
    car_id_offset,
    save_test_dir,
    save_query_dir,
    skip_frame=5
):
    parsed_result, parsed_tuple = get_labels(label_file)
    if not parsed_result:
        return

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    tracks = {}  # {car_id: [(img, cls, x1, y1, x2, y2, frame_id), ...]}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % skip_frame == 0 and frame_id in parsed_result:
            for car_id, x1, y1, x2, y2, conf, cls in parsed_result[frame_id]:
                # In train we checked duration, assume we do same for test?
                # The reference code checked: if duration_time < 900 and duration_time > 15
                if car_id in parsed_tuple:
                    duration = parsed_tuple[car_id][1] - parsed_tuple[car_id][0]
                    if 15 < duration < 900:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            tracks.setdefault(car_id, []).append(
                                (crop, cls, x1, y1, x2, y2, frame_id)
                            )

        frame_id += 1

    cap.release()

    for car_id, imgs in tracks.items():
        if not imgs:
            continue
            
        global_car_id = car_id + car_id_offset
        
        # Random query
        q_idx = random.randint(0, len(imgs) - 1)
        
        for i, (crop, cls, x1, y1, x2, y2, fid) in enumerate(imgs):
            fname = f"{cam_id}_{global_car_id}_{cls}_{x1}_{y1}_{x2}_{y2}_{fid}.jpg"
            if i == q_idx:
                cv2.imwrite(os.path.join(save_query_dir, fname), crop)
            else:
                cv2.imwrite(os.path.join(save_test_dir, fname), crop)


def process_video_directory(video_name, split, video_base, label_base, save_base):
    # video_base and label_base are already dataset-specific paths (e.g. .../caldot1-y5)
    # so we don't append video_name to them — only split goes below
    video_dir = Path(video_base) / split

    # Define output directories
    if split == "train":
        save_dir = Path(save_base) / video_name / "image_train"
        os.makedirs(save_dir, exist_ok=True)
    else:
        # For test, we need image_test (gallery) and image_query (query)
        save_dir_test = Path(save_base) / video_name / "image_test"
        save_dir_query = Path(save_base) / video_name / "image_query"
        os.makedirs(save_dir_test, exist_ok=True)
        os.makedirs(save_dir_query, exist_ok=True)

    if not video_dir.exists():
        print(f"Video directory does not exist: {video_dir}")
        return

    # clips are .mp4 files
    clips = sorted(video_dir.glob("*.mp4"), key=lambda x: x.stem)

    for clip_path in clips:
        try:
            clip_id = int(clip_path.stem)
        except ValueError:
            print(f"Skipping non-integer clip name: {clip_path.name}")
            continue

        print(f"Processing clip {clip_id} from {video_name} ({split})...")

        # Determine label file path (label_base is already dataset-specific)
        label_path_1 = Path(label_base) / str(clip_id) / split / f"label_{clip_id}.txt"
        label_path_2 = Path(label_base) / str(clip_id) / f"label_{clip_id}.txt"

        if label_path_1.exists():
            label_file = str(label_path_1)
        elif label_path_2.exists():
            label_file = str(label_path_2)
        else:
            print(f"Label file not found for clip {clip_id} (checked {label_path_1} and {label_path_2})")
            continue

        # cam_id is unique per clip
        cam_id = clip_id

        # car_id uniqueness: cam_id * 100000 + local_id
        car_id_offset = cam_id * 100000

        if split == "train":
            generate_train_dataset_single_clip(
                video_path=str(clip_path),
                label_file=label_file,
                cam_id=cam_id,
                car_id_offset=car_id_offset,
                save_dir=str(save_dir),
                skip_frame=5
            )
        else:
            generate_test_query_dataset_single_clip(
                video_path=str(clip_path),
                label_file=label_file,
                cam_id=cam_id,
                car_id_offset=car_id_offset,
                save_test_dir=str(save_dir_test),
                save_query_dir=str(save_dir_query),
                skip_frame=5
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_name", type=str, help="Name of the video directory (e.g., caldot1, caldot2)")
    parser.add_argument("--video_base", type=str, default="/data/alenachao/LEAP/data/videos",
                        help="Root directory containing video subdirectories")
    parser.add_argument("--label_base", type=str, default="/data/alenachao/LEAP/data/labels",
                        help="Root directory containing label subdirectories")
    parser.add_argument("--save_base", type=str, default="/data/alenachao/LEAP/datasets",
                        help="Root directory where reid dataset images will be saved")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "test"],
                        help="Which splits to process (default: train test)")
    args = parser.parse_args()

    for split in args.splits:
        process_video_directory(
            args.video_name,
            split=split,
            video_base=args.video_base,
            label_base=args.label_base,
            save_base=args.save_base,
        )