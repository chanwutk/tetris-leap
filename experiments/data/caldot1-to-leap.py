"""
editing otif/data/raw_to_dataset/caldot_one_minute.py

video_dir is of format:
|-- 1.mp4
|-- 2.mp4
...

and tracks_dir is of format:
|-- 1.json
|-- 2.json

where each video has corresponding file with tracks for it.

we want to extract 60 1-min segments from the dataset, we need to know what video the segment came from.
so outputted video should be titled 0.mp4 (0th segment, came from video 1)

we also need to get tracks corresponding to that segment from correct json file and write to a txt file.
specifically it will be called label_caldot1_train.txt

OKAY for now we will just use 1 video, since if we use multiple we risk track IDs overlapping.
so revised:
- cut video into 60 sec clips
- for each 60 sec 


"""
import os
import random
import subprocess
import sys

video_dir = "/data/alenachao/data/otif-dataset/dataset/caldot1/full_dataset/videos"
tracks_dir = "/data/alenachao/data/otif-dataset/dataset/caldot1/full_dataset/tracks"
out_dir = ""

# out_paths = [
#     os.path.join(out_root, 'train/video/'),
#     os.path.join(out_root, 'valid/video/'),
#     os.path.join(out_root, 'test/video/'),
#     os.path.join(out_root, 'tracker/video/'),
# ]

def get_duration(fname):
    output = subprocess.check_output(['ffprobe', '-select_streams', 'v:0', '-show_entries', 'stream=duration', '-of', 'csv=s=,:p=0', fname])
    return float(output.strip())

# list of tuples (fname, seconds)
segments = []

for fname in os.listdir(video_dir):
    # if not fname.startswith('{}-'.format(ds_id)):
    #     continue
    duration = int(get_duration(video_dir+fname))
    for skip in range(0, duration-60, 60):
        segments.append((fname, skip))

random.shuffle(segments)
print('got {} segments'.format(len(segments)))

# for out_path in out_paths:
cur_segments = segments[0:60]
segments = segments[60:]
for i, (fname, skip) in enumerate(cur_segments):
    ffmpeg_args = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-ss', str(skip), '-i', video_dir+fname, '-t', '60', out_dir+str(i)+'.mp4']
    subprocess.call(ffmpeg_args)

