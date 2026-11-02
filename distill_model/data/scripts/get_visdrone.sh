#!/bin/bash
# VisDrone2019 DET dataset
# Download command: bash data/scripts/get_visdrone.sh
# Train command: python train.py --data visdrone.yaml
# Dataset location (default):
#   /parent_folder
#     /VisDrone
#     /yolov5

start=$(date +%s)

mkdir -p ../tmp
cd ../tmp/

echo "Downloading VisDrone2019 DET dataset..."
urls=(
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip"
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip"
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip"
)
for url in "${urls[@]}"; do
  fname=$(basename $url)
  echo "Downloading $fname ..."
  curl -L $url -o $fname && unzip -q $fname -d . && rm $fname &
done

wait
echo "All archives downloaded and extracted."

end=$(date +%s)
runtime=$((end - start))
echo "Completed in" $runtime "seconds."

echo "Converting annotations to YOLO format..."

python3 - <<'END'
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def convert_box(size, box):
    # Convert VisDrone [x, y, w, h] → YOLO [x_center, y_center, w, h]
    dw, dh = 1. / size[0], 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

def visdrone2yolo(dir):
    ann_dir = dir / 'annotations'
    img_dir = dir / 'images'
    label_dir = dir / 'labels'
    label_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(list(ann_dir.glob('*.txt')), desc=f'Converting {dir.name}'):
        img_path = img_dir / f.name.replace('.txt', '.jpg')
        if not img_path.exists():
            continue
        img_size = Image.open(img_path).size
        lines = []
        with open(f, 'r') as ann_file:
            for row in [x.split(',') for x in ann_file.read().strip().splitlines()]:
                if row[4] == '0':  # ignored region
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
        with open(label_dir / f.name, 'w') as out_file:
            out_file.writelines(lines)

root = Path('../tmp')
for d in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
    visdrone2yolo(root / d)

END

echo "Organizing into YOLO structure..."

mkdir -p ../VisDrone/images/train ../VisDrone/images/val
mkdir -p ../VisDrone/labels/train ../VisDrone/labels/val

# Move train/val images and labels into unified structure
cp -r VisDrone2019-DET-train/images/* ../VisDrone/images/train/
cp -r VisDrone2019-DET-val/images/* ../VisDrone/images/val/
cp -r VisDrone2019-DET-train/labels/* ../VisDrone/labels/train/
cp -r VisDrone2019-DET-val/labels/* ../VisDrone/labels/val/

# Clean up
cd ..
rm -rf tmp

echo "✅ VisDrone dataset ready at: $(realpath VisDrone)"
