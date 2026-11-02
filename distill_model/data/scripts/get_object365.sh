#!/bin/bash
# Objects365 dataset downloader and converter
# Download command: bash data/scripts/get_objects365.sh
# Train command: python train.py --data objects365.yaml
# Expected output directory:
#   /parent_folder
#     /Objects365
#     /yolov5

start=$(date +%s)

mkdir -p ../tmp
cd ../tmp/

echo "📦 Downloading Objects365 dataset..."

# These are the primary JSON + image patch URLs from Ultralytics
train_url="https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/"
val_url="https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/"

# Download annotations first
echo "Downloading train and val annotations..."
curl -L "${train_url}zhiyuan_objv2_train.tar.gz" -o zhiyuan_objv2_train.tar.gz &
curl -L "${val_url}zhiyuan_objv2_val.json" -o zhiyuan_objv2_val.json &
wait

# Download a smaller subset of patches for demonstration (to limit size)
# Full dataset requires all patch{i}.tar.gz for train (0–50) and val (0–43)
echo "Downloading subset of image patches (train: 0–2, val: 0–1)..."
for i in 0 1 2; do
  curl -L "${train_url}patch${i}.tar.gz" -o train_patch${i}.tar.gz &
done
for i in 0 1; do
  curl -L "${val_url}images/v1/patch${i}.tar.gz" -o val_patch${i}.tar.gz &
done
wait

echo "Unzipping archives..."
mkdir -p train val
for f in *.tar.gz; do
  echo "Extracting $f..."
  tar -xzf "$f" -C ./
done

echo "✅ All archives downloaded and extracted."
end=$(date +%s)
runtime=$((end - start))
echo "Download stage completed in $runtime seconds."

echo "📁 Converting annotations to YOLO format..."
python3 - <<'END'
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

def xyxy2xywhn(x, w, h, clip=False):
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # center x
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # center y
    y[:, 2] = (x[:, 2] - x[:, 0]) / w        # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h        # height
    if clip:
        y = np.clip(y, 0, 1)
    return y

root = Path('../Objects365')
(root / 'images' / 'train').mkdir(parents=True, exist_ok=True)
(root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
(root / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
(root / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

# Move some images for structure demonstration (customize to full dataset if desired)
# TRAIN IMAGES
os.system('find ../tmp -type f -name "*.jpg" -path "*/train/*" -exec mv {} ../Objects365/images/train/ \\;')

# VAL IMAGES — val patches extract into weird nested dirs like images/v1/patchX
os.system('find ../tmp -type f -name "*.jpg" -path "*/val/*"   -exec mv {} ../Objects365/images/val/ \\;')

# Fallback: if any leftover JPEGs aren't detected by train/val paths, grab them anyway
os.system('find ../tmp -type f -name "*.jpg" -exec mv {} ../Objects365/images/train/ \\;')

splits = {'train': 'zhiyuan_objv2_train.json', 'val': 'zhiyuan_objv2_val.json'}

for split, json_file in splits.items():
    json_path = Path(f'../tmp/{json_file}')
    if not json_path.exists():
        print(f"Warning: {json_file} not found, skipping {split}.")
        continue
    print(f"Parsing {json_file} for {split} split...")
    coco = COCO(json_path)
    names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
    labels_dir = root / 'labels' / split
    for cid, cat in enumerate(names):
        catIds = coco.getCatIds(catNms=[cat])
        imgIds = coco.getImgIds(catIds=catIds)
        for im in tqdm(coco.loadImgs(imgIds), desc=f'{split} class {cid + 1}/{len(names)} {cat}'):
            width, height = im["width"], im["height"]
            path = Path(im["file_name"])
            label_file = labels_dir / path.with_suffix('.txt').name
            annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=False)
            anns = coco.loadAnns(annIds)
            with open(label_file, 'a') as f:
                for a in anns:
                    x, y, w, h = a['bbox']
                    xyxy = np.array([[x, y, x + w, y + h]])
                    x, y, w, h = xyxy2xywhn(xyxy, w=width, h=height, clip=True)[0]
                    f.write(f"{cid} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")
END

cd ..
# rm -rf tmp
echo "tmp folder preserved at: $(realpath tmp)"
echo "✅ Objects365 dataset ready at: $(realpath Objects365)"
