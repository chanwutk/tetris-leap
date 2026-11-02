import os
import urllib.request
from pathlib import Path
from tqdm import tqdm
import tarfile
import numpy as np
from pycocotools.coco import COCO

# ---------- SETTINGS ----------
ROOT = Path("/data/alenachao/LEAP/distill_model/data/Objects365")
ROOT.mkdir(parents=True, exist_ok=True)

IMAGES_TRAIN = ROOT / "images/train"
IMAGES_VAL   = ROOT / "images/val"
LABELS_TRAIN = ROOT / "labels/train"
LABELS_VAL   = ROOT / "labels/val"

for p in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
    p.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86"

# ---------- UTILITY FUNCTIONS ----------
def download_file(url, dest):
    dest = Path(dest)
    if dest.exists():
        return
    print(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)

def extract_tar(tar_path, extract_path):
    tar_path = Path(tar_path)
    extract_path = Path(extract_path)
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_path)
    tar_path.unlink()  # remove tar to save space

def xyxy2xywhn(xyxy, w, h, clip=True):
    y = xyxy.copy()
    y[:,0] = ((xyxy[:,0] + xyxy[:,2])/2)/w
    y[:,1] = ((xyxy[:,1] + xyxy[:,3])/2)/h
    y[:,2] = (xyxy[:,2] - xyxy[:,0])/w
    y[:,3] = (xyxy[:,3] - xyxy[:,1])/h
    if clip:
        y = np.clip(y, 0, 1)
    return y

# ---------- 1️⃣ Download Annotations ----------
download_file(f"{BASE_URL}/train/zhiyuan_objv2_train.tar.gz", ROOT/"zhiyuan_objv2_train.tar.gz")
download_file(f"{BASE_URL}/val/zhiyuan_objv2_val.json", ROOT/"zhiyuan_objv2_val.json")

# Extract train annotations
with tarfile.open(ROOT/"zhiyuan_objv2_train.tar.gz") as tar:
    tar.extractall(path=ROOT)
(ROOT / "zhiyuan_objv2_train.tar.gz").unlink()

# ---------- 2️⃣ Download and Extract Train Patches ----------
for i in range(51):
    url = f"{BASE_URL}/train/patch{i}.tar.gz"
    dest = IMAGES_TRAIN / f"patch{i}.tar.gz"
    download_file(url, dest)
    extract_tar(dest, IMAGES_TRAIN)

# Flatten train images
for f in IMAGES_TRAIN.rglob("*.jpg"):
    f.rename(IMAGES_TRAIN / f.name)

# ---------- 3️⃣ Download and Extract Val Patches ----------
# v1 patches
for i in range(16):
    url = f"{BASE_URL}/val/images/v1/patch{i}.tar.gz"
    dest = IMAGES_VAL / f"patch_v1_{i}.tar.gz"
    download_file(url, dest)
    extract_tar(dest, IMAGES_VAL)

# v2 patches
for i in range(16, 44):
    url = f"{BASE_URL}/val/images/v2/patch{i}.tar.gz"
    dest = IMAGES_VAL / f"patch_v2_{i}.tar.gz"
    download_file(url, dest)
    extract_tar(dest, IMAGES_VAL)

# Flatten val images
for f in IMAGES_VAL.rglob("*.jpg"):
    f.rename(IMAGES_VAL / f.name)

# ---------- 4️⃣ Convert Annotations to YOLO Format ----------
for split, ann_file, labels_dir, images_dir in [
    ("train", ROOT/"zhiyuan_objv2_train.json", LABELS_TRAIN, IMAGES_TRAIN),
    ("val", ROOT/"zhiyuan_objv2_val.json", LABELS_VAL, IMAGES_VAL)
]:
    coco = COCO(ann_file)
    names = [x["name"] for x in coco.loadCats(coco.getCatIds())]
    labels_dir.mkdir(parents=True, exist_ok=True)
    for cid, cat in enumerate(names):
        catIds = coco.getCatIds(catNms=[cat])
        imgIds = coco.getImgIds(catIds=catIds)
        for im in tqdm(coco.loadImgs(imgIds), desc=f"{split} class {cid+1}/{len(names)} {cat}"):
            width, height = im["width"], im["height"]
            path = Path(im["file_name"])
            label_file = labels_dir / path.with_suffix(".txt").name
            annIds = coco.getAnnIds(imgIds=im["id"], catIds=catIds, iscrowd=False)
            with open(label_file, "a") as f:
                for a in coco.loadAnns(annIds):
                    x, y, w, h = a["bbox"]
                    xyxy = np.array([[x, y, x+w, y+h]])
                    x, y, w, h = xyxy2xywhn(xyxy, width, height)[0]
                    f.write(f"{cid} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n")

print("✅ Objects365 full dataset downloaded and converted to YOLO format!")
