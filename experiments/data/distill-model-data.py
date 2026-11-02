import scipy.io
import cv2
import os

# Load the annotations
mat = scipy.io.loadmat("/data/alenachao/LEAP/distill_model/data/stanfordcars/archive/car_devkit/devkit/cars_train_annos.mat")
annos = mat["annotations"][0]

labels_dir = "/data/alenachao/LEAP/distill_model/data/stanfordcars-final/labels/train"
os.makedirs(labels_dir, exist_ok=True)

images_dir = "/data/alenachao/LEAP/distill_model/data/stanfordcars-final/images/train"

for anno in annos:
    # print(anno)
    x1 = int(anno[0][0][0])
    y1 = int(anno[1][0][0])
    x2 = int(anno[2][0][0])
    y2 = int(anno[3][0][0])
    cls = 0  # convert to 0-index
    fname = anno[5][0]

    img_path = os.path.join(images_dir, fname)
    img = cv2.imread(img_path)

    if img is None:
        print("Missing image:", img_path)
        continue

    h, w = img.shape[:2]

    # Compute YOLO normalized bbox
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width    = (x2 - x1) / w
    height   = (y2 - y1) / h

    # Write out the .txt file
    label_path = os.path.join(labels_dir, fname.replace(".jpg", ".txt"))
    
    with open(label_path, "w") as f:
        f.write(f"{cls} {x_center} {y_center} {width} {height}\n")

print("Done! YOLO labels created in:", labels_dir)
