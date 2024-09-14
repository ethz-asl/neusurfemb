import argparse
import cv2
import glob
import numpy as np
import os
import yaml

# Hack to disable WandB callback, which would otherwise cause issues with the
# project name.
os.environ['PYTEST_CURRENT_TEST'] = 'true'

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-folder', type=str, required=True)
parser.add_argument('--object-name', type=str, required=True)

args = parser.parse_args()

image_paths = sorted(
    glob.glob(os.path.join(args.dataset_folder, "rgb", "*.png")))

yolo_folder = os.path.join(args.dataset_folder, "yolo")
os.makedirs(yolo_folder)
os.symlink(os.path.join(args.dataset_folder, "rgb"),
           os.path.join(yolo_folder, "images"))
bbox_folder = os.path.join(yolo_folder, "labels")
os.makedirs(bbox_folder)

# Create labels.
for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_mask = image[..., 3] == 255.

    H, W = image.shape[:2]

    # Find the bounding box.
    y_mask, x_mask = np.where(image_mask)
    bbox = [x_mask.min(), y_mask.min(), x_mask.max(), y_mask.max()]

    x_center = (bbox[0] + bbox[2]) / 2.
    y_center = (bbox[1] + bbox[3]) / 2.
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    # Normalize.
    x_center /= W
    y_center /= H
    width /= W
    height /= H

    basename = os.path.basename(image_path).split(".png")[0]
    with open(os.path.join(bbox_folder, f"{basename}.txt"), 'w') as f:
        f.write(f"0 {x_center} {y_center} {width} {height}\n")

# Create YAML config file.
dataset_name = os.path.basename(os.path.normpath(args.dataset_folder))
dataset_config = {
    "path": args.dataset_folder,
    "train": "yolo",
    # We use the same images for training and validation. You may capture more
    # images to create a validation dataset, if you'd like. In any case, we
    # disable validation during training (cf. below).
    "val": "yolo",
    # Number of classes.
    "nc": 1,
    # Class names.
    "names": [args.object_name]
}
dataset_config_path = os.path.join(yolo_folder, f"{dataset_name}.yaml")
with open(dataset_config_path, "w") as f:
    yaml.dump(data=dataset_config, stream=f)

# Train YOLO.
yolo_model = YOLO("yolov8s.pt")
yolo_model.train(data=dataset_config_path,
                 epochs=150,
                 task="detect",
                 val=False,
                 project=yolo_folder,
                 name="training")
