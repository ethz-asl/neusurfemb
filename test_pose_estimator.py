import argparse
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

from third_party.surfemb.surfemb.pose_estimator import PoseEstimator
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--image-folder', type=str, required=True)
parser.add_argument('--pose-estimation-cfg-file', type=str, required=True)
parser.add_argument('--bbox-folder', type=str)
parser.add_argument('--yolo-checkpoint-path', type=str)

args = parser.parse_args()

image_folder = args.image_folder
pose_estimation_cfg_file = args.pose_estimation_cfg_file
bbox_folder = args.bbox_folder
yolo_checkpoint_path = args.yolo_checkpoint_path

assert ((bbox_folder is None) != (yolo_checkpoint_path is None)), (
    "Provide either the folder to the ground-truth object bounding boxes or "
    "the path to the pre-trained YOLO checkpoint.")

# Instantiate pose estimator.
pose_estimator = PoseEstimator.from_flags(
    flags_yaml_path=pose_estimation_cfg_file)
if (yolo_checkpoint_path is not None):
    # Instantiate YOLO.
    yolo = YOLO(yolo_checkpoint_path)

# Loop over images, estimating their pose.
for image_path in sorted(glob.glob(os.path.join(image_folder, "*.png"))):
    depth_image_path = os.path.join(
        os.path.dirname(os.path.dirname(image_folder)), "depth",
        os.path.basename(image_path))

    image = np.ascontiguousarray(
        cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[..., [2, 1, 0]])
    if (os.path.exists(depth_image_path)):
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
    else:
        depth_image = None
        warnings.warn(
            "Depth image(s) not found at "
            f"'{os.path.dirname(os.path.normpath(depth_image_path))}'.")

    if (yolo_checkpoint_path is not None):
        # [x_min, y_min, x_max, y_max].
        bbox = yolo(image)
        assert (len(bbox) == 1)
        bbox = bbox[0].boxes.xyxy.tolist()
        if (len(bbox) == 0):
            print("\033[93mCould not detect a bounding box for image "
                  f"'{image_path}'\033[0m.")
            continue
        else:
            assert (len(bbox) == 1)
            bbox = bbox[0]
    else:
        bbox_path = os.path.join(
            bbox_folder,
            os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        bbox = np.loadtxt(bbox_path).astype(int)

    C_T_W_meters, rendered_image = pose_estimator.estimate_pose(
        image=image, bbox=bbox, depth_image=depth_image)
    print(C_T_W_meters)
    plt.imshow(rendered_image)
    plt.show()
