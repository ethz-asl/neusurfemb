"""
Based on `instant-ngp/scripts/colmap2nerf.py` from
https://github.com/NVlabs/instant-ngp.
"""
import argparse
import cv2
import glob
import numpy as np
import os
import tqdm


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


parser = argparse.ArgumentParser()

parser.add_argument('--dataset-folder', type=str, required=True)
parser.add_argument('--min-valid-sharpness', type=int, required=True)

args = parser.parse_args()

dataset_folder = args.dataset_folder
min_valid_sharpness = args.min_valid_sharpness

all_paths_and_sharpnesses = []
all_sharpnesses = []
paths_to_remove = []

orig_undistorted_image_paths = sorted(
    glob.glob(os.path.join(dataset_folder, "rgb", "*.png")))

for image_path in tqdm.tqdm(orig_undistorted_image_paths):
    curr_sharpness = sharpness(image_path=image_path)
    all_paths_and_sharpnesses.append((image_path, curr_sharpness))
    all_sharpnesses.append(curr_sharpness)
    if (curr_sharpness < min_valid_sharpness):
        paths_to_remove.append(image_path)

all_sharpnesses = np.array(all_sharpnesses)

# Remove images that are too blurred.
for path_to_remove in paths_to_remove:
    os.remove(path_to_remove)

# Remove the corresponding raw images.
for image_path in sorted(
        glob.glob(os.path.join(dataset_folder, "raw_rgb", "*.png"))):
    if (not os.path.exists(
            os.path.join(dataset_folder, "rgb",
                         os.path.basename(image_path)))):
        os.remove(image_path)

# Rename the images so that all frame numbers are taken.
for subfolder in ["raw_rgb", "rgb"]:
    for image_idx, image_path in enumerate(
            sorted(glob.glob(os.path.join(dataset_folder, subfolder,
                                          "*.png")))):
        os.rename(
            image_path,
            os.path.join(dataset_folder, subfolder, f"_{image_idx:06d}.png"))
    for image_idx, image_path in enumerate(
            sorted(glob.glob(os.path.join(dataset_folder, subfolder,
                                          "*.png")))):
        os.rename(
            image_path,
            os.path.join(dataset_folder, subfolder, f"{image_idx:06d}.png"))

print(
    f"Kept {image_idx+1} out of {len(orig_undistorted_image_paths)} original "
    f"images, using a minimum-sharpness threshold of {min_valid_sharpness}.")
