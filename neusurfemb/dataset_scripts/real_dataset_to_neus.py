"""Converts .txt world-to-camera poses created by `pose_labeling.py` to
transforms.json files that can be used in `instant-ngp`/`NeuS2`.

A large part of this code is based on `instant-ngp/scripts/colmap2nerf.py` from
https://github.com/NVlabs/instant-ngp.
"""
import argparse
import cv2
import glob
import imageio.v3 as imageio
import json
import numpy as np
import os
import tqdm

from neusurfemb.misc_utils.invert_pose import invert_pose
from neusurfemb.misc_utils.transforms import (_W_NEUS_T_W_BOP,
                                              _W_NEUS_T_W_COLMAP,
                                              _W_NEUS_T_SCENE)

# Transformation from the world frame of COLMAP datasets to the scene frame used
# in the pre-processing.
scene_T_W_bop = np.linalg.inv(_W_NEUS_T_SCENE) @ _W_NEUS_T_W_BOP
# Transformation from the COLMAP frame coordinate to the BOP frame coordinate.
W_bop_T_W_colmap = np.linalg.inv(_W_NEUS_T_W_BOP) @ _W_NEUS_T_W_COLMAP


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def closest_point_2_lines(oa, da, ob, db):
    r"""Returns point closest to both rays of form o+t*d, and a weight factor
    that goes to 0 if the lines are parallel.
    """
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset-folder',
    type=str,
    required=True,
    help=("Path to the dataset folder. It is expected to contain a `rgb` "
          "subfolder with images, a `pose` subfolder with world-to-camera "
          "poses as .txt files, each corresponding to an image in `rgb`, and "
          "an `intrinsics.txt` file. A `transforms.json` file will be created "
          "in it."))
parser.add_argument(
    '--do-not-store-intermediate-transformations',
    action='store_true',
    help=("If False, the intermediate matrices/vectors `R` and `center_point` "
          "used to transform the input poses are not stored into the output "
          "JSON file."))

args = parser.parse_args()

_aabb_scale = 1
_dataset_folder = args.dataset_folder
_image_folder = os.path.join(_dataset_folder, "rgb")
_pose_folder = os.path.join(_dataset_folder, "pose")
_intrinsics_file_path = os.path.join(_dataset_folder, "intrinsics.txt")
_output_transform_file = os.path.join(_dataset_folder, "transforms.json")

# List of supported image extensions.
_image_extensions = ["png", "jpg", "jpeg"]

if (not os.path.exists(_image_folder)):
    raise (OSError(f"The image folder '{_image_folder}' could not be found."))
if (not os.path.exists(_pose_folder)):
    raise (OSError(f"The pose folder '{_pose_folder}' could not be found."))
if (not os.path.exists(_intrinsics_file_path)):
    raise (OSError(f"The intrinsics file '{_intrinsics_file_path}' could not "
                   "be found."))
if (os.path.exists(_output_transform_file)):
    raise (OSError(
        f"The output transform file '{_output_transform_file}' "
        "already exists. Please remove it or rename to avoid overriding it."))

# Find the actual extension of the input images and verify that there is exactly
# one pose for each image.
curr_image_extension_idx = 0
image_list = []
while len(image_list) == 0 and curr_image_extension_idx < len(
        _image_extensions):
    image_extension = _image_extensions[curr_image_extension_idx]
    image_list = sorted(
        glob.glob(os.path.join(_image_folder, f"*.{image_extension}")))
    curr_image_extension_idx += 1
assert (len(image_list) > 0), f"Found no images in '{_image_folder}'."
pose_list = sorted(glob.glob(os.path.join(_pose_folder, "*.txt")))
assert (
    [os.path.basename(f).split(f'.{image_extension}')[0] for f in image_list
    ] == [os.path.basename(f).split('.txt')[0] for f in pose_list]
), f"Found non-matching images-poses in '{_image_folder}' and '{_pose_folder}'."

has_depth = os.path.exists(os.path.join(os.path.dirname(_image_folder),
                                        "depth"))

# Read an example image to find the image dimensions.
example_image = cv2.imread(image_list[0])
H, W = example_image.shape[:2]

# Read the camera intrinsics. NOTE: A pinhole camera is assumed.
K = np.loadtxt(_intrinsics_file_path)
fl_x = K[0, 0]
fl_y = K[1, 1]
c_x = K[0, 2]
c_y = K[1, 2]

# angle_x = math.atan(W / (fl_x * 2)) * 2
# angle_y = math.atan(H / (fl_y * 2)) * 2

angle_x = np.arctan2(W / 2, fl_x) * 2
angle_y = np.arctan2(H / 2, fl_y) * 2

# Bottom and up vectors.
bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
up = np.zeros(3)
out = {
    "camera_angle_x": angle_x,
    "camera_angle_y": angle_y,
    "fl_x": fl_x,
    "fl_y": fl_y,
    "k1": 0.0,
    "k2": 0.0,
    "p1": 0.0,
    "p2": 0.0,
    "cx": c_x,
    "cy": c_y,
    "w": W,
    "h": H,
    "aabb_scale": _aabb_scale,
    "frames": [],
}

print(
    f"\033[94mCreating output transform file '{_output_transform_file}'.\033[0m"
)

for image_idx, (image_file_path, pose_file_path) in enumerate(
        tqdm.tqdm(zip(image_list, pose_list), total=len(image_list))):
    image_rel = os.path.relpath(_image_folder)
    relative_image_file_path = f"./rgb/{os.path.basename(image_file_path)}"
    if (has_depth):
        relative_depth_file_path = ("./depth/"
                                    f"{os.path.basename(image_file_path)}")
        relative_masked_mm_depth_file_path = (
            "./depth_masked_mm/"
            f"{os.path.basename(image_file_path)}")
        if (image_idx == 0):
            os.makedirs(
                os.path.join(
                    os.path.dirname(image_file_path), "..",
                    os.path.dirname(relative_masked_mm_depth_file_path)))

    masked_rgb = imageio.imread(image_file_path)
    if (has_depth):
        depth = imageio.imread(
            os.path.join(os.path.dirname(image_file_path), "..",
                         relative_depth_file_path))
        if (masked_rgb.shape[-1] == 4):
            # Save a version of the depth image that only has non-zero depth on
            # the object.
            mask_image = masked_rgb[..., 3] == 255
        else:
            mask_image = np.ones_like(masked_rgb[..., 0])
        assert (masked_rgb.shape[:2] == depth.shape[:2])
        # - Set depth to zero for the out-of-object pixels.
        depth_masked_mm = depth.copy()
        y_out_mask, x_out_mask = np.where(np.logical_not(mask_image))
        depth_masked_mm[y_out_mask, x_out_mask] = 0
        cv2.imwrite(relative_masked_mm_depth_file_path, depth_masked_mm)

    sharpness_value = sharpness(image_file_path)
    # Read world-to-camera pose.
    C_T_W_colmap = np.loadtxt(pose_file_path).reshape(4, 4)
    W_colmap_T_C = invert_pose(C_T_W_colmap)
    # Apply transformations required by the NeRF convention.
    # - Post multiplication for camera, to convert from OpenCV format to OpenGL
    #   format.
    W_colmap_T_C[0:3, 2] *= -1
    W_colmap_T_C[0:3, 1] *= -1
    # - Convert the coordinate system from COLMAP to BOP.
    W_bop_T_C = W_bop_T_W_colmap @ W_colmap_T_C
    # - Pre-multiplication for world:
    scene_T_C = scene_T_W_bop @ W_bop_T_C

    # Update the up vector using the original z axis.
    up += scene_T_C[0:3, 1]

    frame = {
        "file_path": relative_image_file_path,
        "sharpness": sharpness_value,
        "transform_matrix": scene_T_C
    }
    if (has_depth):
        frame["depth_path"] = relative_masked_mm_depth_file_path
    out["frames"].append(frame)
num_frames = len(out["frames"])
up = up / np.linalg.norm(up)
print(f"Found up vector {up}")

# Rotate up vector to [0, 0, 1].
R = rotmat(up, [0, 0, 1])
R = np.pad(R, [0, 1])
R[-1, -1] = 1

# Rotate the transforms so that the up vector is the z axis.
for f in out["frames"]:
    f["transform_matrix"] = np.matmul(R, f["transform_matrix"])

# Find a central point all cameras are looking at.
print("Computing center of attention...")
total_weight = 0.0
center_point = np.array([0.0, 0.0, 0.0])
for f in out["frames"]:
    mf = f["transform_matrix"][0:3, :]
    for g in out["frames"]:
        mg = g["transform_matrix"][0:3, :]
        p, W = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
        if W > 0.01:
            center_point += p * W
            total_weight += W
center_point /= total_weight
# Translate the cameras so that the world origin coincides with the central
# point computed above.
for f in out["frames"]:
    f["transform_matrix"][0:3, 3] -= center_point

# Scale the world coordinate frame (i.e., scale the translation part of the
# camera-to-world transforms) so that the scene fits within a "standard NeRF"
# size.
# NOTE: It is assumed that the poses were computed using `pose_labeling.py`, and
# hence that the depth was in [mm] and that the poses read from .txt files are
# in [m]!
#
# In practice:
# - Denoting as UOM the unit of measure of the training coordinates
#   resulting from the above scaling, the equivalent in meters of 1 UOM is
#   given by the value of one_uom_scene_to_m.
avg_len = 0.
for f in out["frames"]:
    avg_len += np.linalg.norm(f["transform_matrix"][0:3, 3])
avg_len /= num_frames

scale = 1.0 / avg_len
one_uom_scene_to_m = 1.0 / scale
print(f"\033[94mAverage camera distance from origin = {avg_len} m (NOTE: "
      "Assuming the input UOM of the transforms was meters, which is the case "
      "when using `pose_labeling.py` to extract the poses, if depth was "
      "available).\033[0m")

# Write the transforms to file.
for f in out["frames"]:
    f["transform_matrix"] = f["transform_matrix"].tolist()

out["scale"] = scale
out["one_uom_scene_to_m"] = one_uom_scene_to_m

if (has_depth):
    out['enable_depth_loading'] = True
    out['integer_depth_scale'] = out['one_uom_scene_to_m'] / 1000.

if (not args.do_not_store_intermediate_transformations):
    out["R_for_transformations"] = R.tolist()
    out["center_point_for_transformations"] = center_point.tolist()

with open(_output_transform_file, "w") as outfile:
    json.dump(out, outfile, indent=4)
