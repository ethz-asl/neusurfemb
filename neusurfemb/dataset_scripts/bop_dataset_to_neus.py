import argparse
import copy
import cv2
import glob
import os
import json
import numpy as np
import tqdm

from third_party.bop_toolkit.bop_toolkit_lib import inout

from neusurfemb.misc_utils.invert_pose import invert_pose

parser = argparse.ArgumentParser()

parser.add_argument(
    '--scene-path',
    type=str,
    help=("Path to the subfolder of the dataset containing the scene to "
          "process."),
    required=True)
parser.add_argument(
    '--obj-id-to-process',
    type=int,
    choices=[*range(1, 16)],
    help=("If specified, only data for the object with specified ID will be "
          "converted."))
parser.add_argument(
    '--min-num-pixels-per-object',
    type=int,
    default=500,
    help=("For each object, a frame is kept only if it contains at least this "
          "many pixels in which the object is visible."))

args = parser.parse_args()

scene_path = args.scene_path
print(f"Processing folder: '{scene_path}'.")

split = "train"
K = {"train": {}}
H = {"train": {}}
W = {"train": {}}
depth_scale = {"train": None}

transforms = {"train": {}}
scene_T_Cs = {"train": {}}
avg_len = {"train": 0.}
scale = {"train": None}
one_uom_scene_to_m = {"train": None}
num_frames = {"train": 0}

# With respect to the scene frame used in the pre-processing, the object-centric
# world frame of BOP datasets is rotated 90 degrees counter clockwise around the
# z axis.
_scene_T_W_bop = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                           [0, 0, 0, 1]])

masked_rgb_folder = {}

_file_extension = None
rgb_image_paths = None
rgb_image_paths_png = sorted(glob.glob(os.path.join(scene_path, "rgb",
                                                    "*.png")))
rgb_image_paths_jpg = sorted(glob.glob(os.path.join(scene_path, "rgb",
                                                    "*.jpg")))
if (len(rgb_image_paths_png) > 0):
    assert (len(rgb_image_paths_jpg) == 0)
    _file_extension = "png"
    rgb_image_paths = rgb_image_paths_png
else:
    assert (len(rgb_image_paths_jpg) > 0)
    _file_extension = "jpg"
    rgb_image_paths = rgb_image_paths_jpg

# Load camera poses.
scene_camera = inout.load_scene_camera(
    os.path.join(scene_path, "scene_camera.json"))
scene_gt = inout.load_scene_gt(os.path.join(scene_path, "scene_gt.json"))

# Check which objects are contained in the scene and which ones should be
# processed.
id_all_objects_in_the_scene = set()
for scene_gt_curr_im in scene_gt.values():
    for obj_curr_im in scene_gt_curr_im:
        id_all_objects_in_the_scene.add(obj_curr_im['obj_id'])
id_all_objects_in_the_scene = list(id_all_objects_in_the_scene)

objects_to_process = None
if (args.obj_id_to_process is not None):
    assert (args.obj_id_to_process in id_all_objects_in_the_scene)
    objects_to_process = [args.obj_id_to_process]
else:
    objects_to_process = id_all_objects_in_the_scene

for dictionary in [transforms, scene_T_Cs]:
    for dataset_split in dictionary:
        dictionary[dataset_split] = {
            id_curr_obj: {} for id_curr_obj in objects_to_process
        }
for dataset_split in avg_len:
    avg_len[dataset_split] = {
        id_curr_obj: 0.0 for id_curr_obj in objects_to_process
    }
for dictionary in [scale, one_uom_scene_to_m]:
    for dataset_split in dictionary:
        dictionary[dataset_split] = {
            id_curr_obj: None for id_curr_obj in objects_to_process
        }
for dataset_split in num_frames:
    num_frames[dataset_split] = {
        id_curr_obj: 0 for id_curr_obj in objects_to_process
    }
for id_curr_obj in objects_to_process:
    masked_rgb_folder[id_curr_obj] = os.path.join(
        scene_path, f"rgb_masked_obj_{id_curr_obj}")
    os.makedirs(masked_rgb_folder[id_curr_obj])

# - Load the transform (from the global/model coordinate frame to camera).
for rgb_image_path in tqdm.tqdm(rgb_image_paths):
    image_idx = int(
        os.path.basename(rgb_image_path).split(f".{_file_extension}")[0])
    # Iterate over the different objects in the image.
    for obj_curr_im in scene_gt[image_idx]:
        R = obj_curr_im['cam_R_m2c']
        t = obj_curr_im['cam_t_m2c'][..., 0]
        C_T_W_bop = np.eye(4)
        C_T_W_bop[:3, :3] = R
        C_T_W_bop[:3, 3] = t
        W_bop_T_C = invert_pose(C_T_W_bop)

        # - Apply transformation to use the camera convention required by NeuS
        #   (cf. `real_dataset_to_neus.py`).
        # - Post multiplication for camera, to convert from OpenCV format to
        #   OpenGL format.
        W_bop_T_C[0:3, 2] *= -1  # Flip the y and z axis.
        W_bop_T_C[0:3, 1] *= -1
        # - Pre-multiplication for world.
        scene_T_C = _scene_T_W_bop @ W_bop_T_C

        # NOTE: Since it is assumed that the poses were in [mm], here they get
        # converted to meters, so that `one_uom_scene_to_m` can be the
        # inverse of the scale of the scene, in accordance with datasets
        # produced by `pose_labeling.py` (cf. below).
        scene_T_C[:3, 3] = scene_T_C[:3, 3] * 1.e-3

        obj_id = obj_curr_im['obj_id']
        if (obj_id in objects_to_process):
            scene_T_Cs[split][obj_id][image_idx] = scene_T_C

            avg_len[split][obj_id] = avg_len[split][obj_id] + np.linalg.norm(
                scene_T_C[0:3, 3])
            num_frames[split][obj_id] += 1

initialized_common_metadata = {}
for id_curr_obj in objects_to_process:
    print(f"- Object no. {id_curr_obj}:")
    avg_len[split][id_curr_obj] = avg_len[split][id_curr_obj] / num_frames[
        split][id_curr_obj]
    print("\t- Average distance of the cameras from the object center in the "
          f"'{split}' set = {avg_len[split][id_curr_obj]} m.")
    print(f"\t- Number of frames in the '{split}' split = "
          f"{num_frames[split][id_curr_obj]}.")
    # - `scale` is a value that gets multiplied to the translation part of the
    #   poses when training, and scales the scene to a "standard NeRF size".
    # - Denoting as UOM the unit of measure of the training coordinates
    #   resulting from the above scaling, the equivalent in meters of 1 UOM is
    #   given by the value of one_uom_scene_to_m.
    scale[split][id_curr_obj] = 1. / avg_len[split][id_curr_obj]
    one_uom_scene_to_m[split][id_curr_obj] = 1. / scale[split][id_curr_obj]

    initialized_common_metadata[id_curr_obj] = False

# Iterate over the images.
for rgb_image_path in tqdm.tqdm(rgb_image_paths):
    image_idx = int(
        os.path.basename(rgb_image_path).split(f".{_file_extension}")[0])

    # - Obtain intrinsics.
    K[split][image_idx] = scene_camera[image_idx]['cam_K'].reshape(3, 3)

    # - Read RGB image.
    rgb_image = cv2.imread(rgb_image_path, flags=cv2.IMREAD_UNCHANGED)
    assert (rgb_image.ndim == 3)
    H[split][image_idx], W[split][image_idx] = (rgb_image.shape[:2])
    # Initialize the transform dictionaries.
    image_dict = {}
    image_dict["fl_x"] = K[split][image_idx][0, 0]
    image_dict["fl_y"] = K[split][image_idx][1, 1]
    image_dict["cx"] = K[split][image_idx][0, 2]
    image_dict["cy"] = K[split][image_idx][1, 2]
    image_dict["w"] = W[split][image_idx]
    image_dict["h"] = H[split][image_idx]
    image_dict["camera_angle_x"] = np.arctan2(W[split][image_idx] / 2,
                                              K[split][image_idx][0, 0]) * 2
    image_dict["camera_angle_y"] = np.arctan2(H[split][image_idx] / 2,
                                              K[split][image_idx][1, 1]) * 2

    for obj_local_idx, obj_curr_im in enumerate(scene_gt[image_idx]):
        obj_id = obj_curr_im['obj_id']
        if (obj_id in objects_to_process):
            if (not initialized_common_metadata[obj_id]):
                transforms[split][obj_id]["aabb_scale"] = 1
                transforms[split][obj_id]["scale"] = scale[split][obj_id]
                transforms[split][obj_id][
                    "one_uom_scene_to_m"] = one_uom_scene_to_m[split][obj_id]
                transforms[split][obj_id]["frames"] = []
                initialized_common_metadata[obj_id] = True

            # - Read mask image and add alpha channel based on that.
            mask_path = os.path.join(
                scene_path, "mask_visib",
                f"{image_idx:06d}_{obj_local_idx:06d}.png")
            mask_image = cv2.imread(mask_path, flags=cv2.IMREAD_UNCHANGED)
            assert (mask_image.ndim == 2)
            # If the image does not have at least a certain amount of pixels in
            # which the object is visible, discard it (for the current object).
            if (np.count_nonzero(mask_image) < args.min_num_pixels_per_object):
                continue
            else:
                assert (np.unique(mask_image).tolist() == [0, 255])
            masked_rgb_image = np.concatenate(
                [rgb_image.copy(),
                 np.zeros_like(rgb_image[..., 0][..., None])],
                axis=-1)
            masked_rgb_image[..., 3] = mask_image
            masked_rgb_path = os.path.join(
                masked_rgb_folder[obj_id],
                os.path.basename(rgb_image_path).split(_file_extension)[0] +
                "png")
            cv2.imwrite(masked_rgb_path, masked_rgb_image)

            # Save a version of the depth image that only has non-zero depth on
            # the object and that is converted to mm if necessary.
            # - Check the scale of the depth.
            depth_path = os.path.join(scene_path, "depth",
                                      f"{image_idx:06d}.png")
            depth_masked_mm_path = os.path.join(
                scene_path, f"depth_masked_mm_obj_{obj_id}",
                f"{image_idx:06d}.png")
            if (depth_scale[split] is None):
                depth_scale[split] = scene_camera[image_idx]['depth_scale']
                os.makedirs(os.path.dirname(depth_masked_mm_path))
            else:
                assert (scene_camera[image_idx]['depth_scale'] ==
                        depth_scale[split])
            depth = cv2.imread(depth_path, flags=cv2.IMREAD_UNCHANGED)
            assert (depth.ndim == 2)
            if (depth_scale[split] != 1.):
                # - Convert to mm, if necessary.
                depth_masked_mm = depth * depth_scale[split]
                depth_masked_mm = depth_masked_mm.astype(depth.dtype)
            else:
                depth_masked_mm = depth
            assert (depth_masked_mm.dtype == np.uint16)
            # - Set depth to zero for the out-of-object pixels.
            y_out_mask, x_out_mask = np.where(np.logical_not(mask_image))
            depth_masked_mm[y_out_mask, x_out_mask] = 0
            cv2.imwrite(depth_masked_mm_path, depth_masked_mm)

            # Store the information into the transform dictionaries.
            image_dict["file_path"] = os.path.join(
                os.path.basename(os.path.dirname(masked_rgb_path)),
                os.path.basename(masked_rgb_path))
            image_dict["depth_path"] = os.path.relpath(depth_masked_mm_path,
                                                       scene_path)
            image_dict["transform_matrix"] = scene_T_Cs[split][obj_id][
                image_idx].tolist()

            # NOTE: The `deepcopy` here is crucial, because dictionaries are
            # mutable. A simple copy would not do the job, since also the dict
            # entries in this case are mutable.
            transforms[split][obj_id]["frames"].append(
                copy.deepcopy(image_dict))

# Save the transform dictionaries.
for id_curr_obj in objects_to_process:
    transforms[split][id_curr_obj]['enable_depth_loading'] = True
    transforms[split][id_curr_obj]['integer_depth_scale'] = transforms[split][
        id_curr_obj]['one_uom_scene_to_m'] / 1000. / depth_scale[split]
    with open(
            os.path.join(scene_path, f"transforms_{split}_obj_{id_curr_obj}"
                         ".json"), "w") as f:
        json.dump(transforms[split][id_curr_obj], f, indent=4)
