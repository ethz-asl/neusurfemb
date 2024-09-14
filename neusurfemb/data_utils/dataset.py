import cv2
import json
import numpy as np
import os
from scipy.spatial.distance import cdist
import trimesh
import tqdm

from neusurfemb.misc_utils.invert_pose import invert_pose
from neusurfemb.misc_utils.transforms import _W_NEUS_T_W_BOP, _W_NEUS_T_SCENE


def read_transform(root_path, obj_id):
    if (obj_id is None):
        obj_id_str = ""
    else:
        obj_id_str = f"_obj_{obj_id}"
    transform_path = os.path.join(root_path, f'transforms{obj_id_str}.json')
    transform_path_train_only = os.path.join(
        root_path, f'transforms_train{obj_id_str}.json')
    assert (
        os.path.exists(transform_path) or
        os.path.exists(transform_path_train_only)), (
            f"One of '{transform_path}' and '{transform_path_train_only}' must "
            "exist.")
    if (os.path.exists(transform_path)):
        assert (not os.path.exists(transform_path_train_only))
        with open(transform_path, 'r') as f:
            transform = json.load(f)
        actual_transform_path = transform_path
    if (os.path.exists(transform_path_train_only)):
        assert (not os.path.exists(transform_path))
        with open(transform_path_train_only, 'r') as f:
            transform = json.load(f)
        actual_transform_path = transform_path_train_only

    return transform, actual_transform_path


def transform_to_bop_info_files(transform_file_path: str, obj_id: int,
                                output_folder: str):
    r"""Converts a `transform.json` file compatible with NeuS2 training to: A
    ``scene_gt_info.json`, a `scene_gt.json`, and a `scene_camera.json` file
    compatible with a BOP dataset."""
    _W_surfemb_T_scene = invert_pose(_W_NEUS_T_W_BOP) @ _W_NEUS_T_SCENE

    # Read transform.
    with open(transform_file_path, "r") as f:
        transform = json.load(f)

    print("\033[94mGenerating BOP info files from transform "
          f"'{transform_file_path}'\033[0m.")

    # Generate `scene_gt_info.json` file.
    # - Copy masks and mask/crop info.
    mask_output_folder = os.path.join(output_folder, "mask")
    mask_visib_output_folder = os.path.join(output_folder, "mask_visib")
    os.makedirs(mask_output_folder, exist_ok=False)
    os.makedirs(mask_visib_output_folder, exist_ok=False)
    input_image_paths = sorted([
        os.path.join(os.path.dirname(transform_file_path), f['file_path'])
        for f in transform['frames']
    ])

    scene_gt_info = {}
    for frame_idx, input_image_path in enumerate(tqdm.tqdm(input_image_paths)):
        input_image = cv2.imread(input_image_path,
                                 cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]]
        assert (input_image.shape[-1] == 4)
        assert (set(np.unique(input_image[..., 3]).tolist()).issubset([0, 255]))
        input_mask_full_obj = ((input_image[..., 3] == 255).astype(float) *
                               255.).astype(np.uint8)

        # In the current implementation, the generated synthetic dataset does
        # not include occlusions.
        input_mask_visib = input_mask_full_obj.copy()

        file_extension = "." + input_image_path.rsplit(".")[-1]
        output_mask_visib_path = os.path.join(
            mask_visib_output_folder,
            os.path.basename(input_image_path).rsplit(
                file_extension, maxsplit=1)[0] + "_000000.png")
        cv2.imwrite(filename=output_mask_visib_path, img=input_mask_visib)
        output_mask_path = os.path.join(
            mask_output_folder,
            os.path.basename(input_image_path).rsplit(
                file_extension, maxsplit=1)[0] + "_000000.png")
        cv2.imwrite(filename=output_mask_path, img=input_mask_full_obj)

        # - Full object mask.
        assert (set(np.unique(input_mask_full_obj).tolist()).issubset([0, 255]))
        y_in_obj, x_in_obj = np.where(input_mask_full_obj == 255)
        x_in_obj_min = x_in_obj.min()
        x_in_obj_max = x_in_obj.max()
        y_in_obj_min = y_in_obj.min()
        y_in_obj_max = y_in_obj.max()
        bbox_obj = np.array([
            x_in_obj_min, y_in_obj_min, x_in_obj_max - x_in_obj_min,
            y_in_obj_max - y_in_obj_min
        ])
        px_count_all = np.count_nonzero(input_mask_full_obj == 255)
        # - Visible object mask.
        assert (set(np.unique(input_mask_visib).tolist()).issubset([0, 255]))
        y_in_obj_visib, x_in_obj_visib = np.where(input_mask_visib == 255)
        x_in_obj_visib_min = x_in_obj_visib.min()
        x_in_obj_visib_max = x_in_obj_visib.max()
        y_in_obj_visib_min = y_in_obj_visib.min()
        y_in_obj_visib_max = y_in_obj_visib.max()
        bbox_visib = np.array([
            x_in_obj_visib_min, y_in_obj_visib_min,
            x_in_obj_visib_max - x_in_obj_visib_min,
            y_in_obj_visib_max - y_in_obj_visib_min
        ])
        px_count_valid = px_count_all
        # The parameters `visib_fract` and `px_count_visib` are not computed to
        # avoid using this dataset with the assumption that occlusions were
        # already simulated: The intended use is that occlusions should be added
        # to the dataset in post processing.
        scene_gt_info[str(frame_idx)] = [{
            "bbox_obj": bbox_obj.tolist(),
            "bbox_visib": bbox_visib.tolist(),
            "px_count_all": px_count_all,
            "px_count_valid": px_count_valid,
        }]

    with open(os.path.join(output_folder, "scene_gt_info.json"), "w") as f:
        json.dump(obj=scene_gt_info, fp=f, indent=2)

    # Generate `scene_gt.json` file.
    scene_gt = {}
    for frame_idx, frame in enumerate(transform["frames"]):
        scene_T_C = np.array(frame["transform_matrix"]).reshape(4, 4)
        scene_T_C[:3, 3] = scene_T_C[:3, 3] / 1.e-3
        W_surfemb_T_C = _W_surfemb_T_scene @ scene_T_C
        W_surfemb_T_C[0:3, 1] *= -1
        W_surfemb_T_C[0:3, 2] *= -1
        C_T_W_surfemb = invert_pose(W_surfemb_T_C)

        C_R_W_surfemb = C_T_W_surfemb[:3, :3]
        C_t_W_surfemb = C_T_W_surfemb[:3, 3]
        scene_gt[str(frame_idx)] = [{
            "cam_R_m2c": C_R_W_surfemb.reshape(-1).tolist(),
            "cam_t_m2c": C_t_W_surfemb.tolist(),
            "obj_id": obj_id
        }]

    with open(os.path.join(output_folder, "scene_gt.json"), "w") as f:
        json.dump(obj=scene_gt, fp=f, indent=2)

    # Generate `scene_camera.json` file.
    scene_camera = {
        str(frame_idx): {
            "cam_K": [
                frame['fl_x'], 0.0, frame['cx'], 0.0, frame['fl_y'],
                frame['cy'], 0.0, 0.0, 1.0
            ]
        } for frame_idx, frame in enumerate(transform['frames'])
    }

    with open(os.path.join(output_folder, "scene_camera.json"), "w") as f:
        json.dump(obj=scene_camera, fp=f, indent=2)


def generate_models_info(mesh_path: str, obj_id: int):
    r"""Generates a `models_info.json` file for a given mesh.
    """
    # Load mesh.
    mesh = trimesh.load(mesh_path)
    vertices = np.asarray(mesh.vertices)
    x = vertices[..., 0]
    y = vertices[..., 1]
    z = vertices[..., 2]

    # Compute diameter of the mesh. Cf. https://stackoverflow.com/a/60955825.
    convex_hull_points = mesh.convex_hull.vertices
    max_dist_between_points = cdist(convex_hull_points,
                                    convex_hull_points,
                                    metric='euclidean').max()

    return {
        str(obj_id): {
            "diameter": max_dist_between_points,
            "min_x": x.min(),
            "min_y": y.min(),
            "min_z": z.min(),
            "size_x": x.max() - x.min(),
            "size_y": y.max() - y.min(),
            "size_z": z.max() - z.min(),
        }
    }
