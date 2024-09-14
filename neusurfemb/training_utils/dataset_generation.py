from collections import deque
import copy
import cv2
import glob
import imageio.v3 as imageio
import json
import numpy as np
import os
import pickle as pkl
import shutil
import skimage
import torch
import tqdm
from typing import Optional
import yaml

from neusurfemb.misc_utils.transforms import ngp_to_nerf_matrix, _W_NEUS_T_W_BOP
from neusurfemb.neus.network import NeuSNetwork


def generate_synthetic_dataset(
        model: torch.nn.Module,
        synthetic_data_config: dict,
        intrinsics: np.ndarray,
        H: int,
        W: int,
        output_folder: str,
        one_uom_scene_to_m: float,
        W_nerf_point_cloud: np.ndarray,
        orig_frame_T_oriented_bbox_frame: np.ndarray,
        obj_id: Optional[int] = None,
        in_mask_threshold: float = 0.99,
        connected_component_filtering: bool = False) -> str:
    r"""Creates a synthetic dataset by rendering frames from a set of random
    views in the sphere around the object.

    Args:
        model (neus.network.NeuSNetwork): NeuS model to query.
        synthetic_data_config (dict): Configuration of the synthetic dataset to
            generate. It should contain the following keys:
            - 'elevation_max', 'elevation_min': Respectively, maximum and
                minimum elevation in degrees from the horizon that the poses
                should span.
            - 'num_views': Number of views to generate according to the
                parameters.
            - 'fraction_obj_pixels_min_radius',
                'fraction_obj_pixels_max_radius': Respectively, maximum and
                minimum number of object pixels, expressed as a fraction of the
                total amount of image pixels, that should be contained in the
                rendered images. In practice, for the first viewpoint only, a
                maximum and a minimum distance from the scene center at which
                the poses should be sampled are binary-searched: At each step of
                the search the number of in-object pixels is counted, and the
                maximum and minimum distances are set to be found when this
                number of pixels is respectively approximately
                'fraction_obj_pixels_max_radius' and
                'fraction_obj_pixels_min_radius'.
            - `crop_res`: Resolution of the cropped image that gets rendered.
            - `crop_scale_factor`: Factor by which the object-tight bounding box
                used to define the boundaries of the cropped rendered image is
                enlarged. For instance, a value of 1.0 indicates that the
                bounding box is tight to the object boundaries.
            - `in_plane_rotation_max_degrees`: Maximum in-plane rotation (in
                degrees) that is applied to the camera.
        intrinsics (np.ndarray): Intrinsics of the camera to use from rendering,
            in the form of a tensor [fx, fy, cx, cy]. Will be adapted so that
            each image is cropped based on the reprojections of the 3-D bounding
            box of the object and rendered at the desired resolution.
        H (int): Height in pixel of the (depth) image to render to create the
            point cloud.
        W (int): Width in pixel of the (depth) image to render to create the
            point cloud.
        output_folder (str): Output folder where the augmented dataset should be
            saved. NOTE: If not existent, the function will try to create this
            folder. Files will have as name format `XX...XX.png`, where XX...XX
            indicates the position of the frame within the trajectory. If
            pre-existing files are found named according to the format above,
            they are kept and the corresponding frame is not generated again.
            An `images` subfolder contains the RGBA rendering, a `coordinates`
            subfolder contains the coordinate renderings, and a `pose` folder
            contains the camera-to-world poses, which are *NOT* expressed in
            NeRF format and are in *meters*.
        one_uom_scene_to_m (float): Conversion factor from the unit of
            measure of the scene to meters. NOTE: It assumed that the additional
            argument of the dataset `scale` is set to
            1 / `one_uom_scene_to_m`.
        W_nerf_point_cloud (np.ndarray): Point cloud of the object in NeRF
            coordinates, in the same format returned by
            `NeuSNetwork.compute_point_cloud`.
        orig_frame_T_oriented_bbox_frame (np.ndarray): Transformation matrix
            from the frame of the oriented bounding box that best fits the
            object to the original coordinate frame.
        obj_id (int, default=None): If not None, ID of the object, used simply
            to form the filename of the `transform.json` file.
        in_mask_threshold (float, default=0.99): All pixels that in the mask
            image rendered from NeRF have a value above this threshold are
            considered to be part of the object and included in the foreground.
            All other pixels are set to transparent.
        connected_component_filtering (bool, default=False): If True, filtering
            is performed by further enforcing that the object mask is composed
            of large-enough connected components of pixels.
    
    Returns:
        transform_file_path (str): Path to the JSON transform file.
    """
    assert (isinstance(synthetic_data_config, dict))
    _required_config_keys = [
        'elevation_max', 'elevation_min', 'fraction_obj_pixels_max_radius',
        'fraction_obj_pixels_min_radius', 'num_views', 'crop_res',
        'crop_scale_factor', 'in_plane_rotation_max_degrees'
    ]
    assert (sorted(list(
        synthetic_data_config.keys())) == sorted(_required_config_keys)), (
            "The synthetic configuration file should contain exactly the "
            f"following keys: {_required_config_keys}.")
    assert (isinstance(W_nerf_point_cloud, np.ndarray) and
            W_nerf_point_cloud.ndim == 2 and W_nerf_point_cloud.shape[-1] == 3)
    assert (orig_frame_T_oriented_bbox_frame is not None), (
        "To generate synthetic data, please specify whether to compute an "
        "oriented bounding box around the object (flag "
        "`compute_oriented_bounding_box`).")

    elevation_max = synthetic_data_config['elevation_max']
    elevation_min = synthetic_data_config['elevation_min']
    num_views = synthetic_data_config['num_views']
    fraction_obj_pixels_min_radius = synthetic_data_config[
        'fraction_obj_pixels_min_radius']
    fraction_obj_pixels_max_radius = synthetic_data_config[
        'fraction_obj_pixels_max_radius']
    assert (fraction_obj_pixels_max_radius < fraction_obj_pixels_min_radius)
    crop_res = synthetic_data_config['crop_res']
    crop_scale_factor = synthetic_data_config['crop_scale_factor']
    in_plane_rotation_max_degrees = synthetic_data_config[
        'in_plane_rotation_max_degrees']

    specs_original_dataset = {
        "one_uom_scene_to_m": one_uom_scene_to_m,
        "scale": 1. / one_uom_scene_to_m
    }
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]

    # Create the folder if not existing and find the frames that are already
    # existing.
    rgb_folder = os.path.join(output_folder, "images")
    coordinate_folder = os.path.join(output_folder, "coordinates")
    pose_folder = os.path.join(output_folder, "pose")
    intrinsics_folder = os.path.join(output_folder, "intrinsics")
    if (obj_id is None):
        obj_id_str = ""
    else:
        obj_id_str = f"_obj_{obj_id}"
    transform_file_path = os.path.join(output_folder,
                                       f"transforms{obj_id_str}.json")
    synthetic_data_config_path = os.path.join(output_folder,
                                              "synthetic_data_config.yml")
    os.makedirs(name=output_folder, exist_ok=True)
    os.makedirs(name=rgb_folder, exist_ok=True)
    os.makedirs(name=coordinate_folder, exist_ok=True)
    os.makedirs(name=pose_folder, exist_ok=True)
    os.makedirs(name=intrinsics_folder, exist_ok=True)
    existing_frame_paths = sorted(glob.glob(os.path.join(rgb_folder, "*.png")))
    existing_frames = deque()
    if (len(existing_frame_paths) > 0):
        assert (os.path.exists(transform_file_path))
        with open(transform_file_path, "r") as f:
            transform = json.load(f)
        for key in transform.keys():
            if (key != "frames"):
                assert (transform[key] == specs_original_dataset[key])
        assert (os.path.exists(synthetic_data_config_path)), (
            f"A config file '{synthetic_data_config_path}' could not be found "
            "for the existing (possibly partial) dataset at "
            f"'{output_folder}'. Select a different output folder or add the "
            "config file for the existing dataset.")
    else:
        # Initialize transform file path.
        transform = specs_original_dataset
        with open(transform_file_path, "w") as f:
            json.dump(transform, f, indent=2)
    # Check for an existing config file that matches the current config, or save
    # the current config to file.
    if (os.path.exists(synthetic_data_config_path)):
        with open(synthetic_data_config_path, "r") as f:
            existing_config = yaml.load(f, Loader=yaml.Loader)
        assert (existing_config == synthetic_data_config), (
            "The content of the existing config file "
            f"'{synthetic_data_config_path}' found for the dataset in "
            f"'{output_folder}' does not match the given parameters "
            f"'{synthetic_data_config}'. Select a different output path or "
            "change the input config.")
    else:
        with open(synthetic_data_config_path, "w") as f:
            yaml.dump(synthetic_data_config, f)
    for existing_frame_path in existing_frame_paths:
        existing_frame = os.path.basename(existing_frame_path).split(".png")[0]
        assert (
            existing_frame.isnumeric() and int(existing_frame) < num_views), (
                f"Found frame '{existing_frame}' with invalid format. Please "
                "remove it.")
        # If coordinates or pose are missing, regenerate frame.
        if (os.path.exists(
                os.path.join(coordinate_folder, f"{existing_frame}.npy")) and
                os.path.exists(
                    os.path.join(pose_folder, f"{existing_frame}.txt"))):
            existing_frames.append(int(existing_frame))
        else:
            print("\033[93mFound existing rendered frame "
                  f"'{existing_frame_path}', but the corresponding coordinate "
                  "and/or pose files were not found. Regenerating the frame."
                  "\033[0m")

    # Binary-search for maximum and minimum radius based on the input parameters
    # on the number of in-object pixels.
    radius_max, actual_fraction_obj_pixels_max_radius = binary_search_radius(
        target_object_pixel_fraction=fraction_obj_pixels_max_radius,
        model=model,
        intrinsics=intrinsics,
        H=H,
        W=W,
        in_mask_threshold=in_mask_threshold)
    radius_min, actual_fraction_obj_pixels_min_radius = binary_search_radius(
        target_object_pixel_fraction=fraction_obj_pixels_min_radius,
        model=model,
        intrinsics=intrinsics,
        H=H,
        W=W,
        in_mask_threshold=in_mask_threshold,
        max_radius_search=radius_max)
    assert (radius_min < radius_max and actual_fraction_obj_pixels_max_radius
            < actual_fraction_obj_pixels_min_radius)

    print("\033[94mTo generate the synthetic dataset, the maximum radius "
          f"{radius_max:.3f} and the minimum radius {radius_min:.3f} were "
          "selected, for which respectively "
          f"{actual_fraction_obj_pixels_max_radius * 100:.2f}% and "
          f"{actual_fraction_obj_pixels_min_radius * 100:.2f}% of the pixels "
          "belong to the object.\033[0m")

    # Generate the viewpoints.
    W_T_Cs_traj = pose_random_from_hemisphere(num_views=num_views,
                                              elevation_min=elevation_min,
                                              elevation_max=elevation_max,
                                              radius_min=radius_min,
                                              radius_max=radius_max)
    # For each viewpoint, render if necessary.
    if (len(existing_frames) > 0):
        curr_existing_frame = existing_frames.popleft()
    else:
        curr_existing_frame = None

    with torch.no_grad():
        for view_idx, W_T_C_curr_frame in enumerate(tqdm.tqdm(W_T_Cs_traj)):
            frame_idx = view_idx
            rgb_path = os.path.join(rgb_folder, f"{frame_idx:06d}.png")
            coordinate_path = os.path.join(coordinate_folder,
                                           f"{frame_idx:06d}.npy")
            pose_path = os.path.join(pose_folder, f"{frame_idx:06d}.txt")
            intrinsics_path = os.path.join(intrinsics_folder,
                                           f"{frame_idx:06d}.pkl")

            if (curr_existing_frame is None or
                    curr_existing_frame != frame_idx):
                W_T_C_curr_frame_adjusted = (orig_frame_T_oriented_bbox_frame
                                             @ W_T_C_curr_frame.cpu().numpy())

                # - Apply random in-plane rotation.
                theta = np.random.uniform(
                    -np.deg2rad(in_plane_rotation_max_degrees),
                    np.deg2rad(in_plane_rotation_max_degrees))
                C_T_C_perturbed = np.array(
                    [[np.cos(theta), -np.sin(theta), 0., 0.],
                     [np.sin(theta), np.cos(theta), 0., 0.], [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])
                W_T_C_curr_frame_adjusted = (
                    W_T_C_curr_frame_adjusted @ C_T_C_perturbed)

                # - Render color.
                render_rgb, _, K_cropped_and_rescaled = (
                    model.render_from_given_pose_given_point_cloud(
                        K=K,
                        W_nerf_T_C=ngp_to_nerf_matrix(
                            pose=W_T_C_curr_frame_adjusted,
                            scale=transform['scale']),
                        W_nerf_point_cloud=W_nerf_point_cloud,
                        final_crop_res=crop_res,
                        dataset_scale=transform['scale'],
                        render_mode="color",
                        crop_scale_factor=crop_scale_factor))
                render_mask = render_rgb[..., 3].reshape(crop_res, crop_res)
                pixels_in_mask = (render_mask > in_mask_threshold).reshape(
                    crop_res, crop_res)

                if (connected_component_filtering):
                    # Filter the mask so as to remove small artifacts, by
                    # keeping only the large-enough connected components.
                    min_num_pixels_conn_comp = (
                        H * W) * actual_fraction_obj_pixels_max_radius / 5.
                    pixels_in_mask = keep_large_connected_components(
                        input_mask=pixels_in_mask,
                        min_num_pixels_conn_comp=min_num_pixels_conn_comp)

                assert (np.count_nonzero(pixels_in_mask) > 0)

                # Update the alpha channel using the in-pixel mask.
                render_rgb[..., 3] = pixels_in_mask

                # Save the image to file.
                imageio.imwrite(image=(render_rgb * 255).astype(np.uint8),
                                uri=rgb_path)

                if (False):
                    # - Render coordinates.
                    cropped_coordinate_image, _, K_cropped_and_rescaled_coord = (
                        model.render_from_given_pose_given_point_cloud(
                            K=K,
                            W_nerf_T_C=ngp_to_nerf_matrix(
                                pose=W_T_C_curr_frame_adjusted,
                                scale=transform['scale']),
                            W_nerf_point_cloud=W_nerf_point_cloud,
                            final_crop_res=crop_res,
                            dataset_scale=transform['scale'],
                            render_mode="coordinate",
                            crop_scale_factor=crop_scale_factor))
                    assert (np.all(
                        np.isclose(K_cropped_and_rescaled,
                                K_cropped_and_rescaled_coord)))
                    cropped_coordinate_image = cropped_coordinate_image[..., :3]

                    # Transform the coordinate image to mm and to the BOP coordinate
                    # frame.
                    is_coordinate_valid = np.any(cropped_coordinate_image != 0,
                                                axis=-1)
                    cropped_coordinate_image[is_coordinate_valid] = (
                        (np.linalg.inv(_W_NEUS_T_W_BOP) @ np.hstack([
                            cropped_coordinate_image[is_coordinate_valid],
                            np.ones_like(cropped_coordinate_image[
                                is_coordinate_valid][..., 0][..., None])
                        ]).T).T[..., :3] * one_uom_scene_to_m * 1000.)

                    with open(coordinate_path, "wb") as f:
                        np.save(arr=cropped_coordinate_image, file=f)

                # - Save the intrinsic parameters.
                with open(intrinsics_path, "wb") as f:
                    pkl.dump(
                        {
                            "K": K_cropped_and_rescaled,
                            "h": crop_res,
                            "w": crop_res
                        }, f)

                # Save the pose (in m, and *NOT* in NeRF format).
                # NOTE: This needs to be the full pose w.r.t. the original world
                # frame, hence it needs to take into account also the alignment
                # to the oriented bounding box.
                pose_to_output = ngp_to_nerf_matrix(
                    pose=W_T_C_curr_frame_adjusted, scale=transform['scale'])
                np.savetxt(fname=pose_path, X=pose_to_output)

            if (len(existing_frames) > 0):
                curr_existing_frame = existing_frames.popleft()
            else:
                curr_existing_frame = None

    # Complete `transform.json` path.
    transform["frames"] = []
    image_paths_already_in_transform = []
    image_folder = os.path.abspath(
        os.path.join(os.path.dirname(transform_file_path), "images"))
    coordinate_folder = os.path.join(os.path.dirname(image_folder),
                                     "coordinates")
    intrinsics_folder = os.path.join(os.path.dirname(image_folder),
                                     "intrinsics")
    pose_folder = os.path.join(os.path.dirname(image_folder), "pose")

    # List all images in the image folder.
    for image_path in sorted(glob.glob(os.path.join(image_folder, "*.png"))):
        # Find the images that are not yet in the frame list and add them.
        if (not image_path in image_paths_already_in_transform):
            # - Check that for each image, a pose file and a corresponding
            #   coordinate file are available.
            image_name = os.path.basename(image_path).split(".png")[0]
            pose_file_path = (os.path.join(os.path.dirname(image_folder),
                                           "pose", f"{image_name}.txt"))
            coordinate_file_path = os.path.join(coordinate_folder,
                                                f"{image_name}.npy")
            intrinsics_file_path = os.path.join(intrinsics_folder,
                                                f"{image_name}.pkl")
            assert (os.path.exists(pose_file_path))
            # assert (os.path.exists(coordinate_file_path))
            assert (os.path.exists(intrinsics_file_path))
            pose = np.loadtxt(pose_file_path)
            with open(intrinsics_file_path, "rb") as f:
                intrinsics = pkl.load(f)
            K = intrinsics['K']

            transform["frames"].append({
                'file_path':
                    os.path.relpath(path=image_path, start=output_folder),
                'transform_matrix':
                    pose.tolist(),
                'fl_x':
                    K[0, 0],
                'fl_y':
                    K[1, 1],
                'cx':
                    K[0, 2],
                'cy':
                    K[1, 2],
                'h':
                    intrinsics['h'],
                'w':
                    intrinsics['w']
            })
    # Overwrite the transform file.
    with open(transform_file_path, "w") as f:
        json.dump(transform, f, indent=2)

    # Remove the temporary intrinsics and pose folder.
    try:
        shutil.rmtree(intrinsics_folder)
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(pose_folder)
    except FileNotFoundError:
        pass

    return transform_file_path


def crop_and_resize_train_dataset(model: NeuSNetwork, crop_res: int,
                                  crop_scale_factor: float,
                                  one_uom_scene_to_m: float,
                                  W_nerf_point_cloud: np.ndarray,
                                  path_to_scene_json: str,
                                  output_folder_path: str):
    r"""Crops a real dataset to the desired resolution, based on the 3-D
    bounding box of the fitted object, and renders per-pixel 3-D coordinates
    using the trained NeuS model.

    Args:
        model (neus.network.NeuSNetwork): Trained NeuS network.
        crop_res (int): Desired cropping resolution.
        crop_scale_factor (float): Factor by which the object-tight bounding box
            used to define the boundaries of the cropped image is enlarged. For
            instance, a value of 1.0 indicates that the bounding box is tight to
            the object boundaries.
        one_uom_scene_to_m (float): Conversion factor from the unit of measure
            of the scene to meters.
        W_nerf_point_cloud (np.ndarray): Point cloud of the object in NeRF
            coordinates, in the same format returned by
            `NeuSNetwork.compute_point_cloud`.
        path_to_scene_json (str): Path to the JSON transform file of the given
            scene.
        output_folder_path (str): Path to the output folder that will contain a
            subfolders `images` and `coordinates` with the cropped and resized
            version of the dataset.

    Returns:
        new_transform_path (str): Path to the new transform file.
    """
    # Load dataset JSON file.
    with open(path_to_scene_json, "r") as f:
        transform = json.load(f)
    new_transform = copy.deepcopy(transform)

    cropped_image_folder = os.path.join(output_folder_path, "images")
    cropped_coordinate_folder = os.path.join(output_folder_path, "coordinates")
    new_transform_path = os.path.join(output_folder_path,
                                      os.path.basename(path_to_scene_json))
    os.makedirs(output_folder_path, exist_ok=True)
    os.makedirs(cropped_image_folder, exist_ok=True)
    os.makedirs(cropped_coordinate_folder, exist_ok=True)

    print("\033[94mCropping the input dataset to the desired crop resolution "
          f"of {crop_res} and rendering coordinate maps. The new transform "
          f"file will be written at '{new_transform_path}'\033[0m")

    try:
        new_transform.pop("fl_x")
        assert ("fl_y" in new_transform and "cx" in new_transform and
                "cy" in new_transform and "camera_angle_x" in new_transform and
                "camera_angle_y" in new_transform)
        new_transform.pop("fl_y")
        new_transform.pop("cx")
        new_transform.pop("cy")
        new_transform.pop("camera_angle_x")
        new_transform.pop("camera_angle_y")
    except KeyError:
        pass

    for image_idx in tqdm.tqdm(range(len(transform["frames"]))):
        rgb_path = os.path.join(cropped_image_folder, f"{image_idx:06d}.png")
        coordinate_path = os.path.join(cropped_coordinate_folder,
                                       f"{image_idx:06d}.npy")
        try:
            K = np.array([[
                transform["frames"][image_idx]["fl_x"], 0.,
                transform["frames"][image_idx]["cx"]
            ],
                          [
                              0., transform["frames"][image_idx]["fl_y"],
                              transform["frames"][image_idx]["cy"]
                          ], [0., 0., 1.]])
        except KeyError:
            K = np.array([[transform["fl_x"], 0., transform["cx"]],
                          [0., transform["fl_y"], transform["cy"]],
                          [0., 0., 1.]])
        try:
            H = transform["frames"][image_idx]['h']
            W = transform["frames"][image_idx]['w']
            assert (not ("h" in transform or "w" in transform))
        except KeyError:
            H = transform["h"]
            W = transform["w"]

        cropped_coordinate_image, (
            min_u_corners, max_u_corners, min_v_corners,
            max_v_corners), K_cropped_and_rescaled = (
                model.render_from_given_pose_given_point_cloud(
                    K=K,
                    W_nerf_T_C=np.array(
                        transform["frames"][image_idx]["transform_matrix"]),
                    W_nerf_point_cloud=W_nerf_point_cloud,
                    final_crop_res=crop_res,
                    dataset_scale=transform["scale"],
                    render_mode="coordinate",
                    min_valid_x=0,
                    max_valid_x=W - 1,
                    min_valid_y=0,
                    max_valid_y=H - 1,
                    crop_scale_factor=crop_scale_factor))

        cropped_image = cv2.imread(
            os.path.join(os.path.dirname(path_to_scene_json),
                         transform["frames"][image_idx]['file_path']),
            cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]]

        assert ((max_u_corners - min_u_corners) == (max_v_corners -
                                                    min_v_corners))

        if (min_u_corners < 0 or max_u_corners >= W or min_v_corners < 0 or
                max_v_corners >= H):
            # Pad if necessary.
            add_left = -min(0, min_u_corners - 0)
            add_right = -min(0, W - 1 - max_u_corners)
            add_top = -min(0, min_v_corners - 0)
            add_bottom = -min(0, H - 1 - max_v_corners)
            cropped_image = np.pad(cropped_image,
                                   ((add_top, add_bottom),
                                    (add_left, add_right), (0, 0)))

            if (add_left > 0):
                min_u_corners = 0
                max_u_corners = max_u_corners + add_left
            if (add_right > 0):
                max_u_corners = max_u_corners + add_right
            if (add_top > 0):
                min_v_corners = 0
                max_v_corners = max_v_corners + add_top
            if (add_bottom > 0):
                max_v_corners = max_v_corners + add_bottom

            cropped_image = cv2.resize(
                cropped_image[min_v_corners:max_v_corners + 1,
                              min_u_corners:max_u_corners + 1],
                (crop_res, crop_res),
                interpolation=cv2.INTER_NEAREST) / 255.
        else:
            cropped_image = cv2.resize(
                cropped_image[min_v_corners:max_v_corners + 1,
                              min_u_corners:max_u_corners + 1],
                (crop_res, crop_res),
                interpolation=cv2.INTER_NEAREST) / 255.

        assert (np.unique(cropped_image[..., 3]).tolist() == [0., 1.])
        cropped_coordinate_image = cropped_coordinate_image[..., :3]

        # Transform the coordinate image to mm and to the BOP coordinate frame.
        is_coordinate_valid = np.any(cropped_coordinate_image != 0, axis=-1)
        cropped_coordinate_image[is_coordinate_valid] = (
            (np.linalg.inv(_W_NEUS_T_W_BOP) @ np.hstack([
                cropped_coordinate_image[is_coordinate_valid],
                np.ones_like(cropped_coordinate_image[is_coordinate_valid][
                    ..., 0][..., None])
            ]).T).T[..., :3] * one_uom_scene_to_m * 1000.)

        # Update the frame intrinsics.
        new_transform["frames"][image_idx]["fl_x"] = K_cropped_and_rescaled[0,
                                                                            0]
        new_transform["frames"][image_idx]["fl_y"] = K_cropped_and_rescaled[1,
                                                                            1]
        new_transform["frames"][image_idx]["cx"] = K_cropped_and_rescaled[0, 2]
        new_transform["frames"][image_idx]["cy"] = K_cropped_and_rescaled[1, 2]
        new_transform["frames"][image_idx]["h"] = crop_res
        new_transform["frames"][image_idx]["w"] = crop_res
        new_transform["frames"][image_idx]["file_path"] = os.path.relpath(
            path=rgb_path, start=output_folder_path)
        try:
            new_transform["frames"][image_idx].pop("camera_angle_x")
            new_transform["frames"][image_idx].pop("camera_angle_y")
            new_transform["frames"][image_idx].pop("depth_path")
        except KeyError:
            pass

        # Save the data.
        imageio.imwrite(image=(cropped_image * 255).astype(np.uint8),
                        uri=rgb_path)
        with open(coordinate_path, "wb") as f:
            np.save(arr=cropped_coordinate_image, file=f)

    # Save the new transform file.
    with open(new_transform_path, "w") as f:
        json.dump(obj=new_transform, fp=f, indent=2)

    return new_transform_path


# Methods to generate a spherical camera trajectory.
def _trans_t(t):
    r"""Taken from https://github.com/sxyu/svox2/blob/master/opt/util/util.py.
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _rot_phi(phi):
    r"""Taken from https://github.com/sxyu/svox2/blob/master/opt/util/util.py.
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _rot_theta(th):
    r"""Taken from https://github.com/sxyu/svox2/blob/master/opt/util/util.py.
    """
    return np.array(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _pose_spherical(theta: float,
                    phi: float,
                    radius: float,
                    offset: Optional[np.ndarray] = None,
                    vec_up: Optional[np.ndarray] = None):
    """
    Taken from https://github.com/sxyu/svox2/blob/master/opt/util/util.py.

    Generate spherical rendering poses, from NeRF. Forgive the code horror
    :return: r (3,), t (3,)
    """
    c2w = _trans_t(radius)
    c2w = _rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = _rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (np.array(
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    ) @ c2w)
    if (vec_up is not None):
        vec_up = vec_up / np.linalg.norm(vec_up)
        vec_1 = np.array([vec_up[0], -vec_up[2], vec_up[1]])
        vec_2 = np.cross(vec_up, vec_1)

        trans = np.eye(4, 4, dtype=np.float32)
        trans[:3, 0] = vec_1
        trans[:3, 1] = vec_2
        trans[:3, 2] = vec_up
        c2w = trans @ c2w
    c2w = c2w @ np.diag(np.array([-1, 1, -1, 1], dtype=np.float32))
    if (offset is not None):
        c2w[:3, 3] += offset

    return c2w


def pose_spiral(num_views=150,
                elevation=-45.0,
                elevation2=-12.0,
                radius=0.85,
                offset=np.array([0.0, 0.0, 0.0]),
                vec_up_nerf_format=np.array([0., -1., 0.]),
                device=torch.device('cuda:0')):
    r"""Adapted from https://github.com/sxyu/svox2/blob/master/opt/
    render_imgs_circle.py"""
    assert (num_views % 2 == 0)
    num_views = num_views // 2
    angles = np.linspace(-180, 180, num_views + 1)[:-1]
    elevations = np.linspace(elevation, elevation2, num_views)
    c2ws = [
        _pose_spherical(
            angle,
            ele,
            radius,
            offset,
            vec_up=vec_up_nerf_format,
        ) for ele, angle in zip(elevations, angles)
    ]
    c2ws += [
        _pose_spherical(
            angle,
            ele,
            radius,
            offset,
            vec_up=vec_up_nerf_format,
        ) for ele, angle in zip(reversed(elevations), angles)
    ]

    c2ws = np.stack(c2ws, axis=0)

    return c2ws


def pose_random_from_hemisphere(num_views,
                                elevation_min,
                                elevation_max,
                                radius_min,
                                radius_max,
                                offset=np.array([0.0, 0.0, 0.0]),
                                vec_up_nerf_format=np.array([0., -1., 0.]),
                                device=torch.device('cuda:0')):
    r"""Adapted from https://github.com/sxyu/svox2/blob/master/opt/
    render_imgs_circle.py"""
    assert (elevation_min <= elevation_max)
    assert (radius_min <= radius_max)
    angles = -180 + 360 * np.random.rand(num_views)
    elevations = elevation_min + (elevation_max -
                                  elevation_min) * np.random.rand(num_views)
    radii = radius_min + (radius_max - radius_min) * np.random.rand(num_views)
    c2ws = [
        _pose_spherical(
            angle,
            ele,
            radius,
            offset,
            vec_up=vec_up_nerf_format,
        ) for ele, angle, radius in zip(elevations, angles, radii)
    ]
    c2ws = np.stack(c2ws, axis=0)
    c2ws = torch.from_numpy(c2ws).to(device=device)

    return c2ws


@torch.no_grad()
def binary_search_radius(target_object_pixel_fraction,
                         model,
                         intrinsics,
                         H,
                         W,
                         in_mask_threshold,
                         relative_tol=0.05,
                         max_radius_search=20.,
                         min_radius_search=0.15,
                         connected_component_filtering=False):
    """ Binary-search for a radius such that approximately the specified
    target fraction of the image pixels are occupied by the object.
    """

    assert (max_radius_search > min_radius_search)

    fraction_pixels_in_mask = 0.

    radius = None
    prev_radius = min_radius_search

    dataset_scale = model.neus.nerf.training.dataset.scale

    while (np.abs(fraction_pixels_in_mask - target_object_pixel_fraction) /
           target_object_pixel_fraction > relative_tol):
        radius = (min_radius_search + max_radius_search) / 2.
        if (np.abs(radius - prev_radius) / prev_radius < 1.e-4):
            raise ValueError("Unable to binary-search radius. Detected "
                             "infinite loop.")
        W_T_Cs_traj = pose_spiral(num_views=2, radius=radius, elevation=0)
        # - Render from the first view.
        K = np.eye(3)
        K[0, 0] = intrinsics[0]
        K[1, 1] = intrinsics[1]
        K[0, 2] = intrinsics[2]
        K[1, 2] = intrinsics[3]

        for idx in range(len(W_T_Cs_traj)):
            W_T_Cs_traj[idx] = ngp_to_nerf_matrix(pose=W_T_Cs_traj[idx],
                                                  scale=dataset_scale)

        outputs = model.render_from_given_pose(K=K,
                                               W_nerf_T_C=W_T_Cs_traj[0],
                                               H=H,
                                               W=W,
                                               render_mode="color")
        render_mask = outputs[..., 3].reshape(H, W)
        pixels_in_mask = (render_mask > in_mask_threshold)

        if (connected_component_filtering):
            pixels_in_mask = keep_large_connected_components(
                input_mask=pixels_in_mask,
                min_num_pixels_conn_comp=(target_object_pixel_fraction * H * W /
                                          2.))
        fraction_pixels_in_mask = np.count_nonzero(pixels_in_mask) / (H * W)

        if (np.count_nonzero(pixels_in_mask) > 0):
            # Discard hypotheses in which the object is not centered in the
            # image: These might be some artifacts, which might happen
            # especially when using depth supervision.
            y_mask, x_mask = np.where(pixels_in_mask)
            min_x, max_x, min_y, max_y = (x_mask.min(), x_mask.max(),
                                          y_mask.min(), y_mask.max())
            if (not ((min_x < W / 2 < max_x) and (min_y < H / 2 < max_y))):
                fraction_pixels_in_mask = 0.

        if (fraction_pixels_in_mask < target_object_pixel_fraction):
            max_radius_search = copy.deepcopy(radius)
        else:
            min_radius_search = copy.deepcopy(radius)

        prev_radius = copy.deepcopy(radius)

    assert (radius is not None)

    return radius, fraction_pixels_in_mask


def keep_large_connected_components(
        input_mask: np.ndarray, min_num_pixels_conn_comp: int) -> np.ndarray:
    connected_components = skimage.measure.label(input_mask.copy(),
                                                 background=0)
    labels_conn_comp = np.unique(connected_components)
    output_mask = None
    assert (np.all(labels_conn_comp == np.arange(len(labels_conn_comp))))
    if (len(labels_conn_comp) > 2):
        for label_conn_comp in labels_conn_comp[1:]:
            if (np.count_nonzero(connected_components == label_conn_comp)
                    < min_num_pixels_conn_comp):
                connected_components[connected_components ==
                                     label_conn_comp] = 0
        output_mask = connected_components != 0
    else:
        assert (labels_conn_comp[0] == 0)
        output_mask = (connected_components == 1)

    return output_mask