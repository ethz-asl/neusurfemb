import glob
import numpy as np
import os
import time
import torch
from tqdm import tqdm
import trimesh
from typing import Optional

import pyngp as ngp

from third_party.NeuS2.scripts.common import ROOT_DIR, repl

from neusurfemb.misc_utils.image_utils import make_crop_square
from neusurfemb.misc_utils.transforms import nerf_matrix_to_ngp


class NeuSNetwork:

    def __init__(self, device: torch.device = "cuda:0"):

        # Set up basic NeuS network.
        self.neus = ngp.Testbed(ngp.TestbedMode.Nerf)
        configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
        base_network = os.path.join(configs_dir, "base.json")
        self.neus.set_path_to_sdf_weight_folder(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "third_party/NeuS2/utils/"))
        self.neus.reload_network_from_file(base_network)

    def compute_point_cloud(self):
        return self.compute_point_cloud_from_mesh(mesh=self.compute_mesh())

    def compute_point_cloud_from_mesh(self, mesh: trimesh.Trimesh):
        pc_mc_points = np.array(mesh.vertices)
        pc_mc_colors = np.array(mesh.visual.vertex_colors[..., :3] / 255.)
        pc_mc_normals = np.array(mesh.vertex_normals)

        return pc_mc_points, pc_mc_colors, pc_mc_normals

    def compute_mesh(self):
        mesh = self.neus.compute_marching_cubes_mesh(
            resolution=np.array([512, 512, 512]))
        # Center mesh in the origin, between -0.5 and 0.5 (`instant-ngp` uses
        # [0, 1] bounds).
        mesh = trimesh.Trimesh(vertices=mesh['V'] - self.neus.aabb.center(),
                               faces=mesh['F'],
                               vertex_normals=mesh['N'],
                               vertex_colors=mesh['C'])
        mesh.remove_degenerate_faces()

        return mesh

    def load_checkpoint_from_folder(self, checkpoint_folder):
        neus_checkpoint_subfolder = os.path.join(checkpoint_folder, "neus")
        checkpoint_paths = sorted(
            glob.glob(os.path.join(neus_checkpoint_subfolder, f"*.msgpack")))
        assert (len(checkpoint_paths) == 1), checkpoint_paths
        checkpoint_path = checkpoint_paths[0]

        self.neus.load_snapshot(checkpoint_path)
        print("\033[94mLoaded pre-existing NeuS2 checkpoint "
              f"'{checkpoint_path}'.\033[0m")

    def _train_neus(self, path_to_scene_json, num_steps, checkpoint_folder):
        # Load training data.
        self.neus.load_training_data(path_to_scene_json)

        self.neus.shall_train = True
        self.neus.nerf.render_with_camera_distortion = False

        # Train if necessary.
        neus_checkpoint_subfolder = os.path.join(checkpoint_folder, "neus")
        os.makedirs(neus_checkpoint_subfolder, exist_ok=True)
        checkpoint_path = os.path.join(neus_checkpoint_subfolder,
                                       f"{num_steps}.msgpack")

        if (os.path.exists(checkpoint_path)):
            self.neus.load_snapshot(checkpoint_path)
            print("\033[94mLoaded pre-existing NeuS2 checkpoint "
                  f"'{checkpoint_path}'.\033[0m")
        else:
            old_training_step = 0
            tqdm_last_update = 0

            print("\033[94mTraining NeuS2 for reconstruction.\033[0m")

            with tqdm(desc="Training", total=num_steps, unit="step") as t:
                while self.neus.frame():
                    if (self.neus.want_repl()):
                        repl(self.neus)
                    if (self.neus.training_step >= num_steps):
                        break
                    # Update progress bar
                    if (self.neus.training_step < old_training_step or
                            old_training_step == 0):
                        old_training_step = 0
                        t.reset()

                    now = time.monotonic()
                    if (now - tqdm_last_update > 0.1):
                        t.update(self.neus.training_step - old_training_step)
                        t.set_postfix(loss=self.neus.loss)
                        old_training_step = self.neus.training_step
                        tqdm_last_update = now

            # Save checkpoint.
            self.neus.save_snapshot(path=checkpoint_path)
            print(f"\033[94mSaved NeuS checkpoint at '{checkpoint_path}'."
                  "\033[0m")

        self.neus.shall_train = False

    def render_from_given_pose(
            self,
            K: np.ndarray,
            W_nerf_T_C: np.ndarray,
            H: int,
            W: int,
            render_mode: str,
            spp: int = 8,
            threshold_coordinates_filtering: Optional[float] = None):
        assert (isinstance(K, np.ndarray) and K.shape == (3, 3))
        assert (isinstance(W_nerf_T_C, np.ndarray) and
                W_nerf_T_C.shape == (4, 4))
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        self.neus.background_color = [0., 0., 0., 0.]
        self.neus.snap_to_pixel_centers = True
        self.neus.nerf.rendering_min_transmittance = 1e-4
        self.neus.reset_camera()
        # Set extrinsics.
        self.neus.set_nerf_camera_matrix(W_nerf_T_C[:-1, :])
        # Set intrinsics.
        self.neus.screen_center = (1. - cx / W, 1. - cy / H)
        self.neus.relative_focal_length = (fx / (W, H)[self.neus.fov_axis],
                                           fy / (W, H)[self.neus.fov_axis])
        if (render_mode == "color"):
            linear = False
            self.neus.render_mode = ngp.RenderMode.Shade
        elif (render_mode in ["coordinate", "depth"]):
            linear = True
            self.neus.render_mode = ngp.RenderMode.Positions
            if (threshold_coordinates_filtering is None):
                print("\033[93mWARNING: Rendering coordinates/depth without a "
                      "specified threshold for the norm of the gradient of the "
                      "3-D coordinates w.r.t. the 2-D coordinates. This will "
                      "likely create noise at the object discontinuities."
                      "\033[0m")
        else:
            raise ValueError()

        rendered_image = self.neus.render(W, H, spp, linear=linear)

        if (render_mode in ["coordinate", "depth"]):
            # Convert the coordinates to the format such that the scene bound is
            # [-0.5, 0.5].
            # - Undo the coordinate transformation done in
            #   `composite_kernel_nerf`.
            assert (np.all(self.neus.aabb.center() ==
                           self.neus.nerf.training.dataset.offset))
            # Some coordinates might be [0., 0., 0.] or very close, likely due
            # to some bugs in NeuS. -> Filter them out.
            coordinates_to_ignore = np.where(
                np.all(np.abs(rendered_image[..., :3]) < [5.e-2, 5.e-2, 5.e-2],
                       axis=-1))

            rendered_image[..., :3] = 2 * (rendered_image[..., :3] - 0.25)
            # - Center point cloud in the origin (`instant-ngp` uses [0, 1]
            #   bounds).
            rendered_image[
                ..., :3] = rendered_image[..., :3] - self.neus.aabb.center()

            if (threshold_coordinates_filtering is not None):
                # Optionally filter the coordinates based on the norm of the
                # gradient of the 3-D coordinate w.r.t. 2-D coordinates, to
                # avoid taking into account rendered coordinates at the object
                # discontinuities.
                norm_gradient_of_coord = np.sqrt(
                    (np.gradient(rendered_image[..., :3], axis=0)**2 +
                     np.gradient(rendered_image[..., :3], axis=1)**2).sum(
                         axis=-1))
                mask_norm_gradient_of_coord = (norm_gradient_of_coord >
                                               threshold_coordinates_filtering)
                rendered_image[mask_norm_gradient_of_coord] = 0.

            rendered_image[coordinates_to_ignore] = 0.

            # Optionally, convert the coordinates to depth.
            if (render_mode == "depth"):
                dataset_scale = self.neus.nerf.training.dataset.scale
                W_ngp_T_C = nerf_matrix_to_ngp(pose=W_nerf_T_C,
                                               scale=dataset_scale)
                C_coords = (np.linalg.inv(W_ngp_T_C) @ np.hstack([
                    rendered_image[..., :3].reshape(-1, 3),
                    np.ones((len(rendered_image[..., :3].reshape(-1, 3)), 1))
                ]).T).T
                assert (np.all(C_coords[..., 3] == 1.))
                C_coords = C_coords[..., :3]
                uv_unnormalized = (K @ C_coords.T).T
                depth = uv_unnormalized[...,
                                        2].reshape(rendered_image.shape[:2])
                depth[coordinates_to_ignore] = 0.

                # Keep the alpha channel for potential later filtering.
                rendered_image[..., :3] = depth[..., None]

        return rendered_image

    def render_from_given_pose_given_point_cloud(
            self,
            K: np.ndarray,
            W_nerf_T_C: np.ndarray,
            W_nerf_point_cloud: np.ndarray,
            final_crop_res: int,
            render_mode: str,
            dataset_scale: float,
            spp: int = 8,
            min_valid_x: Optional[int] = None,
            max_valid_x: Optional[int] = None,
            min_valid_y: Optional[int] = None,
            max_valid_y: Optional[int] = None,
            crop_scale_factor: float = 1.0):
        r"""Renders an image tightly-cropped to the object and with the desired
        resolution, which is assumed to be the same horizontally and vertically.
        The tight crop is obtained by reprojecting into 2-D a subset of the
        points of the object point cloud. Note: The resolution of the returned
        crop is assumed to be symmetrical vertically and horizontally. The input
        point cloud is assumed to be in the same format and coordinate
        convention as the one returned by `compute_point_cloud`.
        """
        assert (isinstance(K, np.ndarray) and K.shape == (3, 3))
        assert (isinstance(W_nerf_T_C, np.ndarray) and
                W_nerf_T_C.shape == (4, 4))
        assert (isinstance(W_nerf_point_cloud, np.ndarray))
        assert (W_nerf_point_cloud.ndim == 2 and
                W_nerf_point_cloud.shape[-1] == 3)
        assert ((min_valid_x is None) == (max_valid_x is None))
        assert ((min_valid_y is None) == (max_valid_y is None))
        # Reproject point cloud.
        W_ngp_T_C = nerf_matrix_to_ngp(pose=W_nerf_T_C, scale=dataset_scale)

        C_point_cloud = (np.linalg.inv(W_ngp_T_C) @ np.hstack(
            [W_nerf_point_cloud,
             np.ones((len(W_nerf_point_cloud), 1))]).T).T

        assert (np.all(C_point_cloud[..., 3] == 1.))
        C_point_cloud = C_point_cloud[..., :3]
        uv_point_cloud_unnormalized = (K @ C_point_cloud.T).T
        uv_point_cloud = (uv_point_cloud_unnormalized /
                          uv_point_cloud_unnormalized[..., 2][..., None])
        assert (np.all(uv_point_cloud[..., 2] == 1.))
        uv_point_cloud = uv_point_cloud[..., :2]
        # Approximate point cloud points to integer locations.
        uv_point_cloud = uv_point_cloud.astype(int)
        # - Only consider the point cloud points that fall within the visible
        #   image.
        is_point_valid = np.ones(len(uv_point_cloud), dtype=bool)
        if (min_valid_x is not None):
            is_point_valid[np.logical_or(uv_point_cloud[..., 0] < min_valid_x,
                                         uv_point_cloud[..., 0]
                                         > max_valid_x)] = False
        if (min_valid_y is not None):
            is_point_valid[np.logical_or(uv_point_cloud[..., 1] < min_valid_y,
                                         uv_point_cloud[..., 1]
                                         > max_valid_y)] = False
        uv_point_cloud = uv_point_cloud[is_point_valid]

        max_u_point_cloud = uv_point_cloud[..., 0].max()
        min_u_point_cloud = uv_point_cloud[..., 0].min()
        max_v_point_cloud = uv_point_cloud[..., 1].max()
        min_v_point_cloud = uv_point_cloud[..., 1].min()

        (min_u_point_cloud, max_u_point_cloud, min_v_point_cloud,
         max_v_point_cloud) = make_crop_square(x_min=min_u_point_cloud,
                                               x_max=max_u_point_cloud,
                                               y_min=min_v_point_cloud,
                                               y_max=max_v_point_cloud,
                                               min_valid_x=min_valid_x,
                                               max_valid_x=max_valid_x,
                                               min_valid_y=min_valid_y,
                                               max_valid_y=max_valid_y,
                                               scale_factor=crop_scale_factor)

        W_ = max_u_point_cloud - min_u_point_cloud
        H_ = max_v_point_cloud - min_v_point_cloud
        assert (W_ == H_)
        zoom_factor = final_crop_res / W_

        # Adapt intrinsics.
        K_cropped_and_rescaled = K.copy()
        K_cropped_and_rescaled[
            0, 2] = K_cropped_and_rescaled[0, 2] - min_u_point_cloud
        K_cropped_and_rescaled[
            1, 2] = K_cropped_and_rescaled[1, 2] - min_v_point_cloud
        K_cropped_and_rescaled[0] = K_cropped_and_rescaled[0] * zoom_factor
        K_cropped_and_rescaled[1] = K_cropped_and_rescaled[1] * zoom_factor

        # Consider a discontinuity of 0.02 in the scale of NeuS for filtering
        # the coordinates. NOTE: One could also set this threshold in a metric
        # unit, but since the artifacts are related to the raymarching steps and
        # since the objects are pre-normalized to a fixed scale, the metric
        # threshold would be object-specific.
        threshold_coordinates_filtering = 0.02

        return self.render_from_given_pose(
            K=K_cropped_and_rescaled,
            W_nerf_T_C=W_nerf_T_C,
            H=final_crop_res,
            W=final_crop_res,
            render_mode=render_mode,
            spp=spp,
            threshold_coordinates_filtering=threshold_coordinates_filtering), (
                min_u_point_cloud, max_u_point_cloud, min_v_point_cloud,
                max_v_point_cloud), K_cropped_and_rescaled