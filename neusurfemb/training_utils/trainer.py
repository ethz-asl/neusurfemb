import copy
import gc
import json
import numpy as np
import os
import pandas as pd
import pyntcloud
import subprocess
import sys
import time
import torch
import yaml

from neusurfemb.data_utils.dataset import (generate_models_info,
                                           transform_to_bop_info_files)
from neusurfemb.misc_utils.transforms import _W_NEUS_T_W_BOP
from neusurfemb.training_utils.dataset_generation import (
    crop_and_resize_train_dataset, generate_synthetic_dataset)


class Trainer(object):
    """Structure originally based on https://github.com/ashawkey/torch-ngp/blob/
    b6e080468925f0bb44827b4f8f0ed08291dcf8a9/nerf/utils.py#L316.
    """

    def __init__(
            self,
            opt,  # Extra config.
            model,  # Network.
            device=None,  # Device to use.
            workspace='workspace',  # Workspace to save data.
    ):

        self.opt = opt
        self.workspace = workspace
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.device = device if device is not None else torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self._obj_id = opt.obj_id

        self.full_point_cloud = None

        # Prepare the workspace.
        if (self.workspace is not None):
            os.makedirs(self.workspace, exist_ok=True)
            self.ckpt_path = os.path.join(self.workspace, "checkpoints")
            os.makedirs(self.ckpt_path, exist_ok=True)
            self.surfemb_data_folder = os.path.join(self.workspace,
                                                    "data_for_surfemb")
            os.makedirs(self.surfemb_data_folder, exist_ok=True)
            self.surface_samples_folder = os.path.join(self.surfemb_data_folder,
                                                       "surface_samples")
            os.makedirs(self.surface_samples_folder, exist_ok=True)
            self.surface_samples_normals_folder = os.path.join(
                self.surfemb_data_folder, "surface_samples_normals")
            os.makedirs(self.surface_samples_normals_folder, exist_ok=True)
            self.mesh_folder = os.path.join(self.surfemb_data_folder, "models")
            os.makedirs(self.mesh_folder, exist_ok=True)

        self.should_store_git_information = True

        git_save_path = os.path.abspath(os.path.join(self.workspace,
                                                     'git_info'))
        if (self.should_store_git_information):
            # Save the .git information to file.
            os.makedirs(git_save_path, exist_ok=False)
            # - Save commit hash.
            subprocess.run(
                "git rev-parse --verify HEAD > "
                f"{os.path.join(git_save_path, 'commit_hash')}",
                shell=True)
            # - Save uncommitted changes.
            subprocess.run(
                "git diff HEAD > "
                f"{os.path.join(git_save_path, 'uncommitted_changes.patch')}",
                shell=True)
            # - Save untracked changes. From
            #   https://stackoverflow.com/a/35484355.
            subprocess.run(
                "git ls-files --others --exclude-standard -z | xargs -0 -n 1 "
                "git --no-pager diff /dev/null | less > "
                f"{os.path.join(git_save_path, 'untracked_changes.patch')}",
                shell=True)
            # Save command run.
            command_run = ' '.join([sys.executable, *sys.argv])
            with open(os.path.join(self.workspace, "command_run"), "w") as f:
                f.write(command_run)
        else:
            # Check that at least the commit matches.
            with open(os.path.join(git_save_path, 'commit_hash'), "r") as f:
                prev_commit_hash = f.read().strip()
            curr_commit_hash = subprocess.getoutput(
                "git rev-parse --verify HEAD")
            if (prev_commit_hash != curr_commit_hash):
                print("\033[93mNOTE: The current commit hash ("
                      f"'{curr_commit_hash}') does not match the commit hash "
                      "with which the training was started ("
                      f"'{prev_commit_hash}')!\033[0m")

        self._random_bg_fg_creator_real = None
        self._random_bg_fg_creator_synth = None

        self.output_folder_synthetic_dataset = os.path.join(
            self.surfemb_data_folder, "synthetic_dataset",
            f"{self._obj_id:06d}" if self._obj_id is not None else "000001")
        self.converted_train_dataset_folder = os.path.join(
            self.surfemb_data_folder, "converted_train_dataset",
            f"{self._obj_id:06d}" if self._obj_id is not None else "000001")

    def _compute_point_cloud(self, one_uom_scene_to_m):
        # - Original, rendered point cloud, kept for reference.
        full_point_cloud_nondownsampled_path = os.path.join(
            self.workspace, 'full_point_cloud_nondownsampled.ply')
        # - Point cloud downsampled for correspondence training.
        full_point_cloud_path = os.path.join(self.workspace,
                                             'full_point_cloud.ply')
        # - Point cloud downsampled to fixed, low resolution, used to compute
        #   tight object crops when cropping real images and generating a
        #   synthetic dataset, and to compute an object-aligned bounding box, if
        #   required.
        full_point_cloud_downsampled_path = os.path.abspath(
            full_point_cloud_path.split(".ply")[0] + "_downsampled.ply")
        # - Point cloud downsampled for correspondence training, in mm and in
        #   BOP coordinate frame.
        full_point_cloud_mm_bop_path = os.path.join(
            self.surface_samples_folder, f"obj_{self._obj_id:06d}.ply"
            if self._obj_id is not None else "obj_000001.ply")
        # - Normals of the point cloud downsampled for correspondence training,
        #   in BOP coordinate frame.
        normals_bop_path = os.path.join(
            self.surface_samples_normals_folder, f"obj_{self._obj_id:06d}.ply"
            if self._obj_id is not None else "obj_000001.ply")
        # - Mesh in mm and in BOP coordinate frame.
        mesh_path = os.path.join(
            self.mesh_folder, f"obj_{self._obj_id:06d}.ply"
            if self._obj_id is not None else "obj_000001.ply")
        models_info_path = os.path.join(self.mesh_folder, "models_info.json")

        print("\033[94mGenerating/loading point cloud.\033[0m")

        if (os.path.exists(full_point_cloud_path)):
            self.full_point_cloud = np.asarray(
                pyntcloud.io.read_ply(full_point_cloud_path)['points'])[..., :3]
            assert (len(
                self.full_point_cloud
            ) == self.opt.num_points_correspondence_point_cloud), (
                f"The point cloud loaded from '{full_point_cloud_path}' has "
                f"{len(self.full_point_cloud)} points, which does not match "
                "the desired resolution of "
                f"{self.opt.num_points_correspondence_point_cloud} points. Was "
                "the point cloud generated in a different run?")
        else:
            # Compute mesh.
            self.mesh = self.model.compute_mesh()
            # Compute point cloud.
            (self.full_point_cloud, colors_full_pc,
             normals_full_pc) = self.model.compute_point_cloud_from_mesh(
                 mesh=self.mesh)

            # Save the original, non-downsampled point cloud.
            # NOTE: This point cloud is in NeuS frame of reference.
            full_point_cloud_nondownsampled_pynt = pyntcloud.PyntCloud(
                pd.DataFrame(
                    data={
                        "x":
                            self.full_point_cloud[..., 0],
                        "y":
                            self.full_point_cloud[..., 1],
                        "z":
                            self.full_point_cloud[..., 2],
                        "red": (255 * colors_full_pc[..., 0]).astype(np.uint8),
                        "green": (255 *
                                  colors_full_pc[..., 1]).astype(np.uint8),
                        "blue": (255 * colors_full_pc[..., 2]).astype(np.uint8)
                    }))
            full_point_cloud_nondownsampled_pynt.to_file(
                full_point_cloud_nondownsampled_path)

            full_point_cloud_mm_bop = self.full_point_cloud.copy()
            colors_full_pc_bop = colors_full_pc.copy()
            normals_full_pc_bop = normals_full_pc.copy()

            # Downsample the point cloud for later use in correspondence
            # training.
            if (self.opt.num_points_correspondence_point_cloud
                    < len(self.full_point_cloud)):
                permutation = np.random.permutation(
                    np.arange(len(self.full_point_cloud))
                )[:self.opt.num_points_correspondence_point_cloud]
                self.full_point_cloud = self.full_point_cloud[permutation]
                colors_full_pc = colors_full_pc[permutation]
                normals_full_pc = normals_full_pc[permutation]
            else:
                print("\033[93mNOTE: The formed point cloud has "
                      f"{len(self.full_point_cloud)} points, hence it cannot "
                      "be downsampled to have "
                      f"{self.opt.num_points_correspondence_point_cloud} "
                      "points. No downsampling will be applied.\033[0m")
                permutation = np.arange(len(self.full_point_cloud))
            # Save the point cloud used for correspondence.
            full_point_cloud_pynt = pyntcloud.PyntCloud(
                pd.DataFrame(
                    data={
                        "x":
                            self.full_point_cloud[..., 0],
                        "y":
                            self.full_point_cloud[..., 1],
                        "z":
                            self.full_point_cloud[..., 2],
                        "red": (255 * colors_full_pc[..., 0]).astype(np.uint8),
                        "green": (255 *
                                  colors_full_pc[..., 1]).astype(np.uint8),
                        "blue": (255 * colors_full_pc[..., 2]).astype(np.uint8)
                    }))
            full_point_cloud_pynt.to_file(full_point_cloud_path)

            # Save the point cloud in mm and in BOP coordinate frame.
            # - Transform the coordinate system and scale.
            _W_bop_T_W_neus = np.linalg.inv(_W_NEUS_T_W_BOP)
            full_point_cloud_mm_bop = ((_W_bop_T_W_neus @ np.hstack([
                full_point_cloud_mm_bop,
                np.ones_like(full_point_cloud_mm_bop[..., 0][..., None])
            ]).T).T[..., :3] * one_uom_scene_to_m * 1000.)
            normals_full_pc_bop = (_W_bop_T_W_neus @ np.hstack([
                normals_full_pc_bop,
                np.ones_like(normals_full_pc_bop[..., 0][..., None])
            ]).T).T[..., :3]
            # - Save mesh in mm and in BOP coordinate frame, at full resolution.
            mesh_mm_bop = copy.deepcopy(self.mesh)
            mesh_mm_bop.vertices = full_point_cloud_mm_bop
            mesh_mm_bop.vertex_normals = normals_full_pc_bop
            _ = mesh_mm_bop.export(mesh_path)

            # Generate the BOP `models_info.json` file.
            models_info = generate_models_info(
                mesh_path=mesh_path,
                obj_id=self._obj_id if self._obj_id is not None else 1)
            with open(models_info_path, "w") as f:
                json.dump(obj=models_info, fp=f, indent=2)

            # - Save point cloud and normals at the resolution requested for
            #   correspondence training.
            full_point_cloud_mm_bop = full_point_cloud_mm_bop[permutation]
            colors_full_pc_bop = colors_full_pc_bop[permutation]
            normals_full_pc_bop = normals_full_pc_bop[permutation]

            full_point_cloud_mm_bop_pynt = pyntcloud.PyntCloud(
                pd.DataFrame(
                    data={
                        "x":
                            full_point_cloud_mm_bop[..., 0],
                        "y":
                            full_point_cloud_mm_bop[..., 1],
                        "z":
                            full_point_cloud_mm_bop[..., 2],
                        "red": (255 *
                                colors_full_pc_bop[..., 0]).astype(np.uint8),
                        "green": (255 *
                                  colors_full_pc_bop[..., 1]).astype(np.uint8),
                        "blue": (255 *
                                 colors_full_pc_bop[..., 2]).astype(np.uint8)
                    }))
            full_point_cloud_mm_bop_pynt.to_file(full_point_cloud_mm_bop_path)

            normals_full_pc_bop_pynt = pyntcloud.PyntCloud(
                pd.DataFrame(
                    data={
                        "x":
                            normals_full_pc_bop[..., 0],
                        "y":
                            normals_full_pc_bop[..., 1],
                        "z":
                            normals_full_pc_bop[..., 2],
                        "red": (255 * colors_full_pc[..., 0]).astype(np.uint8),
                        "green": (255 *
                                  colors_full_pc[..., 1]).astype(np.uint8),
                        "blue": (255 * colors_full_pc[..., 2]).astype(np.uint8)
                    }))
            normals_full_pc_bop_pynt.to_file(normals_bop_path)

        if (self.opt.synthetic_data_config_path is not None):
            assert (self.opt.compute_oriented_bounding_box is not None), (
                "To generate synthetic data, please specify whether to compute "
                "an oriented bounding box around the object (flag "
                "`compute_oriented_bounding_box`).")

        # - Subsample the full point cloud to a resolution of
        #   `res_downsampled_pc` points.
        res_downsampled_pc = 10000

        if (os.path.exists(full_point_cloud_downsampled_path)):
            self.downsampled_point_cloud = np.asarray(
                pyntcloud.io.read_ply(full_point_cloud_downsampled_path)
                ['points'])[..., :3]
            assert (len(self.downsampled_point_cloud) == res_downsampled_pc), (
                "The downsampled point cloud loaded from "
                f"'{full_point_cloud_downsampled_path}' has "
                f"{len(self.downsampled_point_cloud)} points, which does not "
                f"match the desired resolution of {res_downsampled_pc} points.")
        else:
            permutation = np.random.permutation(
                np.arange(len(self.full_point_cloud)))[:10000]
            self.downsampled_point_cloud = self.full_point_cloud[permutation]
            colors_downsampled_point_cloud = colors_full_pc[permutation]
            full_pc_downsampled_pynt = pyntcloud.PyntCloud(
                pd.DataFrame(
                    data={
                        "x":
                            self.downsampled_point_cloud[..., 0],
                        "y":
                            self.downsampled_point_cloud[..., 1],
                        "z":
                            self.downsampled_point_cloud[..., 2],
                        "red": (255 * colors_downsampled_point_cloud[..., 0]
                               ).astype(np.uint8),
                        "green": (255 * colors_downsampled_point_cloud[..., 1]
                                 ).astype(np.uint8),
                        "blue": (255 * colors_downsampled_point_cloud[..., 2]
                                ).astype(np.uint8)
                    }))
            full_pc_downsampled_pynt.to_file(full_point_cloud_downsampled_path)

        if (self.opt.compute_oriented_bounding_box is True):
            # Extract oriented bounding box from the point cloud. This is done
            # because when rendering novel views to generate an augmented
            # dataset, ideally the global coordinate frame should be well
            # aligned with the object, so that the novel viewpoint do not
            # obliquely look at the object (cf. `generate_synthetic_dataset`).
            oriented_bbox_path = os.path.abspath(
                full_point_cloud_downsampled_path.split(".ply")[0] +
                "_bbox.txt")
            if (os.path.exists(oriented_bbox_path)):
                print("\033[94mReading the oriented bounding box "
                      f"for the object from '{oriented_bbox_path}'."
                      "\033[0m")
            else:
                print("\033[94mComputing an oriented bounding box "
                      "for the object from a downsampled version of "
                      f"the point cloud '{full_point_cloud_path}'."
                      "\033[0m")
                # - Use the CGAL-based `estimate_bbox` script to estimate
                #   oriented bounding box from the downsampled point cloud.
                #   Downsampling improves robustness and generates more tight
                #   bounding boxes. The result is stored to disk.
                curr_file_path = os.path.dirname(os.path.realpath(__file__))
                script_path = os.path.abspath(
                    os.path.join(curr_file_path,
                                 '../../bbox_estimator/build/estimate_bbox'))
                os.system(f"{script_path} "
                          f"{full_point_cloud_downsampled_path}")
            self.oriented_bbox = np.loadtxt(oriented_bbox_path)
            assert (len(self.oriented_bbox) == 8)
            # - Find the transformation from the frame of the oriented bounding
            #   box to the original world frame (which is in NeRF format, as
            #   explained above).
            R_x = self.oriented_bbox[0] - self.oriented_bbox[1]
            R_y = self.oriented_bbox[1] - self.oriented_bbox[2]
            R_z = self.oriented_bbox[5] - self.oriented_bbox[0]
            R_x = R_x / np.linalg.norm(R_x)
            R_y = R_y / np.linalg.norm(R_y)
            R_z = R_z / np.linalg.norm(R_z)
            orig_frame_R_oriented_bbox_frame = np.vstack([R_x, R_y, R_z]).T
            # Center of the bounding box.
            orig_frame_t_oriented_bbox_frame = np.sum(self.oriented_bbox,
                                                      axis=0) / 8.
            self.orig_frame_T_oriented_bbox_frame = np.eye(4)
            self.orig_frame_T_oriented_bbox_frame[:3, :3] = (
                orig_frame_R_oriented_bbox_frame)
            self.orig_frame_T_oriented_bbox_frame[:3, 3] = (
                orig_frame_t_oriented_bbox_frame)
            if (self.opt.visualize_oriented_bounding_box):
                # Optionally visualize the oriented bounding box, the point
                # cloud, the original coordinate frame, and the coordinate frame
                # of the oriented bounding box.
                import open3d as o3d
                lines = o3d.geometry.LineSet()
                lines.points = o3d.utility.Vector3dVector(self.oriented_bbox)
                lines.lines = o3d.utility.Vector2iVector([[0, 1], [1,
                                                                   2], [2, 3],
                                                          [3, 0], [0,
                                                                   5], [5, 6],
                                                          [6, 7], [6,
                                                                   1], [7, 2],
                                                          [3, 4], [4, 5],
                                                          [4, 7]])
                pc = o3d.io.read_point_cloud(full_point_cloud_path)
                coord_frame = (
                    o3d.geometry.TriangleMesh.create_coordinate_frame())

                oriented_bbox_frame = (
                    o3d.geometry.TriangleMesh.create_coordinate_frame(
                        origin=orig_frame_t_oriented_bbox_frame))
                oriented_bbox_frame.rotate(
                    orig_frame_R_oriented_bbox_frame,
                    center=orig_frame_t_oriented_bbox_frame)

                o3d.visualization.draw(
                    [lines, pc, coord_frame, oriented_bbox_frame])
        elif (self.opt.compute_oriented_bounding_box is False):
            self.orig_frame_T_oriented_bbox_frame = np.eye(4)
        else:
            assert (self.opt.compute_oriented_bounding_box is None)
            self.orig_frame_T_oriented_bbox_frame = None

    @property
    def _object_bounding_box(self):
        assert (self.full_point_cloud
                is not None), "Need to compute the object point cloud first."
        if (hasattr(self, "oriented_bbox")):
            # Use the oriented bounding box if it was computed. Cf.
            # visualization option (`visualize_oriented_bounding_box`) for the
            # meaning of the computed corners, which here is made consistent
            # with the case in which no oriented bounding box was computed.
            # However, this is not strictly necessary as long as the bounding
            # box is used simply for reprojection purposes, since obtaining a
            # 2-D bounding box from the reprojection is invariant to the order
            # of the corners.
            assert (self.oriented_bbox.shape == (8, 3))
            corners = np.array([
                self.oriented_bbox[2], self.oriented_bbox[7],
                self.oriented_bbox[1], self.oriented_bbox[6],
                self.oriented_bbox[3], self.oriented_bbox[4],
                self.oriented_bbox[0], self.oriented_bbox[5]
            ])
        else:
            max_corner = self.full_point_cloud.max(axis=0)
            min_corner = self.full_point_cloud.min(axis=0)

            corners = np.array([[min_corner[0], min_corner[1], min_corner[2]],
                                [min_corner[0], min_corner[1], max_corner[2]],
                                [min_corner[0], max_corner[1], min_corner[2]],
                                [min_corner[0], max_corner[1], max_corner[2]],
                                [max_corner[0], min_corner[1], min_corner[2]],
                                [max_corner[0], min_corner[1], max_corner[2]],
                                [max_corner[0], max_corner[1], min_corner[2]],
                                [max_corner[0], max_corner[1], max_corner[2]]])

        return corners

    def train(self, train_transform, train_transform_path):
        # Pre-train NeuS.
        self.model._train_neus(path_to_scene_json=train_transform_path,
                               num_steps=self.opt.num_iters,
                               checkpoint_folder=self.ckpt_path)

        # If necessary, generate the key point cloud.
        if (self.full_point_cloud is None):
            with torch.no_grad():
                self._compute_point_cloud(
                    one_uom_scene_to_m=train_transform['one_uom_scene_to_m'])

        # Generate synthetic dataset if necessary.
        if (self.opt.synthetic_data_config_path is not None):
            assert (os.path.exists(self.opt.synthetic_data_config_path))
            with open(self.opt.synthetic_data_config_path, "r") as f:
                synthetic_data_config = yaml.load(f, Loader=yaml.Loader)
            print("\033[94mGenerating/reading augmented dataset. The "
                  "samples are stored in "
                  f"'{self.output_folder_synthetic_dataset}'.\033[0m")
            try:
                fx_syn_dataset = train_transform['fl_x']
                fy_syn_dataset = train_transform['fl_y']
                cx_syn_dataset = train_transform['cx']
                cy_syn_dataset = train_transform['cy']
                h_syn_dataset = train_transform['h']
                w_syn_dataset = train_transform['w']
            except KeyError:
                # Check that all frames have the same intrinsics parameters. It
                # should not be a requirement, but for now we require it.
                fx_syn_dataset = None
                fy_syn_dataset = None
                cx_syn_dataset = None
                cy_syn_dataset = None
                h_syn_dataset = None
                w_syn_dataset = None
                for frame in train_transform['frames']:
                    curr_frame_fx = frame['fl_x']
                    curr_frame_fy = frame['fl_y']
                    curr_frame_cx = frame['cx']
                    curr_frame_cy = frame['cy']
                    curr_frame_h = frame['h']
                    curr_frame_w = frame['w']
                    if (fx_syn_dataset is not None):
                        assert (curr_frame_fx == fx_syn_dataset)
                        assert (curr_frame_fy == fy_syn_dataset)
                        assert (curr_frame_cx == cx_syn_dataset)
                        assert (curr_frame_cy == cy_syn_dataset)
                        assert (curr_frame_h == h_syn_dataset)
                        assert (curr_frame_w == w_syn_dataset)
                    else:
                        fx_syn_dataset = curr_frame_fx
                        fy_syn_dataset = curr_frame_fy
                        cx_syn_dataset = curr_frame_cx
                        cy_syn_dataset = curr_frame_cy
                        h_syn_dataset = curr_frame_h
                        w_syn_dataset = curr_frame_w
            transform_syn_file_path = generate_synthetic_dataset(
                model=self.model,
                synthetic_data_config=synthetic_data_config,
                intrinsics=np.array([
                    fx_syn_dataset, fy_syn_dataset, cx_syn_dataset,
                    cy_syn_dataset
                ]),
                H=h_syn_dataset,
                W=w_syn_dataset,
                output_folder=self.output_folder_synthetic_dataset,
                one_uom_scene_to_m=train_transform['one_uom_scene_to_m'],
                W_nerf_point_cloud=self.downsampled_point_cloud,
                orig_frame_T_oriented_bbox_frame=(
                    self.orig_frame_T_oriented_bbox_frame),
                obj_id=self._obj_id,
                in_mask_threshold=0.8)

            torch.cuda.empty_cache()
            gc.collect()

            # Save a YAML config file for the synthetic dataset.
            cfg_synthetic = {
                'model_folder': 'models',
                'train_folder': 'synthetic_dataset',
                'test_folder': None,
                'img_folder': 'images',
                'coordinate_folder': 'coordinates',
                'depth_folder': None,
                'img_ext': 'png',
                'coordinate_ext': 'npy',
                'depth_ext': None
            }
            with open(
                    os.path.join(self.surfemb_data_folder, "cfg_synthetic.yml"),
                    "w") as f:
                yaml.dump(cfg_synthetic, f)
            # Save the BOP scene files.
            transform_to_bop_info_files(
                transform_file_path=transform_syn_file_path,
                obj_id=self._obj_id if self._obj_id is not None else 1,
                output_folder=self.output_folder_synthetic_dataset)

        if (self.opt.crop_res_train_dataset is not None):
            # Convert training dataset to a cropped version with coordinate maps
            # from the key point cloud.
            converted_train_transform_path = crop_and_resize_train_dataset(
                model=self.model,
                crop_res=self.opt.crop_res_train_dataset,
                crop_scale_factor=self.opt.crop_scale_factor_train_dataset,
                one_uom_scene_to_m=train_transform['one_uom_scene_to_m'],
                W_nerf_point_cloud=self.downsampled_point_cloud,
                path_to_scene_json=train_transform_path,
                output_folder_path=self.converted_train_dataset_folder)

            # Save a YAML config file for the converted dataset.
            cfg_converted_train = {
                'model_folder': 'models',
                'train_folder': 'converted_train_dataset',
                'test_folder': None,
                'img_folder': 'images',
                'coordinate_folder': 'coordinates',
                'depth_folder': None,
                'img_ext': 'png',
                'coordinate_ext': 'npy',
                'depth_ext': None
            }
            with open(
                    os.path.join(self.surfemb_data_folder,
                                 "cfg_converted_train.yml"), "w") as f:
                yaml.dump(cfg_converted_train, f)
            # Save the BOP scene files.
            transform_to_bop_info_files(
                transform_file_path=converted_train_transform_path,
                obj_id=self._obj_id if self._obj_id is not None else 1,
                output_folder=self.converted_train_dataset_folder)
