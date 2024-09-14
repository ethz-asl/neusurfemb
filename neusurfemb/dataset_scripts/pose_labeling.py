# Based on https://github.com/ethz-asl/autolabel/blob/main/scripts/mapping.py,
# https://github.com/ethz-asl/autolabel/blob/main/autolabel/undistort.py and
# https://github.com/ethz-asl/autolabel/blob/main/autolabel/utils/__init__.py.
import argparse
import copy
import cv2
import numpy as np
import open3d as o3d
import os
from pathlib import Path
import pickle as pkl
import pycolmap

from hloc import (extract_features, match_features, reconstruction,
                  pairs_from_exhaustive, pairs_from_retrieval)
from hloc.utils import viz_3d


def transform_points(T, points):
    R = T[:3, :3]
    t = T[:3, 3]
    return (R @ points[..., :, None])[..., :, 0] + t


class Scene:

    def __init__(self, scene_path):
        self.path = scene_path
        self.raw_rgb_path = os.path.join(scene_path, 'raw_rgb')
        self.raw_depth_path = os.path.join(scene_path, 'raw_depth')
        self.depth_path = os.path.join(scene_path, 'depth')

        self.poses = []

    def __iter__(self):
        rgb_frames = self.rgb_paths()
        depth_frames = self.depth_paths()
        for rgb, depth in zip(rgb_frames, depth_frames):
            yield (rgb, depth)

    def _get_paths(self, directory):
        try:
            frames = os.listdir(directory)
            frames = [
                frame for frame in frames
                if os.path.isfile(os.path.join(directory, frame))
            ]
            frames = sorted(frames, key=lambda x: int(x.split('.')[0]))
            return [os.path.join(directory, f) for f in frames]
        except FileNotFoundError:
            return []

    def raw_rgb_paths(self):
        return self._get_paths(self.raw_rgb_path)

    def raw_depth_paths(self):
        return self._get_paths(self.raw_depth_path)

    def depth_paths(self):
        return self._get_paths(self.depth_path)

    @property
    def rgb_size(self):
        if (hasattr(self, "_rgb_size")):
            return self._rgb_size
        else:
            rgb_paths = self.raw_rgb_paths()
            image = cv2.imread(rgb_paths[0], -1)
            # (W, H).
            self._rgb_size = (image.shape[1], image.shape[0])
            return self._rgb_size

    @property
    def depth_size(self):
        """
        Return: the size (width, height) of the depth images.
        """
        if (not self.has_depth):
            return None

        if (hasattr(self, "_depth_size")):
            return self._depth_size
        else:
            depth_paths = self.raw_depth_paths()
            image = cv2.imread(depth_paths[0], -1)
            # (W, H).
            self._depth_size = (image.shape[1], image.shape[0])
            return self._depth_size

    @property
    def has_depth(self):
        return len(self.raw_depth_paths()) > 0

    def create_camera(self, intrinsics, H, W):
        assert (not hasattr(self, "camera"))
        self.camera = Camera(camera_matrix=intrinsics, size=(W, H))


class Camera:

    def __init__(self, camera_matrix, size):
        self.camera_matrix = camera_matrix
        self.size = size

    def scale(self, new_size):
        scale_x = new_size[0] / self.size[0]
        scale_y = new_size[1] / self.size[1]
        camera_matrix = self.camera_matrix.copy()
        camera_matrix[0, :] = scale_x * self.camera_matrix[0, :]
        camera_matrix[1, :] = scale_y * self.camera_matrix[1, :]
        return Camera(camera_matrix, new_size)

    @property
    def fx(self):
        return self.camera_matrix[0, 0]

    @property
    def fy(self):
        return self.camera_matrix[1, 1]

    @property
    def cx(self):
        return self.camera_matrix[0, 2]

    @property
    def cy(self):
        return self.camera_matrix[1, 2]

    def write(self, path):
        np.savetxt(path, self.camera_matrix)


class ImageUndistorter:
    """Undistorts images

    Args:
        K (np.ndarray): Intrinsics matrix. Shape: (3, 3).
        D (np.ndarray): Distortion coefficients (k_1, k_2, p_1, p_2) from the
            "OPENCV" COLMAP model or (k1., k2, k3, k4) from the "OPENCV_FISHEYE"
            COLMAP model. Shape: (4, ).
        H (int): Image height.
        W (int): Image width.
    """

    def __init__(self, K: np.ndarray, D: np.ndarray, H: int, W: int,
                 fisheye: bool):
        self._K = K
        self._D = D
        if (fisheye):
            self._D = np.array([D[0], D[1], 0., 0., D[2], D[3], 0., 0.])
        self._H = H
        self._W = W

        self._compute_source_to_target_mapping()

    def _compute_source_to_target_mapping(self) -> None:
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self._K, self._D, np.eye(3), self._K, (self._W, self._H),
            cv2.CV_32FC2)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, self.map1, self.map2, cv2.INTER_NEAREST)


class HLoc:

    def __init__(self, tmp_dir, scene, flags):
        self.flags = flags
        self.scene = scene
        self.scene_path = Path(self.scene.path)
        # Run exhaustive matching if less than 250 frames are provided.
        self.exhaustive = (len((self.scene.raw_rgb_paths())) < 250) or True

        self.tmp_dir = Path(tmp_dir)
        self.sfm_pairs = self.tmp_dir / 'sfm-pairs.txt'
        self.loc_pairs = self.tmp_dir / 'sfm-pairs-loc.txt'
        self.features = self.tmp_dir / 'features.h5'
        self.matches = self.tmp_dir / 'matches.h5'
        self.feature_conf = extract_features.confs['superpoint_aachen']
        self.retrieval_conf = extract_features.confs['netvlad']
        self.matcher_conf = match_features.confs['superglue']

    def _run_sfm(self):
        image_dir = Path(self.scene.path) / 'raw_rgb'
        image_list = []
        image_paths = self.scene.raw_rgb_paths()

        image_list_path = []
        indices = np.arange(len(image_paths))
        for index in indices:
            image_list.append(image_paths[index])
            image_list_path.append(
                str(Path(image_paths[index]).relative_to(image_dir)))
        if (self.exhaustive):
            extract_features.main(self.feature_conf,
                                  image_dir,
                                  feature_path=self.features,
                                  image_list=image_list_path)

            pairs_from_exhaustive.main(self.sfm_pairs,
                                       image_list=image_list_path)
            match_features.main(self.matcher_conf,
                                self.sfm_pairs,
                                features=self.features,
                                matches=self.matches)
            self.model = reconstruction.main(
                self.tmp_dir,
                image_dir,
                self.sfm_pairs,
                self.features,
                self.matches,
                image_list=image_list_path,
                camera_mode=pycolmap.CameraMode.SINGLE,
                image_options={
                    'camera_model':
                    "OPENCV_FISHEYE" if self.flags.fisheye else "OPENCV",
                },
                mapper_options={
                    'ba_refine_principal_point': True,
                    'ba_refine_extra_params': True,
                    'ba_refine_focal_length': True,
                    'ba_local_num_images': 15,
                },
                verbose=True)
        else:
            retrieval_path = extract_features.main(self.retrieval_conf,
                                                   image_dir,
                                                   self.tmp_dir,
                                                   image_list=image_list_path)
            pairs_from_retrieval.main(retrieval_path,
                                      self.sfm_pairs,
                                      num_matched=50)
            feature_path = extract_features.main(self.feature_conf,
                                                 image_dir,
                                                 self.tmp_dir,
                                                 image_list=image_list_path)
            match_path = match_features.main(self.matcher_conf,
                                             self.sfm_pairs,
                                             self.feature_conf['output'],
                                             self.tmp_dir,
                                             matches=self.matches)
            self.model = reconstruction.main(
                self.tmp_dir,
                image_dir,
                self.sfm_pairs,
                feature_path,
                match_path,
                image_list=image_list_path,
                camera_mode=pycolmap.CameraMode.SINGLE,
                image_options={
                    'camera_model':
                    "OPENCV_FISHEYE" if self.flags.fisheye else "OPENCV"
                },
                mapper_options={
                    'ba_refine_principal_point': True,
                    'ba_refine_extra_params': True,
                    'ba_refine_focal_length': True
                },
                verbose=True)

        if (self.flags.vis):
            fig = viz_3d.init_figure()
            viz_3d.plot_reconstruction(fig,
                                       self.model,
                                       color='rgba(255,0,0,0.5)',
                                       name="mapping")
            fig.show()

        # Save mapping metadata.
        colmap_output_dir = os.path.join(self.scene.path, 'colmap_output')
        os.makedirs(colmap_output_dir, exist_ok=True)
        self.model.write_text(colmap_output_dir)

        # Save the intrinsics matrix and the distortion parameters.
        assert (len(self.model.cameras) == 1 and 1 in self.model.cameras)
        (focal_length_x, focal_length_y, c_x, c_y, k_1, k_2, p_1,
         p_2) = self.model.cameras[1].params
        self.colmap_K = np.eye(3)
        self.colmap_K[0, 0] = focal_length_x
        self.colmap_K[1, 1] = focal_length_y
        self.colmap_K[0, 2] = c_x
        self.colmap_K[1, 2] = c_y
        self.colmap_distortion_params = np.array([k_1, k_2, p_1, p_2])
        np.savetxt(fname=os.path.join(self.scene.path, 'intrinsics.txt'),
                   X=self.colmap_K)
        np.savetxt(fname=os.path.join(self.scene.path,
                                      'distortion_parameters.txt'),
                   X=self.colmap_distortion_params)

        W, H = self.scene.rgb_size
        self.scene.create_camera(intrinsics=self.colmap_K, H=H, W=W)

    def _undistort_images(self):
        print("Undistorting images according to the estimated intrinsics...")
        undistorted_image_folder = os.path.join(self.scene.path, "rgb")
        os.makedirs(undistorted_image_folder, exist_ok=True)
        color_undistorter = ImageUndistorter(K=self.colmap_K,
                                             D=self.colmap_distortion_params,
                                             H=self.scene.camera.size[1],
                                             W=self.scene.camera.size[0],
                                             fisheye=self.flags.fisheye)
        if (self.scene.has_depth):
            undistorted_depth_folder = os.path.join(self.scene.path, "depth")
            os.makedirs(undistorted_depth_folder, exist_ok=True)

            depth_camera = Camera(self.colmap_K, self.scene.camera.size).scale(
                self.scene.depth_size)
            depth_undistorter = ImageUndistorter(
                K=depth_camera.camera_matrix,
                D=self.colmap_distortion_params,
                H=depth_camera.size[1],
                W=depth_camera.size[0],
                fisheye=self.flags.fisheye)

        # Undistort all the images and save the undistorted versions.
        image_paths = self.scene.raw_rgb_paths()
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

            undistorted_image = color_undistorter.undistort_image(image=image)
            cv2.imwrite(img=undistorted_image,
                        filename=os.path.join(undistorted_image_folder,
                                              os.path.basename(image_path)))

        if (self.scene.has_depth):
            depth_paths = self.scene.raw_depth_paths()
            for depth_path in depth_paths:
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

                undistorted_depth = depth_undistorter.undistort_image(
                    image=depth)
                cv2.imwrite(img=undistorted_depth,
                            filename=os.path.join(
                                undistorted_depth_folder,
                                os.path.basename(depth_path)))

    def run(self):
        self._run_sfm()
        self._undistort_images()


class ScaleEstimation:
    min_depth = 0.05

    def __init__(self, scene, colmap_dir):
        self.scene = scene
        self.colmap_dir = colmap_dir
        self.reconstruction = pycolmap.Reconstruction(colmap_dir)
        self._read_trajectory()
        if (self.scene.has_depth):
            self._read_depth_maps()

    def _read_depth_maps(self):
        self.depth_maps = {}
        for path in self.scene.depth_paths():
            frame_name = os.path.basename(path).split('.')[0]
            self.depth_maps[frame_name] = cv2.imread(path, -1) / 1000.0
        depth_shape = next(iter(self.depth_maps.values())).shape
        depth_size = np.array([depth_shape[1], depth_shape[0]],
                              dtype=np.float64)
        self.depth_to_color_ratio = depth_size / np.array(
            self.scene.camera.size, dtype=np.float64)

    def _read_trajectory(self):
        poses = []
        for image in self.reconstruction.images.values():
            T_CW = np.eye(4)
            T_CW[:3, :3] = image.rotmat()
            T_CW[:3, 3] = image.tvec
            frame_name = image.name.split('.')[0]
            poses.append((frame_name, T_CW))
        self.poses = dict(poses)

    def _lookup_depth(self, frame, xy):
        xy_depth = np.floor(self.depth_to_color_ratio * xy).astype(int)
        return self.depth_maps[frame][xy_depth[1], xy_depth[0]]

    def _estimate_scale(self):
        images = self.reconstruction.images
        point_depths = []
        measured_depths = []
        for image in images.values():
            frame_name = image.name.split('.')[0]
            points = image.get_valid_points2D()
            points3D = self.reconstruction.points3D
            for point in points:
                depth_map_value = self._lookup_depth(frame_name, point.xy)

                if (depth_map_value < self.min_depth):
                    continue

                T_CW = self.poses[frame_name]
                point3D = points3D[point.point3D_id]

                p_C = transform_points(T_CW, point3D.xyz)
                measured_depths.append(depth_map_value)
                point_depths.append(p_C[2])

        point_depths = np.stack(point_depths)
        measured_depths = np.stack(measured_depths)
        scales = measured_depths / point_depths
        return self._ransac(scales)

    def _ransac(self, scales):
        best_set = None
        best_inlier_count = 0
        indices = np.arange(0, scales.shape[0])
        inlier_threshold = np.median(scales) * 1e-2
        for i in range(10000):
            selected = np.random.choice(indices)
            estimate = scales[selected]
            inliers = np.abs(scales - estimate) < inlier_threshold
            inlier_count = inliers.sum()
            if (inlier_count > best_inlier_count):
                best_set = scales[inliers]
                best_inlier_count = inlier_count
        print(f"Scale estimation inlier count: {best_inlier_count} / "
              f"{scales.size}")
        return np.mean(best_set)

    def _scale_poses(self, ratio):
        scaled_poses = {}
        for key, pose in self.poses.items():
            new_pose = pose.copy()
            new_pose[:3, 3] *= ratio
            scaled_poses[key] = new_pose
        return scaled_poses

    def run(self):
        if (self.scene.has_depth):
            scale_ratio = self._estimate_scale()
            return self._scale_poses(scale_ratio)
        else:
            return self.poses


class PoseSaver:

    def __init__(self, scene, scaled_poses):
        self.scene = scene
        self.poses = scaled_poses

    def compute_bbox(self, poses):
        """
        poses: Metrically scaled transforms from camera to world frame.
        """
        # Compute axis-aligned bounding box of the depth values in world frame.
        # Then get the center.
        min_bounds = np.zeros(3)
        max_bounds = np.zeros(3)
        pc = o3d.geometry.PointCloud()
        depth_frame = o3d.io.read_image(self.scene.depth_paths()[0])
        depth_size = np.asarray(depth_frame).shape[::-1]
        K = self.scene.camera.scale(depth_size).camera_matrix
        intrinsics = o3d.camera.PinholeCameraIntrinsic(int(depth_size[0]),
                                                       int(depth_size[1]),
                                                       K[0, 0], K[1, 1],
                                                       K[0, 2], K[1, 2])
        pc = o3d.geometry.PointCloud()
        depth_frames = dict([(os.path.basename(p).split('.')[0], p)
                             for p in self.scene.depth_paths()])
        for key, T_WC in poses.items():
            depth = o3d.io.read_image(f"{depth_frames[key]}")

            pc_C = o3d.geometry.PointCloud.create_from_depth_image(
                depth, depth_scale=1000.0, intrinsic=intrinsics)
            pc_C = np.asarray(pc_C.points)
            pc_W = transform_points(T_WC, pc_C)

            min_bounds = np.minimum(min_bounds, pc_W.min(axis=0))
            max_bounds = np.maximum(max_bounds, pc_W.max(axis=0))
            pc += o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(pc_W)).uniform_down_sample(50)

        filtered, _ = pc.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
        bbox = filtered.get_oriented_bounding_box(robust=True)
        T = np.eye(4)
        T[:3, :3] = bbox.R.T
        o3d_aabb = o3d.geometry.PointCloud(filtered).transform(
            T).get_axis_aligned_bounding_box()
        center = o3d_aabb.get_center()
        T[:3, 3] = -center
        aabb = np.zeros((2, 3))
        aabb[0, :] = o3d_aabb.get_min_bound() - center
        aabb[1, :] = o3d_aabb.get_max_bound() - center
        return T, aabb, filtered

    def _write_poses(self, poses):
        pose_dir = os.path.join(self.scene.path, 'pose')
        os.makedirs(pose_dir, exist_ok=True)
        for key, T_CW in poses.items():
            pose_file = os.path.join(pose_dir, f'{key}.txt')
            np.savetxt(pose_file, T_CW)

    def _write_bounds(self, bounds):
        with open(os.path.join(self.scene.path, 'bbox.txt'), 'wt') as f:
            min_str = " ".join([str(x) for x in bounds[0]])
            max_str = " ".join([str(x) for x in bounds[1]])
            f.write(f"{min_str} {max_str} 0.01")

    def run(self):
        T_WCs = {}
        for key, T_CW in self.poses.items():
            T_WCs[key] = np.linalg.inv(T_CW)
        if (self.scene.has_depth):
            T, aabb, _ = self.compute_bbox(T_WCs)
        else:
            T = np.eye(4)

        T_CWs = {}
        for key, T_WC in T_WCs.items():
            T_CWs[key] = np.linalg.inv(T @ T_WC)
        self._write_poses(T_CWs)
        if (self.scene.has_depth):
            self._write_bounds(aabb)


class Pipeline:

    def __init__(self, flags):
        self.tmp_dir = os.path.join(flags.scene, "hloc")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.flags = flags
        self.scene = Scene(flags.scene)

    def run(self):
        self.hloc = HLoc(self.tmp_dir, self.scene, self.flags)
        self.hloc.run()
        scale_estimation = ScaleEstimation(self.scene, self.tmp_dir)
        scaled_poses = scale_estimation.run()
        # Save the matches.
        self.save_matches()
        # Save the poses.
        pose_saver = PoseSaver(self.scene, scaled_poses)
        pose_saver.run()

    def save_matches(self):
        # Store correspondences between 2-D pixels and 3-D points.
        image_to_points_3D = {}
        all_points3D = {
            point3D_id: point3D.xyz.tolist()
            for point3D_id, point3D in self.hloc.model.points3D.items()
        }

        for image in self.hloc.model.images.values():
            pixel_to_point3D_id = {}
            point3D_id_to_pixel = {}
            for point in image.points2D:
                if (point.has_point3D()):
                    pixel = tuple(point.xy.astype(int))
                    pixel_to_point3D_id[pixel] = point.point3D_id
                    if (point.point3D_id in point3D_id_to_pixel):
                        point3D_id_to_pixel[point.point3D_id].append(pixel)
                    else:
                        point3D_id_to_pixel[point.point3D_id] = [pixel]
            image_to_points_3D[image.name] = copy.deepcopy({
                'pixel_to_point3D_id':
                pixel_to_point3D_id,
                'point3D_id_to_pixel':
                point3D_id_to_pixel
            })
        # Save to disk.
        with open(os.path.join(self.tmp_dir, "image_to_points_3D.pkl"),
                  "wb") as f:
            pkl.dump(image_to_points_3D, f)

        with open(os.path.join(self.tmp_dir, "all_points3D.pkl"), "wb") as f:
            pkl.dump(all_points3D, f)


parser = argparse.ArgumentParser()
parser.add_argument('scene', help="Scene to infer poses for.")
parser.add_argument('--vis', action='store_true')
parser.add_argument('--fisheye', action='store_true')
args = parser.parse_args()

pipeline = Pipeline(args)
pipeline.run()
