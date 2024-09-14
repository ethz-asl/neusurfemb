"""The following file defines a class to compute the transform between two point
clouds of the same object. This can be used to register the coordinate frames of
two datasets representing the same object (cf.
`data_utils/dataset_transforms.py`) or to find symmetry transformations for an
object (`data_utils/find_symmetries.py`). An initial PnP-based estimation that
relies on user-defined correspondences is followed by an ICP-based local
refinement.
"""
import copy
import numpy as np
import open3d as o3d


class ICPRegistrator:

    def __init__(self):
        pass

    def _draw_registration_result(self, point_cloud_source, point_cloud_target,
                                  frame_target_T_frame_source):
        # Adapted from http://www.open3d.org/docs/release/tutorial/
        # visualization/interactive_visualization.html.
        source_temp = copy.deepcopy(point_cloud_source)
        target_temp = copy.deepcopy(point_cloud_target)
        # Use uniform colors to separately highlight the point clouds.
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(frame_target_T_frame_source)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    def _pick_points(self, point_cloud):
        # From http://www.open3d.org/docs/release/tutorial/visualization/
        # interactive_visualization.html.
        print("\n1) Please pick at least three correspondences using [shift "
              "+ left click].")
        print("   Press [shift + right click] to undo point picking.")
        print("2) After picking points, press 'Q' to close the window.")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(point_cloud)
        vis.run()  # User picks points.
        vis.destroy_window()

        return vis.get_picked_points()

    def _compute_diagonal_point_cloud(self, point_cloud):
        points = np.asarray(point_cloud.points)
        X = points[..., 0].max() - points[..., 0].min()
        Y = points[..., 1].max() - points[..., 1].min()
        Z = points[..., 2].max() - points[..., 2].min()

        return np.sqrt(X**2 + Y**2 + Z**2)

    def estimate_transform(self, dataset):
        raise NotImplementedError("Implement the virtual method.")

    def _estimate_transform(self, point_cloud_source, point_cloud_target,
                            compute_scale,
                            transformed_target_point_cloud_output_path):

        frame_target_T_frame_source = (
            self._estimate_transform_from_correspondences(
                point_cloud_source=point_cloud_source,
                point_cloud_target=point_cloud_target,
                compute_scale=compute_scale,
                transformed_target_point_cloud_output_path=
                transformed_target_point_cloud_output_path))

        # Compute the scale.
        estimated_scale_frame_target_T_frame_source = np.sqrt(
            (frame_target_T_frame_source[:3, :3]
             @ frame_target_T_frame_source[:3, :3].T)[0, 0])
        print("Estimated a scale of "
              f"{estimated_scale_frame_target_T_frame_source:.3f} from "
              "source to target.")

        return (frame_target_T_frame_source,
                estimated_scale_frame_target_T_frame_source)

    def _estimate_transform_from_correspondences(
            self, point_cloud_source, point_cloud_target, compute_scale,
            transformed_target_point_cloud_output_path):
        # Adapted from http://www.open3d.org/docs/release/tutorial/
        # visualization/interactive_visualization.html.

        # Pick points from the two point clouds and build correspondences.
        picked_id_source = self._pick_points(point_cloud_source)
        picked_id_target = self._pick_points(point_cloud_target)
        assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
        assert (len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target

        # Estimate rough transformation using the correspondences.
        print("Computing rough transform using the correspondences given by "
              "user.")
        p2p = (o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=compute_scale))
        trans_init = p2p.compute_transformation(
            point_cloud_source, point_cloud_target,
            o3d.utility.Vector2iVector(corr))
        # Point-to-point ICP for refinement.
        # - Compute length of the diagonal of the axis-aligned bounding box.
        np.asarray(point_cloud_source.points)[..., 0]
        print("Performing point-to-point ICP refinement.")
        # - Define the threshold based on the size of the point clouds.
        min_diagonal_point_cloud = min(
            self._compute_diagonal_point_cloud(point_cloud=point_cloud_source),
            self._compute_diagonal_point_cloud(point_cloud=point_cloud_target))
        threshold = 0.01 * min_diagonal_point_cloud
        frame_target_T_frame_source = (
            o3d.pipelines.registration.registration_icp(
                point_cloud_source, point_cloud_target, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                )))
        frame_target_T_frame_source = frame_target_T_frame_source.transformation
        self._draw_registration_result(
            point_cloud_source=point_cloud_source,
            point_cloud_target=point_cloud_target,
            frame_target_T_frame_source=frame_target_T_frame_source)

        # Visualization with original coloring of the point clouds.
        frame_target_point_cloud_source = copy.deepcopy(point_cloud_source)
        frame_source_points_source = np.asarray(
            frame_target_point_cloud_source.points)

        frame_target_point_cloud_source.points = (o3d.utility.Vector3dVector(
            ((frame_target_T_frame_source) @ np.concatenate(
                [
                    frame_source_points_source,
                    np.ones_like(frame_source_points_source[..., 0])[..., None]
                ],
                axis=-1).T).T[..., :3]))

        o3d.visualization.draw(
            [frame_target_point_cloud_source, point_cloud_target])

        if (transformed_target_point_cloud_output_path is not None):
            o3d.io.write_point_cloud(transformed_target_point_cloud_output_path,
                                     frame_target_point_cloud_source)

        return frame_target_T_frame_source
