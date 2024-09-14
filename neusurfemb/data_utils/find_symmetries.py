import argparse
import glob
import json
import numpy as np
import open3d as o3d
import os

from neusurfemb.data_utils.icp_registration import ICPRegistrator


class SymmetryFinder(ICPRegistrator):

    def __init__(self):
        super().__init__()

    def estimate_transform(self,
                           dataset,
                           transformed_point_cloud_output_path=None):
        mesh_path = glob.glob(
            os.path.join(dataset, "data_for_surfemb", "models_eval",
                         "obj_*.ply"))
        assert (len(mesh_path) == 1)
        mesh_path = mesh_path[0]
        mesh = o3d.io.read_triangle_mesh(filename=mesh_path)
        self._point_cloud = o3d.geometry.PointCloud()
        self._point_cloud.points = o3d.utility.Vector3dVector(mesh.vertices)
        self._point_cloud.colors = o3d.utility.Vector3dVector(
            mesh.vertex_colors)

        (frame_target_T_frame_source,
         estimated_scale_frame_target_T_frame_source
        ) = self._estimate_transform(point_cloud_source=self._point_cloud,
                                     point_cloud_target=self._point_cloud,
                                     compute_scale=False,
                                     transformed_target_point_cloud_output_path=
                                     transformed_point_cloud_output_path)
        assert (np.isclose(estimated_scale_frame_target_T_frame_source, 1.0))

        # Update the `models_info.json` file.
        models_info_path = os.path.join(dataset, "data_for_surfemb",
                                        "models_eval", "models_info.json")
        with open(models_info_path, "r") as f:
            models_info = json.load(f)
        obj_id = int(
            os.path.basename(mesh_path).split(".")[0].split("obj_")[-1])
        models_info[str(obj_id)]["symmetries_discrete"] = [
            frame_target_T_frame_source.reshape(-1).tolist()
        ]
        with open(models_info_path, "w") as f:
            models_info = json.dump(models_info, f)

        return frame_target_T_frame_source


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)

    args = parser.parse_args()

    dataset_path = args.dataset_path

    symmetry_finder = SymmetryFinder()
    T = symmetry_finder.estimate_transform(dataset=dataset_path)
    print(T)
