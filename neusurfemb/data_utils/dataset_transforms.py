"""The following file contains allows computing the transform between the
coordinate frames of two datasets representing the same object. This can be
useful, e.g., to use one dataset for validating when training on the other one.
The point clouds constructed from NeuS models trained on the two datasets are
used to compute the transforms by using an initial PnP-based estimation that
relies on user-defined correspondences, followed by an ICP-based local
refinement.
"""

import glob
import hashlib
import json
import numpy as np
import open3d as o3d
import os
from pathlib import Path
import yaml

from neusurfemb.data_utils.icp_registration import ICPRegistrator


def hash_dataset(dataset_root_path, print_files_used_for_hash=False):
    # Based on https://stackoverflow.com/a/54477583.
    def _md5_update_from_file(filename, hash):
        assert Path(filename).is_file()
        if (print_files_used_for_hash):
            print(f"\tUsing file {filename} for hashing")
        with open(str(filename), "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash.update(chunk)
        return hash

    def _md5_update_from_dir(directory, hash):
        _valid_folder_names = ["depth", "rgb", "pose"]
        assert Path(directory).is_dir()
        for path in sorted(Path(directory).iterdir()):
            if (path.is_file()):
                extension = path.name.rsplit('.', maxsplit=1)[-1]
                if (path.name == "transforms.json" or
                        path.parent.name in _valid_folder_names):
                    hash.update(path.name.encode())
                    hash = _md5_update_from_file(filename=path, hash=hash)
            elif (path.is_dir() and path.name in _valid_folder_names):
                hash.update(path.name.encode())
                hash = _md5_update_from_dir(directory=path, hash=hash)
        return hash

    return _md5_update_from_dir(directory=dataset_root_path,
                                hash=hashlib.md5()).hexdigest()


class DatasetTransformComputer(ICPRegistrator):

    def __init__(self):
        super().__init__()
        self._hash_id = {}
        self._transforms_from_dataset = {}
        self._point_clouds = {}
        self._transform_path = {}

    def _write_transform_to_file(self, transform_dict, output_path):
        with open(output_path, "w") as f:
            yaml.dump(transform_dict, f)

    def _check_existing_target_T_source(self, source_dataset, target_dataset):
        # Generate the hash ID associated to the datasets used to train each of
        # the two NeRF models.
        # - Compute hash IDs.
        hash_id_source = self._hash_id["source"] = hash_dataset(
            dataset_root_path=source_dataset)
        hash_id_target = self._hash_id["target"] = hash_dataset(
            dataset_root_path=target_dataset)
        # - If available, read transforms from each of the datasets.
        self._transform_path["source"] = os.path.join(
            source_dataset, "transform_between_datasets")
        self._transform_path["target"] = os.path.join(
            target_dataset, "transform_between_datasets")
        for dataset_type in ["source", "target"]:
            if (os.path.exists(self._transform_path[dataset_type])):
                with open(self._transform_path[dataset_type], "r") as f:
                    self._transforms_from_dataset[dataset_type] = yaml.load(
                        f, Loader=yaml.SafeLoader)
            else:
                self._transforms_from_dataset[dataset_type] = {'targets': {}}

        # Check if the transforms between the pair of datasets were already
        # computed.
        frame_target_T_frame_source = None
        frame_source_T_frame_target = None

        transforms_from_source = self._transforms_from_dataset["source"][
            "targets"]
        transforms_from_target = self._transforms_from_dataset["target"][
            "targets"]
        should_compute_new_transforms = True
        if (hash_id_target in transforms_from_source):
            frame_target_T_frame_source = np.array(
                transforms_from_source[hash_id_target])
            print("\033[94mAlready found transform from dataset "
                  f"'{hash_id_source}' to dataset '{hash_id_target}': "
                  f"{frame_target_T_frame_source}. Will not compute a new one."
                  "\033[0m")
            should_compute_new_transforms = False
        if (hash_id_source in transforms_from_target):
            frame_source_T_frame_target = np.array(
                transforms_from_target[hash_id_source])
            print("\033[94mAlready found transform from dataset "
                  f"'{hash_id_target}' to dataset '{hash_id_source}': "
                  f"{frame_source_T_frame_target}. Will not compute a new one."
                  "\033[0m")
            should_compute_new_transforms = False
        if (frame_target_T_frame_source is None):
            if (frame_source_T_frame_target is not None):
                frame_target_T_frame_source = np.linalg.inv(
                    frame_source_T_frame_target)
                transforms_from_source[
                    hash_id_target] = frame_target_T_frame_source.tolist()
                print("\033[94mComputed transform from dataset "
                      f"'{hash_id_source}' to dataset '{hash_id_target}' from "
                      "its inverse.\033[0m")
                # Write to file.
                self._write_transform_to_file(
                    transform_dict={
                        "source": hash_id_source,
                        "targets": transforms_from_source
                    },
                    output_path=self._transform_path["source"])
        else:
            if (frame_source_T_frame_target is not None):
                # Check that matrices are the inverse of one another.
                if (not np.all(
                        np.isclose(
                            frame_source_T_frame_target
                            @ frame_target_T_frame_source, np.eye(4)))):
                    raise ValueError(
                        "Found both transform from dataset "
                        f"'{hash_id_source}' to dataset '{hash_id_target}' "
                        "and viceversa, but they are not inverse of each "
                        "other.")
            else:
                frame_source_T_frame_target = np.linalg.inv(
                    frame_target_T_frame_source)
                transforms_from_target[
                    hash_id_source] = frame_source_T_frame_target.tolist()
                print("\033[94mComputed transform from dataset "
                      f"'{hash_id_target}' to dataset '{hash_id_source}' from "
                      "its inverse.\033[0m")
                # Write to file.
                self._write_transform_to_file(
                    transform_dict={
                        "source": hash_id_target,
                        "targets": transforms_from_target
                    },
                    output_path=self._transform_path["target"])

        if (should_compute_new_transforms):
            return None
        else:
            return frame_target_T_frame_source

    def manually_set_transform(self, source_dataset, target_dataset,
                               target_T_source):
        assert (isinstance(target_T_source, np.ndarray) and
                target_T_source.shape == (4, 4) and np.all(
                    np.isclose(
                        target_T_source[:3, :3] @ target_T_source[:3, :3].T,
                        np.eye(3))))

        # Check if there exists already a transform between the two datasets.
        existing_target_T_source = self._check_existing_target_T_source(
            source_dataset=source_dataset, target_dataset=target_dataset)

        hash_id_source = self._hash_id["source"]
        hash_id_target = self._hash_id["target"]

        # Find the scale that the resulting transform should have, based on the
        # `one_uom_scene_to_m` values extracted from the `transforms.json`
        # files.
        scale_frame_target_T_frame_source = (
            self._find_one_uom_source_to_uom_target(
                source_dataset=source_dataset, target_dataset=target_dataset))

        # Multiply the input rotation matrix by the scale matrix.
        target_T_source[:3, :3] = (target_T_source[:3, :3] *
                                   scale_frame_target_T_frame_source)

        if (existing_target_T_source is not None):
            assert (
                np.all(np.isclose(target_T_source, existing_target_T_source))
            ), ("Could not set the transform from source frame "
                f"'{hash_id_source}' to target frame '{hash_id_target}' to "
                f"{target_T_source} because it was already previously set to "
                f"{existing_target_T_source}.")
            print(f"\033[94mThe transform from source frame '{hash_id_source}' "
                  f"to target frame '{hash_id_target}' had been already set to "
                  f"the desired value {target_T_source}. Nothing else is done."
                  "\033[0m")
        else:
            # Write the new transform to file.
            # Write to file.
            self._transforms_from_dataset["source"]["targets"][
                hash_id_target] = target_T_source.tolist()
            self._transforms_from_dataset["target"]["targets"][
                hash_id_source] = np.linalg.inv(target_T_source).tolist()
            self._write_transform_to_file(
                transform_dict={
                    "source":
                        hash_id_source,
                    "targets":
                        self._transforms_from_dataset["source"]["targets"]
                },
                output_path=self._transform_path["source"])
            self._write_transform_to_file(
                transform_dict={
                    "source":
                        hash_id_target,
                    "targets":
                        self._transforms_from_dataset["target"]["targets"]
                },
                output_path=self._transform_path["target"])

    def estimate_transform(self,
                           source_dataset,
                           target_dataset,
                           also_compute_scale_from_jsons=False):
        # Check if there exists already a transform between the two datasets.
        frame_target_T_frame_source = self._check_existing_target_T_source(
            source_dataset=source_dataset, target_dataset=target_dataset)

        # If necessary to compute the inter-dataset transform, read the
        # corresponding point clouds.
        if (frame_target_T_frame_source is None):
            mesh_source_path = glob.glob(
                os.path.join(source_dataset, "data_for_surfemb", "models",
                             "obj_*.ply"))
            assert (len(mesh_source_path) == 1)
            mesh_source_path = mesh_source_path[0]
            self._point_clouds["source"] = o3d.geometry.PointCloud()
            mesh_source = o3d.io.read_triangle_mesh(filename=mesh_source_path)
            self._point_clouds["source"].points = o3d.utility.Vector3dVector(
                mesh_source.vertices)
            self._point_clouds["source"].colors = o3d.utility.Vector3dVector(
                mesh_source.vertex_colors)
            mesh_target_path = glob.glob(
                os.path.join(target_dataset, "data_for_surfemb", "models",
                             "obj_*.ply"))
            assert (len(mesh_target_path) == 1)
            mesh_target_path = mesh_target_path[0]
            self._point_clouds["target"] = o3d.geometry.PointCloud()
            mesh_target = o3d.io.read_triangle_mesh(filename=mesh_target_path)
            self._point_clouds["target"].points = o3d.utility.Vector3dVector(
                mesh_target.vertices)
            self._point_clouds["target"].colors = o3d.utility.Vector3dVector(
                mesh_target.vertex_colors)

            (frame_target_T_frame_source,
             estimated_scale_frame_target_T_frame_source) = (
                 self._estimate_transform(
                     point_cloud_source=self._point_clouds["source"],
                     point_cloud_target=self._point_clouds["target"],
                     compute_scale=True,
                     transformed_target_point_cloud_output_path=None))

            # Write the estimated transforms to file.
            hash_id_source = self._hash_id["source"]
            hash_id_target = self._hash_id["target"]
            self._transforms_from_dataset["source"]["targets"][
                hash_id_target] = frame_target_T_frame_source.tolist()
            self._transforms_from_dataset["target"]["targets"][
                hash_id_source] = np.linalg.inv(
                    frame_target_T_frame_source).tolist()

            self._write_transform_to_file(
                transform_dict={
                    "source":
                        hash_id_source,
                    "targets":
                        self._transforms_from_dataset["source"]["targets"]
                },
                output_path=self._transform_path["source"])
            self._write_transform_to_file(
                transform_dict={
                    "source":
                        hash_id_target,
                    "targets":
                        self._transforms_from_dataset["target"]["targets"]
                },
                output_path=self._transform_path["target"])

            if (also_compute_scale_from_jsons):
                # Compare the estimated scale and the actual scale from the
                # poses.
                actual_scale_frame_target_T_frame_source = (
                    self._find_one_uom_source_to_uom_target(
                        source_dataset=source_dataset,
                        target_dataset=target_dataset))
                print("The scale computed from the `one_uom_scene_to_m` fields "
                      "in the `transforms.json` files is "
                      f"{actual_scale_frame_target_T_frame_source:.3f}.")

        return frame_target_T_frame_source

    def _find_one_uom_source_to_uom_target(self, source_dataset,
                                           target_dataset):
        # Find the scale conversion between the datasets by reading the
        # `transforms.json` files.
        json_paths = {
            k: sorted([*Path(dataset).glob("transforms.json")])
            for (k, dataset
                ) in zip(["source", "target"], [source_dataset, target_dataset])
        }
        one_uom_scene_to_m = {}
        for k, dataset in zip(["source", "target"],
                              [source_dataset, target_dataset]):
            assert (len(json_paths[k]) == 1)
            transforms_json_path = json_paths[k][0]
            with open(transforms_json_path, "r") as f:
                one_uom_scene_to_m[k] = float(
                    json.load(f)['one_uom_scene_to_m'])

        return one_uom_scene_to_m["source"] / one_uom_scene_to_m["target"]
