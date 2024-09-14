import argparse
import copy
import glob
import json
import os
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('--datasets-to-merge', nargs='+', required=True)
parser.add_argument('--path-output-dataset', type=str, required=True)

args = parser.parse_args()

datasets_to_merge = args.datasets_to_merge
path_output_dataset = args.path_output_dataset
path_output_converted_train = os.path.join(path_output_dataset,
                                           "converted_train_dataset")
path_output_synthetic = os.path.join(path_output_dataset, "synthetic_dataset")
path_output_models = os.path.join(path_output_dataset, "models")
path_output_surface_samples = os.path.join(path_output_dataset,
                                           "surface_samples")
path_output_surface_samples_normals = os.path.join(path_output_dataset,
                                                   "surface_samples_normals")

assert (len(args.datasets_to_merge) > 1)

# Create the output folders.
for output_folder in [
        path_output_dataset, path_output_converted_train, path_output_synthetic,
        path_output_models, path_output_surface_samples,
        path_output_surface_samples_normals
]:
    os.makedirs(name=output_folder, exist_ok=False)

# Check that the YAML config files match and copy them over.
cfg_converted_train = None
cfg_synthetic = None

for dataset_to_merge in datasets_to_merge:
    path_cfg_converted_train = os.path.join(dataset_to_merge,
                                            "cfg_converted_train.yml")
    path_cfg_synthetic = os.path.join(dataset_to_merge, "cfg_synthetic.yml")
    with open(path_cfg_converted_train, "r") as f:
        curr_cfg_converted_train = yaml.load(f, Loader=yaml.SafeLoader)
    with open(path_cfg_synthetic, "r") as f:
        curr_cfg_synthetic = yaml.load(f, Loader=yaml.SafeLoader)
    if (cfg_converted_train is None):
        cfg_converted_train = copy.deepcopy(curr_cfg_converted_train)
        cfg_synthetic = copy.deepcopy(curr_cfg_synthetic)
    else:
        # - Check match.
        assert (cfg_converted_train == curr_cfg_converted_train)
        assert (cfg_synthetic == curr_cfg_synthetic)
# - Add symbolic link to files.
os.symlink(src=os.path.abspath(path_cfg_converted_train),
           dst=os.path.join(path_output_dataset,
                            os.path.basename(path_cfg_converted_train)))
os.symlink(src=os.path.abspath(path_cfg_synthetic),
           dst=os.path.join(path_output_dataset,
                            os.path.basename(path_cfg_synthetic)))

# Merge the datasets, checking that they have different object IDs.
obj_id_to_dataset = {}
merged_models_info = {}
for dataset_to_merge in datasets_to_merge:
    curr_obj_id = sorted(
        glob.glob(os.path.join(dataset_to_merge, "converted_train_dataset",
                               "*")))
    assert (len(curr_obj_id) == 1)
    curr_converted_train_path = curr_obj_id[0]
    curr_obj_id = os.path.basename(curr_converted_train_path)

    assert (curr_obj_id not in obj_id_to_dataset)
    obj_id_to_dataset[curr_obj_id] = dataset_to_merge
    # - Add symbolic link to current converted-train dataset.
    os.symlink(src=os.path.abspath(curr_converted_train_path),
               dst=os.path.join(path_output_dataset,
                                path_output_converted_train, curr_obj_id))

    # - Check that the obj ID is consistent in the other folders.
    curr_synthetic_path = sorted(
        glob.glob(os.path.join(dataset_to_merge, "synthetic_dataset", "*")))
    assert (len(curr_synthetic_path) == 1)
    curr_synthetic_path = curr_synthetic_path[0]
    assert (os.path.basename(curr_synthetic_path) == curr_obj_id)
    # - Add symbolic link to current synthetic dataset.
    os.symlink(src=os.path.abspath(curr_synthetic_path),
               dst=os.path.join(path_output_dataset, path_output_synthetic,
                                curr_obj_id))

    curr_models_folder = os.path.join(dataset_to_merge, "models")
    curr_models_info_path = os.path.join(curr_models_folder, "models_info.json")
    curr_obj_path = os.path.join(curr_models_folder, f"obj_{curr_obj_id}.ply")
    curr_models_paths = sorted(glob.glob(os.path.join(curr_models_folder, "*")))
    assert (len(curr_models_paths) == 2 and
            curr_models_info_path in curr_models_paths and
            curr_obj_path in curr_models_paths)
    # - Read `models_info.json` file.
    with open(curr_models_info_path, "r") as f:
        curr_models_info = json.load(f)

    curr_obj_id_without_zeros = str(int(curr_obj_id))
    assert (list(curr_models_info.keys()) == [curr_obj_id_without_zeros])
    assert (list(curr_models_info[curr_obj_id_without_zeros].keys()) == sorted(
        ["diameter", "min_x", "min_y", "min_z", "size_x", "size_y", "size_z"]))
    merged_models_info.update(curr_models_info)
    # - Add symbolic link to current object model.
    os.symlink(src=os.path.abspath(curr_obj_path),
               dst=os.path.join(path_output_dataset, path_output_models,
                                os.path.basename(curr_obj_path)))

    curr_surface_samples_path = sorted(
        glob.glob(os.path.join(dataset_to_merge, "surface_samples", "*")))
    assert (len(curr_surface_samples_path) == 1)
    curr_surface_samples_path = curr_surface_samples_path[0]
    assert (
        os.path.basename(curr_surface_samples_path) == f"obj_{curr_obj_id}.ply")
    # - Add symbolic link to current surface samples.
    os.symlink(src=os.path.abspath(curr_surface_samples_path),
               dst=os.path.join(path_output_dataset,
                                path_output_surface_samples,
                                os.path.basename(curr_surface_samples_path)))

    curr_surface_samples_normals_path = sorted(
        glob.glob(os.path.join(dataset_to_merge, "surface_samples_normals",
                               "*")))
    assert (len(curr_surface_samples_normals_path) == 1)
    curr_surface_samples_normals_path = curr_surface_samples_normals_path[0]
    assert (os.path.basename(curr_surface_samples_normals_path) ==
            f"obj_{curr_obj_id}.ply")
    # - Add symbolic link to current surface samples normals.
    os.symlink(src=os.path.abspath(curr_surface_samples_normals_path),
               dst=os.path.join(
                   path_output_dataset, path_output_surface_samples_normals,
                   os.path.basename(curr_surface_samples_normals_path)))

# Save merged `models_info.json` file.
with open(os.path.join(path_output_models, "models_info.json"), "w") as f:
    json.dump(obj=merged_models_info, fp=f, indent=2)

# Add symbolic link to the Neus2 checkpoints.
for dataset_to_merge in datasets_to_merge:
    curr_checkpoint_folder = os.path.join(dataset_to_merge, "checkpoints")
    os.symlink(src=os.path.abspath(curr_checkpoint_folder),
               dst=os.path.join(
                   path_output_dataset,
                   f"checkpoints_{os.path.basename(dataset_to_merge)}"))
