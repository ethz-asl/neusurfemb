"""Partly based on https://github.com/19reborn/NeuS2/blob/main/scripts/run.py.
"""

import argparse
import copy
import glob
import json
import numpy as np
import os
import pandas as pd
import sys
import time

from pyntcloud import PyntCloud

import pyngp as ngp  # noqa

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
from neusurfemb.neus.network import NeuSNetwork

parser = argparse.ArgumentParser(description=(
    "Run neural graphics primitives testbed with additional configuration & "
    "output options"))

parser.add_argument("--name", type=str, required=True)
parser.add_argument("--scene-transform-path",
                    type=str,
                    required=True,
                    help="Path to the transform file of the scene to load.")
parser.add_argument("--n-steps",
                    type=int,
                    required=True,
                    help="Number of steps to train for before quitting.")

parser.add_argument(
    '--bound-extent',
    type=float,
    required=True,
    help=("Extent of the scene bounds (between 0 and 1) that should contain "
          "the object."))
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--should-convert-to-mm', action='store_true')

args = parser.parse_args()

output_folder = os.path.join(os.path.dirname(args.scene_transform_path),
                             "neus_rescaling")
os.makedirs(output_folder)
output_path = os.path.join(output_folder, args.name)
os.makedirs(os.path.join(output_path, "checkpoints"), exist_ok=True)

time_name = time.strftime("%m_%d_%H_%M", time.localtime())

n_steps = args.n_steps

neus_network = NeuSNetwork()

checkpoint_folder = os.path.join(output_path, "checkpoints")
neus_network._train_neus(path_to_scene_json=args.scene_transform_path,
                         num_steps=n_steps,
                         checkpoint_folder=checkpoint_folder)
pc_mc_points, pc_mc_colors, _ = neus_network.compute_point_cloud()

num_points_downsampled = 75000
random_permutation_mc = np.random.permutation(len(pc_mc_points))

_J_bop = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.], [0., -1., 0., 0.],
                   [0., 0., 0., 1.]])
W_BOP_T_W_NeuS = np.linalg.inv(_J_bop)

# Bring to BOP coordinates.
pc_mc_points = (pc_mc_points[random_permutation_mc[:num_points_downsampled]]
                @ W_BOP_T_W_NeuS[:3, :3].T)
pc_mc_colors = pc_mc_colors[random_permutation_mc[:num_points_downsampled]]

# Save the point cloud.
point_cloud_BOP_dict = {
    "x": pc_mc_points[..., 0],
    "y": pc_mc_points[..., 1],
    "z": pc_mc_points[..., 2]
}
# - Include color if present.
assert (pc_mc_colors.ndim == 2)
if (pc_mc_colors.shape[-1] != 0):
    assert (pc_mc_colors.shape[-1] == 3)
    point_cloud_BOP_dict["red"] = (255 * pc_mc_colors[..., 0]).astype(np.uint8)
    point_cloud_BOP_dict["green"] = (255 * pc_mc_colors[..., 1]).astype(
        np.uint8)
    point_cloud_BOP_dict["blue"] = (255 * pc_mc_colors[..., 2]).astype(
        np.uint8)
point_cloud_BOP = PyntCloud(pd.DataFrame(data=point_cloud_BOP_dict))
point_cloud_BOP_path = os.path.join(output_path, "full_point_cloud.ply")
point_cloud_BOP.to_file(point_cloud_BOP_path)

print(f"\033[94mSaved point cloud in NeuS scale to '{point_cloud_BOP_path}'."
      "\033[0m")

B = 0.5

assert (np.all(neus_network.neus.aabb.center() == (
    neus_network.neus.nerf.training.dataset.offset)))
assert (np.all(neus_network.neus.aabb.center() == [B, B, B]))
assert (np.all(neus_network.neus.aabb.min == [0., 0., 0.]) and
        np.all(neus_network.neus.aabb.max == [2 * B, 2 * B, 2 * B]))

if (args.visualize):
    assert (not args.should_convert_to_mm)
    import open3d as o3d
    pc_mc_o3d = o3d.geometry.PointCloud()
    pc_mc_o3d.points = o3d.utility.Vector3dVector(pc_mc_points)
    pc_mc_o3d.colors = o3d.utility.Vector3dVector(pc_mc_colors)

    points = [[-B, -B, -B], [-B, -B, B], [-B, B, -B], [-B, B, B], [B, -B, -B],
              [B, -B, B], [B, B, -B], [B, B, B]]

    lines = []
    for idx_i in range(len(points)):
        for idx_j in range(idx_i + 1, len(points)):
            if (np.count_nonzero(
                    np.array(points[idx_i]) - np.array(points[idx_j])) == 1):
                lines.append([idx_i, idx_j])

    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw([pc_mc_o3d, line_set])

# Find the bounds of the (downsampled) point cloud.
coords_max = pc_mc_points.max(axis=0)
coords_min = pc_mc_points.min(axis=0)

coords_center = (coords_max + coords_min) / 2.

one_uom_scene_to_m = 1. / neus_network.neus.nerf.training.dataset.scale

coords_center_in_m = coords_center * one_uom_scene_to_m

old_scale_to_new_scale = 2 * args.bound_extent * B / (coords_max -
                                                      coords_min).max()

# Create a transformed version of the dataset.
all_json_paths = neus_network.neus.get_json_paths()
assert (len(all_json_paths))
json_path = all_json_paths[0]
with open(json_path, "r") as f:
    orig_transform = json.load(f)

new_transform = copy.deepcopy(orig_transform)
new_transform['one_uom_scene_to_m'] /= old_scale_to_new_scale
new_transform['scale'] *= old_scale_to_new_scale
try:
    new_transform['integer_depth_scale'] /= old_scale_to_new_scale
except KeyError:
    pass

# Copy all the dataset data into the temporary folder.
new_dataset_folder = os.path.join(output_path, "new_dataset")
print(f"Copying original dataset to '{new_dataset_folder}'...")
os.makedirs(new_dataset_folder)

for folder_or_file in sorted(
        glob.glob(os.path.join(os.path.dirname(json_path), "*"))):
    if (folder_or_file[-5:] == ".json"):
        continue
    else:
        os.symlink(
            folder_or_file,
            os.path.join(new_dataset_folder, os.path.basename(folder_or_file)))

# Add the JSON file.
output_json_path = os.path.join(new_dataset_folder, os.path.basename(json_path))

with open(output_json_path, "w") as f:
    json.dump(new_transform, f, indent=4)

# Optionally convert to point cloud to mm.
if (args.should_convert_to_mm):
    pc_mc_points = (pc_mc_points /
                    neus_network.neus.nerf.training.dataset.scale * 1000.)
    point_cloud_mm_dict = {
        "x": pc_mc_points[..., 0],
        "y": pc_mc_points[..., 1],
        "z": pc_mc_points[..., 2]
    }
    for color in ["red", "green", "blue"]:
        point_cloud_mm_dict[color] = point_cloud_BOP_dict[color]
    point_cloud_mm = PyntCloud(pd.DataFrame(data=point_cloud_mm_dict))
    point_cloud_mm_path = os.path.join(output_path, "full_point_cloud_mm.ply")
    point_cloud_mm.to_file(point_cloud_mm_path)

    print(f"\033[94mSaved point cloud in mm to '{point_cloud_mm_path}'.\033[0m")
