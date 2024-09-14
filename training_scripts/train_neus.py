import argparse
import os
import torch

from neusurfemb.data_utils.dataset import read_transform
from neusurfemb.misc_utils.argparse_utils import string_to_bool
from neusurfemb.misc_utils.flags import save_experiment_flags
from neusurfemb.misc_utils.random import seed_everything
from neusurfemb.neus.network import NeuSNetwork
from neusurfemb.training_utils.trainer import Trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="Path to the training dataset.")
    parser.add_argument(
        '--obj-id',
        type=int,
        help=("If given, ID of the object on which to train. This is required "
              "for datasets in which more than one object is contained, and "
              "has associated poses, in the scene (e.g, YCB-V)."))
    parser.add_argument('--workspace',
                        type=str,
                        help="Workspace of the training experiment.")
    parser.add_argument('--seed', type=int, default=0)

    ### Training options.
    parser.add_argument('--num-iters',
                        type=int,
                        default=4000,
                        help="Number of NeuS training iterations.")
    parser.add_argument('--ckpt', type=str, default='latest')

    ### Options for data generation.
    parser.add_argument(
        '--crop-res-train-dataset',
        type=int,
        help=("If specified, the training dataset is cropped and resized to "
              "this size, and coordinate maps are rendered for each image, "
              "after NeuS2 training."))
    parser.add_argument(
        '--crop-scale-factor-train-dataset',
        type=float,
        help=("If `--crop-res-train-dataset` is specified, the object-tight "
              "bounding boxes used to crop the images in the training dataset "
              "are enlarged by this factor. For instance, a value of 1.0 "
              "indicates that the bounding boxes are tight to the object "
              "boundaries."))
    parser.add_argument(
        '--synthetic-data-config-path',
        type=str,
        help=("If given, a synthetic dataset will be generated and used for "
              "training based on the configuration contained at this file "
              "path."))
    parser.add_argument(
        '--compute-oriented-bounding-box',
        type=string_to_bool,
        required=True,
        help=("If True, an oriented bounding box tightly fitted around the "
              "object is computed, so that when generating synthetic data, the "
              "sampled poses assume that the object is roughly parallel to the "
              "ground plane. If the original world frame is already properly "
              "aligned to the object (as it might be the case for datasets), "
              "set this flag to `False` for optimal results."))
    parser.add_argument(
        '--visualize-oriented-bounding-box',
        action='store_true',
        help=("Whether to visualize the oriented bounding box fitted around "
              "the object and used to adjusted the global coordinate frame so "
              "as to align better with the object."))

    ### Point cloud formation.
    parser.add_argument(
        '--num-points-correspondence-point-cloud',
        # Default value is same resolution as SurfEmb.
        default=75000,
        type=int,
        help=("Target number of points to which the formed point cloud later "
              "used for correspondence learning should be downsampled."))

    opt = parser.parse_args()

    assert ((opt.crop_res_train_dataset
             is not None) == (opt.crop_scale_factor_train_dataset is not None))

    print(opt)

    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuSNetwork()

    train_transform, train_transform_path = read_transform(root_path=opt.path,
                                                           obj_id=opt.obj_id)

    trainer = Trainer(opt, model=model, device=device, workspace=opt.workspace)

    # Save experiment config file.
    output_experiment_flags_path = os.path.join(
        opt.workspace,
        f"{os.path.basename(os.path.normpath(opt.workspace))}.yml")
    save_experiment_flags(
        parsed_training_arguments=opt,
        one_uom_scene_to_m=train_transform['one_uom_scene_to_m'],
        output_yaml_file_path=output_experiment_flags_path)

    trainer.train(train_transform=train_transform,
                  train_transform_path=train_transform_path)
