import numpy as np

# Denoting the world frame with NeuS convention as `W_N`, one has
# J = W_N_T_W_{not_N}, that is, `J` is the transform from the original world
# frame to the world frame with NeuS convention. `J` is `_W_NEUS_T_W_COLMAP` for
# datasets processed with `dataset_scripts/pose_labeling.py` and
# `_W_NEUS_T_W_BOP` for BOP datasets. Cf. also
# `dataset_scripts/real_dataset_to_neus.py` and
# `dataset_scripts/bop_dataset_to_neus.py`.
_W_NEUS_T_W_BOP = np.array([[1., 0., 0., 0.], [0., 0., 1., 0.],
                            [0., -1., 0., 0.], [0., 0., 0., 1.]])
_W_NEUS_T_W_COLMAP = np.array([[1., 0., 0., 0.], [0., 0., -1., 0.],
                               [0., 1., 0., 0.], [0., 0., 0., 1.]])
_W_NEUS_T_SCENE = np.array([[0., 1., 0., 0.], [0., 0., 1., 0.],
                            [1., 0., 0., 0.], [0., 0., 0., 1.]])


def ngp_to_nerf_matrix(pose, scale):
    new_pose = np.array([
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] / scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] / scale],
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] / scale],
        [0, 0, 0, 1],
    ],
                        dtype=np.float32)
    return new_pose


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ],
                        dtype=np.float32)
    return new_pose
