import numpy as np
from spatialmath import SE3, SO3
from spatialmath.base import trnorm


def invert_pose(pose: np.ndarray):
    assert (isinstance(pose, np.ndarray) and pose.shape == (4, 4))

    assert (np.isclose(np.sqrt((pose[:3, :3] @ pose[:3, :3].T)[0, 0]), 1.))

    try:
        return SE3(pose).inv().data[0]
    except ValueError:
        # Try orthogonalizing the rotation matrix if it is not orthogonal.
        R = pose[:3, :3].copy()
        t = pose[:3, 3].copy()
        R = SO3(trnorm(R)).data[0]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return SE3(T).inv().data[0]
