import numpy as np
import os
import random
import torch

def seed_everything(seed):
    # From https://github.com/ashawkey/torch-ngp/blob/b6e080468925f0bb44827b4f8f0ed08291dcf8a9/nerf/utils.py#L140
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)