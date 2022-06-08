import random

import numpy as np
import torch


def set_seed(seed):  # pragma: no cover
    """Set the provided seed in python/numpy/DL framework.

    :param seed: (int) the seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
