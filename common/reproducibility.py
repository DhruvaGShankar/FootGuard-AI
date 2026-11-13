"""
FootGuard AI - Reproducibility Module
Ensures deterministic operations across random, numpy, and torch.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set fixed random seed across Python random, NumPy, and PyTorch (CPU and CUDA).
    Forces deterministic CUDNN behavior for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

