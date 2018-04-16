import numpy as np
import torch

from torch_ac.utils.dictlist import DictList
from torch_ac.utils.multienv import MultiEnv

use_gpu = torch.cuda.is_available()

def seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)