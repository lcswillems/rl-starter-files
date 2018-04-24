import random
import numpy as np
import torch

from torch_ac.utils.dictlist import DictList
from torch_ac.utils.multienv import MultiEnv

gpu_available = torch.cuda.is_available()

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu_available:
        torch.cuda.manual_seed_all(seed)