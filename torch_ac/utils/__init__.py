import random
import numpy as np
import torch

from torch_ac.utils.dictlist import DictList
from torch_ac.utils.multienv import MultiEnv

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)