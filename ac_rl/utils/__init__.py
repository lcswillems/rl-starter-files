import numpy as np
import torch

from ac_rl.utils.dictlist import dictlist
from ac_rl.utils.multienv import MultiEnv

use_gpu = torch.cuda.is_available()

def seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)