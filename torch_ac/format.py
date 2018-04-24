import torch
from torch.autograd import Variable
import numpy as np

from torch_ac.utils import gpu_available

def default_preprocess_obss(obss, volatile=True, use_gpu=False):
    obss = torch.from_numpy(np.array(obss)).float()
    if use_gpu:
        obss = obss.cuda()
    return Variable(obss, volatile=volatile)

def default_reshape_reward(obs, action, reward):
    return reward