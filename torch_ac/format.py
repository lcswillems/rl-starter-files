import torch

from torch_ac.utils import gpu_available

def default_preprocess_obss(obss, requires_grad=False, use_gpu=False):
    obss = torch.tensor(obss, requires_grad=requires_grad)
    if use_gpu:
        obss = obss.cuda()
    return obss

def default_reshape_reward(obs, action, reward):
    return reward