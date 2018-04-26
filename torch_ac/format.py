import torch

def default_preprocess_obss(obss, use_gpu=False):
    obss = torch.tensor(obss)
    if use_gpu:
        obss = obss.cuda()
    return obss

def default_reshape_reward(obs, action, reward):
    return reward