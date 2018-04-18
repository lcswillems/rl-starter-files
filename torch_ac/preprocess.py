import torch
from torch.autograd import Variable
import numpy as np

def default_preprocess_obss(obss, volatile):
    obss = torch.from_numpy(np.array(obss)).float()
    return Variable(obss, volatile=volatile)

def default_preprocess_reward(obs, action, reward):
    return reward