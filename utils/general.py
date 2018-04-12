from os import path
import torch

use_gpu = torch.cuda.is_available()

def weights_initialization(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))