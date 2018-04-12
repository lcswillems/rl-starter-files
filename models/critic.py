import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Value(nn.Module):
    def __init__(self, obs_space, activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(obs_space.shape[1], 128))
        self.layers.append(nn.Linear(128, 128))

        self.value_head = nn.Linear(128, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

        self.apply(weights_init)

    def forward(self, x):
        for affine in self.layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value

    def get_loss(self, x, target):
        pred = self.forward(x)
        return (pred - target).pow(2).mean()