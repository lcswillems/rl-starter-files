import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x):
        for affine in self.layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value
