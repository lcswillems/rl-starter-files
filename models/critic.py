import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import weights_initialization

class Value(nn.Module):
    def __init__(self, obs_space, activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid

        self.fc1 = nn.Linear(obs_space.shape[1], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        self.apply(weights_initialization)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x

    def get_loss(self, x, target):
        pred = self.forward(x)
        return (pred - target).pow(2).mean()