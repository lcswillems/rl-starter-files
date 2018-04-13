import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import initialize_weights

class Value(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        self.fc1 = nn.Linear(obs_space.shape[1], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

        self.apply(initialize_weights)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.fc3(x)
        return x