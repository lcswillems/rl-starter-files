import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.utils import initialize_parameters

class Policy(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()

        self.fc1 = nn.Linear(obs_space.shape[1], 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, action_space.n)

        self.policy_head.weight.data.mul_(0.1)
        self.policy_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        x = F.tanh(x)
        x = self.policy_head(x)
        return x

    def get_dist(self, x):
        return F.softmax(self(x), dim=1)

    def get_sampled_action(self, x):
        return self.get_dist(x).multinomial().data
    
    def get_best_action(self, x):
        return np.argmax(self.get_dist(x).data)