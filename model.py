import torch.nn as nn
import torch.nn.functional as F

import ac_rl

class ACModel(ac_rl.ACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        self.a_fc1 = nn.Linear(obs_space.shape[1], 128)
        self.a_fc2 = nn.Linear(128, 128)
        self.a_head = nn.Linear(128, action_space.n)

        self.c_fc1 = nn.Linear(obs_space.shape[1], 128)
        self.c_fc2 = nn.Linear(128, 128)
        self.c_head = nn.Linear(128, 1)

        self.a_head.weight.data.mul_(0.1)
        self.a_head.bias.data.mul_(0.0)

    def forward(self, obs):
        x = self.a_fc1(obs)
        x = F.tanh(x)
        x = self.a_fc2(x)
        x = F.tanh(x)
        rdist = self.a_head(x)

        x = self.c_fc1(obs)
        x = F.tanh(x)
        x = self.c_fc2(x)
        x = F.tanh(x)
        value = self.c_head(x)

        return rdist, value