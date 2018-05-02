import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ACModel(torch_ac.ACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        # Define actor's model
        self.a_fc1 = nn.Linear(obs_space["image"], 64)
        self.a_fc2 = nn.Linear(64, 64)
        self.a_fc3 = nn.Linear(64, action_space.n)

        # Define critic's model
        self.c_fc1 = nn.Linear(obs_space["image"], 64)
        self.c_fc2 = nn.Linear(64, 64)
        self.c_fc3 = nn.Linear(64, 1)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs):
        x = self.a_fc1(obs["image"])
        x = F.tanh(x)
        x = self.a_fc2(x)
        x = F.tanh(x)
        x = self.a_fc3(x)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.c_fc1(obs["image"])
        x = F.tanh(x)
        x = self.c_fc2(x)
        x = F.tanh(x)
        x = self.c_fc3(x)
        value = x.squeeze(1)

        return dist, value