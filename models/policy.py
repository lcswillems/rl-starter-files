import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.math import *


class Policy(nn.Module):
    def __init__(self, obs_space, action_space, activation='tanh'):
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

        self.action_head = nn.Linear(128, action_space.n)
        self.action_head.weight.data.mul_(0.1)
        self.action_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.layers:
            x = self.activation(affine(x))

        # print(self.action_head(x))

        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

    def select_action(self, x):
        action_prob = self.forward(x)
        action = action_prob.multinomial()
        return action.data

    def get_kl(self, x):
        action_prob1 = self.forward(x)
        action_prob0 = Variable(action_prob1.data)
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        action_prob = self.forward(x)
        return torch.log(action_prob.gather(1, actions.unsqueeze(1)))

    def get_fim(self, x):
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).data
        return M, action_prob, {}

