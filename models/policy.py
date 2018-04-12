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

        self.fc1 = nn.Linear(obs_space.shape[1], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_space.n)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def select_action(self, x):
        raw_dist = self.forward(x)
        dist = F.softmax(raw_dist, dim=1)
        action = dist.multinomial()
        return action.data
    
    def get_loss(self, x, action, advantage):
        raw_dist = self.forward(x)
        log_dist = F.log_softmax(raw_dist, dim=1)
        dist = F.softmax(raw_dist, dim=1)
        entropy = -(log_dist * dist).sum(dim=1).mean()
        action_log_prob = log_dist.gather(1, action)
        action_loss = - (action_log_prob * advantage).mean()
        # print("entropy: {:.5f}".format(entropy.data[0]))
        # print("action loss: {:.5f}".format(action_loss.data[0]))
        return action_loss - 0.01 * entropy

    def get_kl(self, x):
        raise NotImplementedError
        action_prob1 = self.forward(x)
        action_prob0 = Variable(action_prob1.data)
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, x, actions):
        raise NotImplementedError
        action_prob = self.forward(x)
        return torch.log(action_prob.gather(1, actions))
    
    def get_entropy(self, x):
        raise NotImplementedError
        action_prob = self.forward(x)
        return -(torch.log(action_prob) * action_prob).sum(-1).mean()

    def get_fim(self, x):
        raise NotImplementedError
        action_prob = self.forward(x)
        M = action_prob.pow(-1).view(-1).data
        return M, action_prob, {}

