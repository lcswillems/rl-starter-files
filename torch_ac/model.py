from abc import abstractmethod
import torch.nn as nn
import torch.nn.functional as F

def get_action(rdist):
    dist = F.softmax(rdist, dim=1)
    return dist.multinomial()

class ACModel(nn.Module):
    @abstractmethod
    def forward(self, obs):
        pass

    def get_rdist_n_value(self, obs):
        return self(obs)

    def get_action_n_value(self, obs):
        rdist, value = self(obs)
        return get_action(rdist), value

    def get_rdist(self, obs):
        rdist, _ = self(obs)
        return rdist

    def get_action(self, obs):
        rdist, _ = self(obs)
        return get_action(rdist)
    
    def get_value(self, obs):
        _, value = self(obs)
        return value