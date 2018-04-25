from abc import abstractmethod
import torch.nn as nn
import torch.nn.functional as F

class ACModel(nn.Module):
    def forward(self):
        pass

    @abstractmethod
    def get_rdist(self, obs):
        pass

    def get_action(self, obs, deterministic=False):
        dist = F.softmax(self.get_rdist(obs), dim=1)
        if deterministic:
            return dist.max(1, keepdim=True)[1]
        return dist.multinomial(1)

    @abstractmethod
    def get_value(self, obs):
        pass