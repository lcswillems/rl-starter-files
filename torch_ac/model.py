from abc import abstractmethod
import torch.nn as nn
import torch.nn.functional as F

class ACModel(nn.Module):
    def forward(self):
        pass

    @abstractmethod
    def get_dist(self, obs):
        pass

    @abstractmethod
    def get_action(self, obs, deterministic=False):
        pass

    @abstractmethod
    def get_value(self, obs):
        pass