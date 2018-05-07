from abc import abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F

class ACModel:
    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass
        
    @abstractmethod
    def forward(self, obs):
        pass

class RecurrentACModel(ACModel):
    @abstractmethod
    def forward(self, obs, memory):
        pass
    
    @property
    @abstractmethod
    def memory_size(self):
        pass