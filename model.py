import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_ac

class ACModel(torch_ac.ACModel):
    def __init__(self, obs_space, action_space):
        super().__init__()

        # Decide if the instruction is taken into account
        self.forget_instr = True

        # Define sizes of some layers
        self.instr_embedding_size = 10
        self.obs_embedding_size = obs_space["image"]
        if not(self.forget_instr):
            self.obs_embedding_size += self.instr_embedding_size

        # Define instruction embedding model
        if not(self.forget_instr):
            self.instr_gru = nn.GRU(obs_space["instr"], self.instr_embedding_size)

        # Define actor's model
        self.a_fc1 = nn.Linear(self.obs_embedding_size, 128)
        self.a_fc2 = nn.Linear(128, 128)
        self.a_head = nn.Linear(128, action_space.n)

        # Define critic's model
        self.c_fc1 = nn.Linear(self.obs_embedding_size, 128)
        self.c_fc2 = nn.Linear(128, 128)
        self.c_head = nn.Linear(128, 1)

        # Reduce values of critic's head layer for increasing entropy
        self.a_head.weight.data.mul_(0.1)
        self.a_head.bias.data.mul_(0.0)

    def get_embed_instr(self, instr):
        _, hidden = self.instr_gru(instr)
        return hidden[-1]
    
    def get_embed_obs(self, obs):
        if self.forget_instr:
            return obs["image"]
        return torch.cat((obs["image"], self.get_embed_instr(obs["instr"])), dim=1)

    def get_rdist(self, obs):
        x = self.get_embed_obs(obs)
        x = self.a_fc1(x)
        x = F.tanh(x)
        x = self.a_fc2(x)
        x = F.tanh(x)
        return self.a_head(x)

    def get_value(self, obs):
        x = self.get_embed_obs(obs)
        x = self.c_fc1(x)
        x = F.tanh(x)
        x = self.c_fc2(x)
        x = F.tanh(x)
        return self.c_head(x)