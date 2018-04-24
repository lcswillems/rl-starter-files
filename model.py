import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_ac

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

        # Decide if the instruction is taken into account
        self.use_instr = "instr" in obs_space.keys()

        # Define instruction embedding
        self.instr_embedding_size = 0
        if self.use_instr:
            self.instr_embedding_size = 128
            self.instr_gru = nn.GRU(obs_space["instr"], self.instr_embedding_size)

        # Define actor's model
        self.a_fc1 = nn.Linear(obs_space["image"] + self.instr_embedding_size, 64)
        self.a_fc2 = nn.Linear(64, 64)
        self.a_head = nn.Linear(64, action_space.n)

        # Define critic's model
        self.c_fc1 = nn.Linear(obs_space["image"] + self.instr_embedding_size, 64)
        self.c_fc2 = nn.Linear(64, 64)
        self.c_head = nn.Linear(64, 1)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def get_embed_instr(self, instr):
        self.instr_gru.flatten_parameters()
        _, hidden = self.instr_gru(instr)
        return hidden[-1]

    def get_rdist(self, obs):
        x = obs["image"]
        if self.use_instr:
            x = torch.cat((x, self.get_embed_instr(obs["instr"])), dim=1)
        x = self.a_fc1(x)
        x = F.tanh(x)
        x = self.a_fc2(x)
        x = F.tanh(x)
        return self.a_head(x)

    def get_value(self, obs):
        x = obs["image"]
        if self.use_instr:
            x = torch.cat((x, self.get_embed_instr(obs["instr"])), dim=1)
        x = self.c_fc1(x)
        x = F.tanh(x)
        x = self.c_fc2(x)
        x = F.tanh(x)
        return self.c_head(x)