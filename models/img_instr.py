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

# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class Controller(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        return x * self.weight(y) + self.bias(y)

class ACModel(nn.Module, torch_ac.ACModel):
    word_embedding_size = 32
    instr_embedding_size = 128

    def __init__(self, obs_space, action_space):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = "instr" in obs_space.keys()

        # Define instruction embedding
        if self.use_instr:
            self.word_embedding = nn.Embedding(obs_space["instr"], self.word_embedding_size)
            self.instr_rnn = nn.GRU(self.word_embedding_size, self.instr_embedding_size, batch_first=True)

        # Define actor's model
        self.a_fc1 = nn.Linear(obs_space["image"], 64)
        self.a_fc2 = nn.Linear(64, 64)
        if self.use_instr:
            self.a_controller = Controller(self.instr_embedding_size, 64)
        self.a_fc3 = nn.Linear(64, action_space.n)

        # Define critic's model
        self.c_fc1 = nn.Linear(obs_space["image"], 64)
        self.c_fc2 = nn.Linear(64, 64)
        if self.use_instr:
            self.c_controller = Controller(self.instr_embedding_size, 64)
        self.c_fc3 = nn.Linear(64, 1)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs):
        if self.use_instr:
            embed_instr = self._get_embed_instr(obs.instr)
        
        x = self.a_fc1(obs.image)
        x = F.tanh(x)
        x = self.a_fc2(x)
        if self.use_instr:
            x = self.a_controller(x, embed_instr)
        x = F.tanh(x)
        x = self.a_fc3(x)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.c_fc1(obs.image)
        x = F.tanh(x)
        x = self.c_fc2(x)
        if self.use_instr:
            x = self.c_controller(x, embed_instr)
        x = F.tanh(x)
        x = self.c_fc3(x)
        value = x.squeeze(1)

        return dist, value

    def _get_embed_instr(self, instr):
        self.instr_rnn.flatten_parameters()
        _, hidden = self.instr_rnn(self.word_embedding(instr))
        return hidden[-1]