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
    instr_embedding_size = 128

    def __init__(self, obs_space, action_space):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = "instr" in obs_space.keys()

        # Define instruction embedding
        if self.use_instr:
            self.instr_rnn = nn.GRU(obs_space["instr"], self.instr_embedding_size)

        self.obs_embedding_size = obs_space["image"]
        if self.use_instr:
            self.obs_embedding_size += self.instr_embedding_size

        # Define actor's model
        self.a_fc1 = nn.Linear(self.obs_embedding_size, 64)
        self.a_fc2 = nn.Linear(64, 64)
        self.a_fc3 = nn.Linear(64, action_space.n)

        # Define critic's model
        self.c_fc1 = nn.Linear(self.obs_embedding_size, 64)
        self.c_fc2 = nn.Linear(64, 64)
        self.c_fc3 = nn.Linear(64, 1)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs):
        embedding = self._get_embedding(obs)
        
        x = self.a_fc1(embedding)
        x = F.tanh(x)
        x = self.a_fc2(x)
        x = F.tanh(x)
        x = self.a_fc3(x)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.c_fc1(embedding)
        x = F.tanh(x)
        x = self.c_fc2(x)
        x = F.tanh(x)
        x = self.c_fc3(x)
        value = x.squeeze(1)

        return dist, value

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            dist, _ = self(obs)
        if deterministic:
            action = dist.probs.max(1, keepdim=True)[1]
        else:
            action = dist.sample()
        return action

    def _get_embedding(self, obs):
        embed_image = obs["image"]
        if self.use_instr:
            embed_instr = self._get_embed_instr(obs["instr"])
            return torch.cat((embed_image, embed_instr), dim=1)
        return embed_image

    def _get_embed_instr(self, instr):
        self.instr_rnn.flatten_parameters()
        _, hidden = self.instr_rnn(instr)
        return hidden[-1]