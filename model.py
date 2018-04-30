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

class ACModel(torch_ac.RecurrentACModel):
    image_embedding_size = 64
    instr_embedding_size = 128
    
    @property
    def state_size(self):
        return 2*self.semi_state_size

    @property
    def semi_state_size(self):
        return self.obs_embedding_size

    def __init__(self, obs_space, action_space):
        super().__init__()

        # Decide which components are enabled
        self.use_instr = "instr" in obs_space.keys()
        self.use_memory = False

        # Define image embedding
        self.image_fc1 = nn.Linear(obs_space["image"], 64)
        self.image_fc2 = nn.Linear(64, self.image_embedding_size)

        # Define instruction embedding
        if self.use_instr:
            self.instr_rnn = nn.GRU(obs_space["instr"], self.instr_embedding_size)

        # Define observation embedding size
        self.obs_embedding_size = self.image_embedding_size
        if self.use_instr:
            self.obs_embedding_size += self.instr_embedding_size

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.obs_embedding_size, self.semi_state_size)

        # Define actor's model
        self.a_fc1 = nn.Linear(self.obs_embedding_size, 64)
        self.a_head = nn.Linear(64, action_space.n)

        # Define critic's model
        self.c_fc1 = nn.Linear(self.obs_embedding_size, 64)
        self.c_head = nn.Linear(64, 1)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs, state):
        embedding = self._get_embedding(obs)
        if self.use_memory:
            hidden = (state[:, :self.semi_state_size], state[:, self.semi_state_size:])
            hidden = self.memory_rnn(embedding, hidden)
            embedding = hidden[0]
            state = torch.cat(hidden, dim=1)
        dist = self._get_dist_from_embedding(embedding)
        value = self._get_value_from_embedding(embedding)
        return dist, value, state

    def get_action(self, obs, state, deterministic=False):
        with torch.no_grad():
            dist, _, state = self(obs, state)
        if deterministic:
            action = dist.probs.max(1, keepdim=True)[1]
        else:
            action = dist.sample()
        return action, state

    def _get_embedding(self, obs):
        embed_image = self._get_embed_image(obs["image"])
        if self.use_instr:
            embed_instr = self._get_embed_instr(obs["instr"])
            return torch.cat((embed_image, embed_instr), dim=1)
        return embed_image

    def _get_dist_from_embedding(self, embedding):
        x = self.a_fc1(embedding)
        x = F.tanh(x)
        x = self.a_head(x)
        return Categorical(logits=F.log_softmax(x, dim=1))

    def _get_value_from_embedding(self, embedding):
        x = self.c_fc1(embedding)
        x = F.tanh(x)
        return self.c_head(x)

    def _get_embed_image(self, image):
        x = self.image_fc1(image)
        x = F.tanh(x)
        return self.image_fc2(x)

    def _get_embed_instr(self, instr):
        self.instr_rnn.flatten_parameters()
        _, hidden = self.instr_rnn(instr)
        return hidden[-1]