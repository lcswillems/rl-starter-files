import torch

import utils

class Agent:
    """An abstraction of the behavior of an agent. The agent is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, save_dir, observation_space, argmax=False):
        self.preprocess_obss = utils.ObssPreprocessor(save_dir, observation_space)
        self.model = utils.load_model(save_dir)
        self.argmax = argmax

        if self.model.recurrent:
            self._initialize_memory()

    def _initialize_memory(self):
        self.memory = torch.zeros(1, self.model.memory_size)

    def get_action(self, obs):
        preprocessed_obss = self.preprocess_obss([obs])

        with torch.no_grad():
            if self.model.recurrent:
                dist, _, self.memory = self.model(preprocessed_obss, self.memory)
            else:
                dist, _ = self.model(preprocessed_obss)

        if self.argmax:
            action = dist.probs.max(1, keepdim=True)[1]
        else:
            action = dist.sample()

        return action.item()

    def analyze_feedback(self, reward, done):
        if done and self.model.recurrent:
            self._initialize_memory()