import torch

import utils

class Agent:
    def __init__(self, model_name, observation_space, deterministic=False):
        self.obss_preprocessor = utils.ObssPreprocessor(model_name, observation_space)
        self.model = utils.load_model(model_name)
        self.deterministic = deterministic

        if self.model.recurrent:
            self._initialize_memory()
    
    def _initialize_memory(self):
        self.memory = torch.zeros(1, self.model.memory_size)

    def get_action(self, obs):
        preprocessed_obs = self.obss_preprocessor([obs])

        with torch.no_grad():
            if self.model.recurrent:
                dist, _, self.memory = self.model(preprocessed_obs, self.memory)
            else:
                dist, _ = self.model(preprocessed_obs)
        
        if self.deterministic:
            action = dist.probs.max(1, keepdim=True)[1]
        else:
            action = dist.sample()
        
        return action.item()
    
    def analyze_feedback(self, reward, done):
        if done and self.model.recurrent:
            self._initialize_memory()