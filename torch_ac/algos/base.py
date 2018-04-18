from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable
import numpy as np

from torch_ac.preprocess import default_preprocess_obss, default_preprocess_reward
from torch_ac.utils import use_gpu, DictList, MultiEnv

class BaseAlgo(ABC):
    def __init__(self, envs, acmodel, frames_per_update, discount, lr, gae_tau, entropy_coef,
                 value_loss_coef, max_grad_norm, preprocess_obss=None, preprocess_reward=None):
        self.env = MultiEnv(envs)
        self.acmodel = acmodel
        self.frames_per_update = frames_per_update
        self.discount = discount
        self.lr = lr
        self.gae_tau = gae_tau
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.preprocess_obss = preprocess_obss if preprocess_obss != None else default_preprocess_obss
        self.preprocess_reward = preprocess_reward if preprocess_reward != None else default_preprocess_reward

        self.num_processes = len(envs)
        self.obs = self.env.reset()
    
    def collect_transitions(self):
        ts = DictList()

        # Add obs, action, reward, mask and value to transitions

        log_episode_return = np.zeros(self.num_processes)
        log_return = np.zeros(self.num_processes)

        for _ in range(self.frames_per_update):
            obs = self.preprocess_obss(self.obs, volatile=True)
            action = self.acmodel.get_action(obs)
            value = self.acmodel.get_value(obs)
            action = action.data.squeeze(1).cpu().numpy()
            value = value.data.squeeze(1).cpu().numpy()
            obs, reward, done, _ = self.env.step(action)
            reward = [
                self.preprocess_reward(obs_, action_, reward_)
                for obs_, action_, reward_ in zip(obs, action, reward)
            ]
            mask = [0 if done_ else 1 for done_ in done]
            ts.append({"obs": self.obs, "action": action, "reward": reward, "mask": mask, "value": value})
            
            self.obs = obs

            reward = np.array(reward)
            mask = np.array(mask)
            log_episode_return += reward
            log_return *= mask
            log_return += (1 - mask) * log_episode_return
            log_episode_return *= mask

        ts.action = torch.from_numpy(np.array(ts.action))
        ts.reward = torch.from_numpy(np.array(ts.reward)).float()
        ts.mask = torch.from_numpy(np.array(ts.mask)).float()
        ts.value = torch.from_numpy(np.array(ts.value)).float()
        if use_gpu:
            ts.action = ts.action.cuda()
            ts.reward = ts.reward.cuda()
            ts.mask = ts.mask.cuda()
            ts.value = ts.value.cuda()
        
        # Add advantage and return to transitions

        ts.advantage = torch.zeros(*ts.reward.shape).float()
        if use_gpu:
            ts.advantage = ts.advantage.cuda()

        obs = self.preprocess_obss(self.obs, volatile=True)
        next_value = self.acmodel.get_value(obs).data.squeeze(1)

        for i in reversed(range(self.frames_per_update)):
            next_value = ts.value[i+1] if i < self.frames_per_update - 1 else next_value
            next_advantage = ts.advantage[i+1] if i < self.frames_per_update - 1 else 0
            
            delta = ts.reward[i] + self.discount * next_value * ts.mask[i] - ts.value[i]
            ts.advantage[i] = delta + self.discount * self.gae_tau * next_advantage * ts.mask[i]

        ts.returnn = ts.advantage + ts.value

        # Reshape each transitions attribute

        ts.obs = [obs for obss in ts.obs for obs in obss]
        ts.action = ts.action.view(-1, *ts.action.shape[2:]).unsqueeze(1)
        ts.reward = ts.reward.view(-1, *ts.reward.shape[2:]).unsqueeze(1)
        ts.mask = ts.mask.view(-1, *ts.mask.shape[2:]).unsqueeze(1)
        ts.value = ts.value.view(-1, *ts.value.shape[2:]).unsqueeze(1)
        ts.advantage = ts.advantage.view(-1, *ts.advantage.shape[2:]).unsqueeze(1)
        ts.returnn = ts.returnn.view(-1, *ts.returnn.shape[2:]).unsqueeze(1)

        # Log some values

        log = {"return": log_return}

        return ts, log

    @abstractmethod
    def update_parameters(self):
        pass