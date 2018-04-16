from abc import ABC, abstractmethod
import torch
from torch.autograd import Variable
import numpy as np

from ac_rl.utils import use_gpu, dictlist, MultiEnv

class BaseAlgo(ABC):
    def __init__(self, envs, num_step_frames, acmodel,
                 discount, lr, gae_tau, entropy_coef, value_loss_coef, max_grad_norm):
        self.env = MultiEnv(envs)
        self.num_step_frames = num_step_frames
        self.acmodel = acmodel
        self.discount = discount
        self.lr = lr
        self.gae_tau = gae_tau
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.num_processes = len(envs)
        self.next_obs = self.env.reset()
    
    def collect_transitions(self):
        ts = dictlist()

        """add obs, action, reward, mask and value to transitions"""
        log_episode_return = np.zeros(self.num_processes)
        log_return = np.zeros(self.num_processes)

        for _ in range(self.num_step_frames):
            obs = torch.from_numpy(np.array(self.next_obs)).float()
            action, value = self.acmodel.get_action_n_value(Variable(obs, volatile=True))
            action = action.data.squeeze(1).cpu().numpy()
            value = value.data.squeeze(1).cpu().numpy()
            next_obs, reward, done, _ = self.env.step(action)
            mask = [0 if done_ else 1 for done_ in done]
            ts.append({"obs": self.next_obs, "action": action, "reward": reward, "mask": mask, "value": value})
            
            self.next_obs = next_obs

            reward = np.array(reward)
            mask = np.array(mask)
            log_episode_return += reward
            log_return *= mask
            log_return += (1 - mask) * log_episode_return
            log_episode_return *= mask

        ts.obs = torch.from_numpy(np.array(ts.obs)).float()
        ts.action = torch.from_numpy(np.array(ts.action))
        ts.reward = torch.from_numpy(np.array(ts.reward)).float()
        ts.mask = torch.from_numpy(np.array(ts.mask)).float()
        ts.value = torch.from_numpy(np.array(ts.value)).float()
        if use_gpu:
            ts.obs = ts.obs.cuda()
            ts.action = ts.action.cuda()
            ts.reward = ts.reward.cuda()
            ts.mask = ts.mask.cuda()
            ts.value = ts.value.cuda()
        
        """add advantage and return to transitions"""
        ts.advantage = torch.zeros(*ts.reward.shape).float()
        if use_gpu:
            ts.advantage = ts.advantage.cuda()

        obs = torch.from_numpy(np.array(self.next_obs)).float()
        next_value = self.acmodel.get_value(Variable(obs, volatile=True)).data.squeeze(1)

        for i in reversed(range(self.num_step_frames)):
            next_value = ts.value[i+1] if i < self.num_step_frames - 1 else next_value
            next_advantage = ts.advantage[i+1] if i < self.num_step_frames - 1 else 0
            
            delta = ts.reward[i] + self.discount * next_value * ts.mask[i] - ts.value[i]
            ts.advantage[i] = delta + self.discount * self.gae_tau * next_advantage * ts.mask[i]

        ts.returnn = ts.advantage + ts.value

        """reshape each transitions attribute"""
        ts.obs = ts.obs.view(-1, *ts.obs.shape[2:])
        ts.action = ts.action.view(-1, *ts.action.shape[2:]).unsqueeze(1)
        ts.reward = ts.reward.view(-1, *ts.reward.shape[2:]).unsqueeze(1)
        ts.mask = ts.mask.view(-1, *ts.mask.shape[2:]).unsqueeze(1)
        ts.value = ts.value.view(-1, *ts.value.shape[2:]).unsqueeze(1)
        ts.advantage = ts.advantage.view(-1, *ts.advantage.shape[2:]).unsqueeze(1)
        ts.returnn = ts.returnn.view(-1, *ts.returnn.shape[2:]).unsqueeze(1)

        """log some values"""
        log = {"return": log_return}

        return ts, log

    @abstractmethod
    def step(self):
        pass