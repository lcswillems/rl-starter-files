from abc import ABC, abstractmethod
import torch
import numpy

from torch_rl.format import default_preprocess_obss
from torch_rl.model import RecurrentACModel
from torch_rl.utils import DictList, MultiEnv

class BaseAlgo(ABC):
    def __init__(self, envs, acmodel, num_frames_per_proc, discount, lr, gae_tau, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward):
        # Store parameters

        self.env = MultiEnv(envs)
        self.acmodel = acmodel
        self.acmodel.train()
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_tau = gae_tau
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        # Store helpers values

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        self.is_recurrent = isinstance(self.acmodel, RecurrentACModel)

        # Control parameters

        if not(self.is_recurrent):
            self.recurrence = 1

        assert self.num_frames_per_proc % self.recurrence == 0

        # Store experiences values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        if self.is_recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Store log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)
        self.log_return = torch.zeros(self.num_procs, device=self.device)
        self.log_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_num_frames = torch.zeros(self.num_procs, device=self.device)

    def collect_experiences(self):        
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.is_recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()

            obs, reward, done, _ = self.env.step(action.cpu().numpy())
            
            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            if self.is_recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            self.log_return *= self.mask
            self.log_return += (1 - self.mask) * self.log_episode_return
            self.log_reshaped_return *= self.mask
            self.log_reshaped_return += (1 - self.mask) * self.log_episode_reshaped_return
            self.log_num_frames *= self.mask
            self.log_num_frames += (1 - self.mask) * self.log_episode_num_frames

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.is_recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            
            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_tau * next_advantage * next_mask

        # Defines experiences

        exps = DictList()
        exps.obs = [obs for obss in self.obss for obs in obss]
        if self.is_recurrent:
            exps.memory = self.memories.view(-1, *self.memories.shape[2:])
            exps.mask = self.masks.view(-1, *self.masks.shape[2:]).unsqueeze(1)
        exps.action = self.actions.view(-1, *self.actions.shape[2:])
        exps.value = self.values.view(-1, *self.values.shape[2:])
        exps.reward = self.rewards.view(-1, *self.rewards.shape[2:])
        exps.advantage = self.advantages.view(-1, *self.advantages.shape[2:])
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.view(-1, *self.log_probs.shape[2:])

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)        

        # Log some values

        log = {
            "return_per_episode": self.log_return.cpu().numpy(),
            "reshaped_return_per_episode": self.log_reshaped_return.cpu().numpy(),
            "num_frames_per_episode": self.log_num_frames.cpu().numpy(),
            "num_frames": self.num_frames
        }

        return exps, log

    @abstractmethod
    def update_parameters(self):
        pass