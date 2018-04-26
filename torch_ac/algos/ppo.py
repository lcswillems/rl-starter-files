import random
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    def __init__(self, envs, acmodel, frames_per_update=None, discount=0.99, lr=7e-4, gae_tau=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, adam_eps=1e-5, clip_eps=0.2,
                 epochs=4, batch_size=256, preprocess_obss=None, reshape_reward=None):
        frames_per_update = frames_per_update or 128

        super().__init__(envs, acmodel, frames_per_update, discount, lr, gae_tau, entropy_coef,
                         value_loss_coef, max_grad_norm, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
    
    def update_parameters(self):
        # Collect transitions

        ts, log = self.collect_transitions()

        # Add old action log probs and old values to transitions

        preprocessed_obs = self.preprocess_obss(ts.obs, use_gpu=torch.cuda.is_available())
        with torch.no_grad():
            ts.old_log_prob = self.acmodel.get_dist(preprocessed_obs).log_prob(ts.action)
            ts.old_value = self.acmodel.get_value(preprocessed_obs)

        if self.batch_size == 0:
            self.batch_size = len(ts)

        for _ in range(self.epochs):
            ts.shuffle()

            for i in range(0, len(ts), self.batch_size):
                b = ts[i:i+self.batch_size]

                # Compute loss

                preprocessed_obs = self.preprocess_obss(b.obs, use_gpu=torch.cuda.is_available())
                dist = self.acmodel.get_dist(preprocessed_obs)
                value = self.acmodel.get_value(preprocessed_obs)

                ratio = torch.exp(dist.log_prob(b.action) - b.old_log_prob)
                surr1 = ratio * b.advantage
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b.advantage
                action_loss = -torch.min(surr1, surr2).mean()

                entropy = dist.entropy().mean()

                value_clipped = b.old_value + torch.clamp(value - b.old_value, -self.clip_eps, self.clip_eps)
                surr1 = (value - b.returnn).pow(2)
                surr2 = (value_clipped - b.returnn).pow(2)
                value_loss = torch.max(surr1, surr2).mean()

                loss = action_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                # Update actor-critic

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Log some values

        log["value_loss"] = value_loss.item()
        log["action_loss"] = action_loss.item()
        log["entropy"] = entropy.item()

        return log