import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    def __init__(self, envs, acmodel, frames_per_update=50, discount=0.99, lr=7e-4, gae_tau=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, adam_eps=1e-5, clip_eps=0.2,
                 epochs=4, batch_size=32, preprocess_obss=None, reshape_reward=None):
        super().__init__(envs, acmodel, frames_per_update, discount, lr, gae_tau, entropy_coef,
                         value_loss_coef, max_grad_norm, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
    
    def update_parameters(self):
        # Collect transitions

        ts, log = self.collect_transitions()       

        # Add old action log probs to transitions

        rdist = self.acmodel.get_rdist(self.preprocess_obss(ts.obs, volatile=True))
        log_dist = F.log_softmax(rdist, dim=1)
        ts.old_action_log_prob = log_dist.gather(1, Variable(ts.action, volatile=True)).data

        for _ in range(self.epochs):
            # random.shuffle(ts)

            for i in range(0, len(ts), self.batch_size):
                b = ts[i:i+self.batch_size]

                # Compute loss

                obs = self.preprocess_obss(b.obs, volatile=False)
                rdist = self.acmodel.get_rdist(obs)
                value = self.acmodel.get_value(obs)

                log_dist = F.log_softmax(rdist, dim=1)
                dist = F.softmax(rdist, dim=1)
                entropy = -(log_dist * dist).sum(dim=1).mean()

                action_log_prob = log_dist.gather(1, Variable(b.action))
                ratio = torch.exp(action_log_prob - Variable(b.old_action_log_prob))
                advantage_var = Variable(b.advantage)
                surr1 = ratio * advantage_var
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage_var
                action_loss = -torch.min(surr1, surr2).mean()

                value_loss = (value - Variable(b.returnn)).pow(2).mean()

                loss = action_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                # Update actor-critic

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Log some values

        log["value_loss"] = value_loss.data[0]
        log["action_loss"] = action_loss.data[0]
        log["entropy"] = entropy.data[0]

        return log