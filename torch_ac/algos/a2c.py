import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_tau=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 5

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_tau, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)
    
    def update_parameters(self):
        # Collect experiences

        exps, log = self.collect_experiences()

        # Compute loss

        if self.is_recurrent:
            dist, value, _ = self.acmodel(exps.obs, exps.memory * exps.mask)
        else:
            dist, value = self.acmodel(exps.obs)

        entropy = dist.entropy().mean()

        policy_loss = -(dist.log_prob(exps.action) * exps.advantage).mean()

        value_loss = (value - exps.returnn).pow(2).mean()
        
        loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

        # Update actor-critic

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        log["entropy"] = entropy.mean().item()
        log["value"] = value.mean().item()
        log["policy_loss"] = policy_loss.item()
        log["value_loss"] = value_loss.item()

        return log