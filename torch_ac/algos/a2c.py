import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_tau=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-5, preprocess_obss=None, reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_tau, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)
    
    def update_parameters(self):
        # Collect experiences

        exps, log = self.collect_experiences()

        # Compute starting indexes

        inds = self.starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        memory = exps.memory[inds]

        for _ in range(self.recurrence):
            # Compute loss

            if self.is_recurrent:
                dist, value, memory = self.acmodel(exps.obs[inds], memory * exps.mask[inds])
            else:
                dist, value = self.acmodel(exps.obs[inds])

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(exps.action[inds]) * exps.advantage[inds]).mean()

            value_loss = (value - exps.returnn[inds]).pow(2).mean()
            
            loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

            # Update indexes

            inds += 1

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        log["entropy"] = update_entropy
        log["value"] = update_value
        log["policy_loss"] = update_policy_loss
        log["value_loss"] = update_value_loss

        return log
    
    def starting_indexes(self):
        return numpy.arange(0, self.num_frames, self.recurrence)