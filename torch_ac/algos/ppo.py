import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo
from torch_ac.utils.batch import batchify

class PPOAlgo(BaseAlgo):
    def __init__(self, envs, acmodel, num_frames_per_proc=None, discount=0.99, lr=7e-4, gae_tau=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-5, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, num_frames_per_proc, discount, lr, gae_tau, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
    
    def update_parameters(self):
        # Collect experiences

        exps, log = self.collect_experiences()

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []

            for inds in self.batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.is_recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Compute loss

                    if self.is_recurrent:
                        dist, value, memory = self.acmodel(exps.obs[inds], memory * exps.mask[inds])
                    else:
                        dist, value = self.acmodel(exps.obs[inds])

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(exps.action[inds]) - exps.log_prob[inds])
                    surr1 = ratio * exps.advantage[inds]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * exps.advantage[inds]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = exps.value[inds] + torch.clamp(value - exps.value[inds], -self.clip_eps, self.clip_eps)
                    surr1 = (value - exps.returnn[inds]).pow(2)
                    surr2 = (value_clipped - exps.returnn[inds]).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update indexes

                    inds += 1

                    # Store states

                    if self.is_recurrent and i < self.recurrence - 1:
                        exps.memory[inds] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
        
        # Log some values

        log["entropy"] = sum(log_entropies) / len(log_entropies)
        log["value"] = sum(log_values) / len(log_values)
        log["policy_loss"] = sum(log_policy_losses) / len(log_policy_losses)
        log["value_loss"] = sum(log_value_losses) / len(log_value_losses)

        return log
    
    def batches_starting_indexes(self):
        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]
        
        return batches_indexes