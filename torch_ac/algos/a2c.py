import torch
from torch.autograd import Variable
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    def __init__(self, envs, acmodel, frames_per_update=None, discount=0.99, lr=7e-4, gae_tau=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, rmsprop_alpha=0.99,
                 rmsprop_eps=1e-5, preprocess_obss=None, reshape_reward=None):
        frames_per_update = frames_per_update or 5

        super().__init__(envs, acmodel, frames_per_update, discount, lr, gae_tau, entropy_coef,
                         value_loss_coef, max_grad_norm, preprocess_obss, reshape_reward)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)
    
    def update_parameters(self):
        # Collect transitions

        ts, log = self.collect_transitions()

        # Compute loss

        preprocessed_obs = self.preprocess_obss(ts.obs, volatile=False)
        rdist = self.acmodel.get_rdist(preprocessed_obs)
        value = self.acmodel.get_value(preprocessed_obs)

        log_dist = F.log_softmax(rdist, dim=1)
        dist = F.softmax(rdist, dim=1)
        entropy = -(log_dist * dist).sum(dim=1).mean()

        action_log_prob = log_dist.gather(1, Variable(ts.action))
        action_loss = -(action_log_prob * Variable(ts.advantage)).mean()

        value_loss = (value - Variable(ts.returnn)).pow(2).mean()
        
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