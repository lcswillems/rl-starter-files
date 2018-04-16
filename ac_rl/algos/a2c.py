import torch
from torch.autograd import Variable
import torch.nn.functional as F

from ac_rl.algos.base import BaseAlgo

class A2CAlgo(BaseAlgo):
    def __init__(self, envs, num_step_frames, acmodel,
                 discount, lr, gae_tau, entropy_coef, value_loss_coef, max_grad_norm,
                 rmsprop_alpha, rmsprop_eps):
        super().__init__(envs, num_step_frames, acmodel,
                         discount, lr, gae_tau, entropy_coef, value_loss_coef, max_grad_norm)

        self.optimizer = torch.optim.RMSprop(self.acmodel.parameters(), lr,
                                             alpha=rmsprop_alpha, eps=rmsprop_eps)
    
    def step(self):
        # Collect transitions

        ts, log = self.collect_transitions()

        # Compute loss

        rdist, value = self.acmodel.get_rdist_n_value(Variable(ts.obs))

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