import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ac_rl.core import use_gpu, collect_trajectories, estimate_advantages
from ac_rl.utils import split_dict_into_torch_batches

def ppo_step(envs, num_episodes, discount, gae_tau, entropy_coef,
             clip_eps, epochs, batch_size,
             policy_net, value_net, policy_optimizer, value_optimizer):
    """collect trajectories and log"""
    ts, log = collect_trajectories(envs, policy_net, num_episodes)

    """transform samples into torch object"""
    ts.obs = torch.from_numpy(np.stack(ts.obs)).float()
    ts.action = torch.from_numpy(np.stack(ts.action))
    ts.reward = torch.from_numpy(np.stack(ts.reward)).float()
    ts.mask = torch.from_numpy(np.stack(ts.mask)).float()
    if use_gpu:
        ts.obs = ts.obs.cuda()
        ts.action = ts.action.cuda()
        ts.reward = ts.reward.cuda()
        ts.mask = ts.mask.cuda()
    
    """get values estimations from trajectories"""
    values = value_net(Variable(ts.obs, volatile=True)).data

    """get advantages estimations from estimated values"""
    advantages, returns = estimate_advantages(ts.reward, ts.mask, values, discount, gae_tau)

    """normalize advantages"""
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    """get old action log probs to transitions"""
    raw_dists = policy_net(Variable(ts.obs, volatile=True))
    log_dists = F.log_softmax(raw_dists, dim=1)
    old_action_log_probs = log_dists.gather(1, Variable(ts.action, volatile=True)).data

    """add advantages, returns, old action log probs to transitions"""
    ts.advantage = advantages
    ts.returnn = returns
    ts.old_action_log_prob = old_action_log_probs

    for _ in range(epochs):
        for b in split_dict_into_torch_batches(ts, batch_size):
            """compute value loss"""
            pred_values = value_net(Variable(b.obs))
            value_loss = (pred_values - Variable(b.returnn)).pow(2).mean()

            """compute policy loss"""
            raw_dists = policy_net(Variable(b.obs))
            log_dists = F.log_softmax(raw_dists, dim=1)
            dists = F.softmax(raw_dists, dim=1)
            entropy = -(log_dists * dists).sum(dim=1).mean()
            action_log_probs = log_dists.gather(1, Variable(b.action))
            ratio = torch.exp(action_log_probs - Variable(b.old_action_log_prob))
            advantages_var = Variable(b.advantage)
            surr1 = ratio * advantages_var
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_var
            action_loss = -torch.min(surr1, surr2).mean()
            policy_loss = action_loss - entropy_coef * entropy

            """update value"""
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            """update policy"""
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
    
    """log some values"""
    log["value_loss"] = value_loss.data[0]
    log["action_loss"] = action_loss.data[0]
    log["entropy"] = entropy.data[0]
    
    return log