import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ac_rl.core.common import use_gpu, collect_trajectories, estimate_advantages

def a2c_step(envs, num_episodes, discount, gae_tau, entropy_coef,
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

    """compute value loss"""
    pred_values = value_net(Variable(ts.obs))
    value_loss = (pred_values - Variable(returns)).pow(2).mean()

    """compute policy loss"""
    raw_dists = policy_net(Variable(ts.obs))
    log_dists = F.log_softmax(raw_dists, dim=1)
    dists = F.softmax(raw_dists, dim=1)
    entropy = -(log_dists * dists).sum(dim=1).mean()
    action_log_probs = log_dists.gather(1, Variable(ts.action))
    action_loss = -(action_log_probs * Variable(advantages)).mean()
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