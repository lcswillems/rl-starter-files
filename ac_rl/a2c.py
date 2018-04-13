import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .common import use_gpu, collect_trajectories, estimate_advantages

def train(envs, num_episodes, discount, gae_coef, entropy_reg,
          policy_net, value_net, policy_optimizer, value_optimizer):
    """collect samples"""
    transs, log = collect_trajectories(envs, policy_net, num_episodes)

    """transform samples into torch object"""
    obss = torch.from_numpy(np.stack(transs["obs"])).float()
    actions = torch.from_numpy(np.stack(transs["action"]))
    rewards = torch.from_numpy(np.stack(transs["reward"])).float()
    masks = torch.from_numpy(np.stack(transs["mask"])).float()
    if use_gpu:
        obss, actions, rewards, masks = obss.cuda(), actions.cuda(), rewards.cuda(), masks.cuda()

    """get values estimations from trajectories"""
    values = value_net(Variable(obss, volatile=True)).data

    """get advantages estimations from estimated values"""
    advantages, returns = estimate_advantages(rewards, masks, values, discount, gae_coef)

    """compute value loss"""
    pred_values = value_net(Variable(obss))
    value_loss = (pred_values - Variable(returns)).pow(2).mean()

    """update value"""
    value_loss = value_net.get_loss(Variable(obss), Variable(returns))
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    """compute policy loss"""
    raw_dists = policy_net(Variable(obss))
    log_dists = F.log_softmax(raw_dists, dim=1)
    dists = F.softmax(raw_dists, dim=1)
    entropy = -(log_dists * dists).sum(dim=1).mean()
    action_log_probs = log_dists.gather(1, Variable(actions))
    action_loss = -(action_log_probs * Variable(advantages)).mean()
    policy_loss = action_loss - entropy_reg * entropy

    """update policy"""
    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    log["value_loss"] = value_loss.data[0]
    log["action_loss"] = action_loss.data[0]
    log["entropy"] = entropy.data[0]

    return log