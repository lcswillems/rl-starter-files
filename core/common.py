import torch

def estimate_advantages(rewards, masks, values, discount, tau, use_gpu):
    if use_gpu:
        rewards, masks, values = rewards.cpu(), masks.cpu(), values.cpu()
    advantages = torch.zeros(rewards.size(0), 1).float()

    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        delta = rewards[i] + discount * prev_value * masks[i] - values[i]
        advantages[i] = delta + discount * tau * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages

    if use_gpu:
        advantages, returns = advantages.cuda(), returns.cuda()
    return advantages, returns
