import torch
from torch.autograd import Variable


def ppo_step(policy_net, value_net, policy_optimizer, value_optimizer, optim_value_iternum, obss, actions,
             returns, advantages, fixed_log_probs, lr_mult, lr, clip_epsilon, l2_reg):

    policy_optimizer.lr = lr * lr_mult
    value_optimizer.lr = lr * lr_mult
    clip_epsilon = clip_epsilon * lr_mult

    """update critic"""
    values_target = Variable(returns)
    for _ in range(optim_value_iternum):
        values_pred = value_net(Variable(obss))
        value_loss = (values_pred - values_target).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

    """update policy"""
    advantages_var = Variable(advantages)
    log_probs = policy_net.get_log_prob(Variable(obss), Variable(actions))
    ratio = torch.exp(log_probs - Variable(fixed_log_probs))
    surr1 = ratio * advantages_var
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_var
    policy_surr = -torch.min(surr1, surr2).mean()
    policy_optimizer.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    policy_optimizer.step()
