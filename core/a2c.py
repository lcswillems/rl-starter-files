import torch
from torch.autograd import Variable


def a2c_step(policy_net, value_net, optimizer_policy, optimizer_value, obss, actions, returns, advantages, l2_reg):

    """update critic"""
    value_loss = value_net.get_loss(Variable(obss), Variable(returns))
    # weight decay
    for param in value_net.parameters():
        value_loss += param.pow(2).sum() * l2_reg
    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()

    """update policy"""
    policy_loss = policy_net.get_loss(Variable(obss), Variable(actions), Variable(advantages))
    optimizer_policy.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    optimizer_policy.step()

    print("value loss {:.5f} / policy loss {:.5f}".format(value_loss.data[0], policy_loss.data[0]))