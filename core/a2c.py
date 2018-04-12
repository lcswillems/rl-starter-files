import torch
from torch.autograd import Variable


def a2c_step(policy_net, value_net, policy_optimizer, value_optimizer, obss, actions, returns, advantages):

    """update critic"""
    value_loss = value_net.get_loss(Variable(obss), Variable(returns))
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    """update policy"""
    policy_loss = policy_net.get_loss(Variable(obss), Variable(actions), Variable(advantages))
    policy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
    policy_optimizer.step()

    print("value loss {:.5f} / policy loss {:.5f}".format(value_loss.data[0], policy_loss.data[0]))