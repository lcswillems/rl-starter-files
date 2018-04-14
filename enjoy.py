import argparse
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import torch
from torch.autograd import Variable
import time

import ac_rl
from utils import get_model_path, load_model, save_model

parser = argparse.ArgumentParser(description='PyTorch RL example')
parser.add_argument('--env', required=True,
                    help='name of the environment to be run')
parser.add_argument('--model', required=True,
                    help='name of the pre-trained model'),
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
args = parser.parse_args()

"""set numpy and pytorch seeds"""
ac_rl.seed(args.seed)

"""generate environment"""
env = gym.make(args.env)
env.seed(args.seed)
env = FlatObsWrapper(env)

"""define model path"""
model_path = get_model_path(args.model)

"""define policy and value networks"""
policy_net, value_net = load_model(env.observation_space, env.action_space, model_path)
policy_net.cpu()
value_net.cpu()

"""run the agent"""
renderer = env.render('human')
obs = env.reset()

while True:
    obs_var = Variable(torch.from_numpy(obs).float().unsqueeze(0), volatile=True)
    action = policy_net.get_action(obs_var)[0].numpy()
    obs, reward, done, _ = env.step(action)

    env.render('human')
    time.sleep(0.01)

    if done:
        env.reset()
    if renderer.window == None:
        break