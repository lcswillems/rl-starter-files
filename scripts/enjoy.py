#!/usr/bin/env python3

import argparse
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import torch
from torch.autograd import Variable
import time

import torch_ac
from utils import get_model_path, load_model, save_model

# Parse arguments

parser = argparse.ArgumentParser(description='PyTorch RL example')
parser.add_argument('--env', required=True,
                    help='name of the environment to be run')
parser.add_argument('--model', required=True,
                    help='name of the pre-trained model'),
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
args = parser.parse_args()

# Set numpy and pytorch seeds

torch_ac.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)
env = FlatObsWrapper(env)

# Define model path

model_path = get_model_path(args.model)

# Define actor-critic model

acmodel = load_model(env.observation_space, env.action_space, model_path)

# Run the agent

renderer = env.render('human')
obs = env.reset()

while True:
    obs = torch.from_numpy(obs).float().unsqueeze(0)
    action = acmodel.get_action(Variable(obs, volatile=True)).data[0].numpy()
    obs, reward, done, _ = env.step(action)

    env.render('human')
    time.sleep(0.05)

    if done:
        env.reset()
    if renderer.window == None:
        break