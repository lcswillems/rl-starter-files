#!/usr/bin/env python3

import argparse
import gym
import gym_minigrid
import torch
from torch.autograd import Variable
import time

import torch_ac
import utils

# Parse arguments

parser = argparse.ArgumentParser(description='PyTorch RL example')
parser.add_argument('--env', required=True,
                    help='name of the environment to be run')
parser.add_argument('--model', required=True,
                    help='name of the pre-trained model')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
args = parser.parse_args()

# Set numpy and pytorch seeds

torch_ac.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

# Define model path

model_path = utils.get_model_path(args.model)

# Define actor-critic model

obs_space = utils.preprocess_obs_space(env.observation_space)
acmodel = utils.load_model(obs_space, env.action_space, model_path)

# Run the agent

obs = env.reset()

while True:
    time.sleep(0.5)
    renderer = env.render("human")
    print("Mission:", obs["mission"])

    obs = utils.preprocess_obss([obs], volatile=True)
    action = acmodel.get_action(obs, deterministic=True).data[0,0]
    obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()
    if renderer.window == None:
        break