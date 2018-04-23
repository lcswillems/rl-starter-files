#!/usr/bin/env python3

import argparse
import gym
import gym_minigrid
import time

import torch_ac
import utils

# Parse arguments

parser = argparse.ArgumentParser(description="PyTorch RL example")
parser.add_argument("--env", required=True,
                    help="name of the environment to be run")
parser.add_argument("--model", required=True,
                    help="name of the trained model")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected")
args = parser.parse_args()

# Set numpy and pytorch seeds

torch_ac.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

# Define actor-critic model

obs_space = utils.preprocess_obs_space(env.observation_space)
model_path = utils.get_model_path(args.model)
acmodel = utils.load_model(obs_space, env.action_space, model_path)

# Run the agent

obs = env.reset()

while True:
    time.sleep(0.1)
    renderer = env.render("human")
    print("Mission:", obs["mission"])

    preprocessed_obs = utils.preprocess_obss([obs], volatile=True)
    action = acmodel.get_action(preprocessed_obs, deterministic=args.deterministic).data[0,0]
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()

    if renderer.window is None:
        break