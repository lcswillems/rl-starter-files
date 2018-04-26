#!/usr/bin/env python3

import argparse
import gym
import gym_minigrid
import time

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="action with highest probability is selected")
args = parser.parse_args()

# Set numpy and pytorch seeds

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

# Define model name

model_name = args.model or args.env+"_"+args.algo

# Define obss preprocessor

obss_preprocessor = utils.ObssPreprocessor(model_name, env.observation_space)

# Define actor-critic model

acmodel = utils.load_model(obss_preprocessor.obs_space, env.action_space, model_name)

# Run the agent

obs = env.reset()

while True:
    time.sleep(0.1)
    renderer = env.render("human")
    print("Mission:", obs["mission"])

    preprocessed_obs = obss_preprocessor([obs])
    action = acmodel.get_action(preprocessed_obs, deterministic=args.deterministic).item()
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()

    if renderer.window is None:
        break