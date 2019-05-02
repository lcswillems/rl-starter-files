#!/usr/bin/env python3

import argparse
import gym
import gym_minigrid
import time
import numpy as np
from array2gif import write_gif


import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent")
parser.add_argument("--gif", type=str,
                    help="store output as gif with the given filename")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)
for _ in range(args.shift):
    env.reset()

# Define agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(args.env, env.observation_space, model_dir, args.argmax)

# Run the agent

if args.gif is not None:
   frames = []

done = True

while True:
    if done:
        obs = env.reset()

    time.sleep(args.pause)
    renderer = env.render()
    if args.gif:
        frames.append(np.moveaxis(env.render("rgb_array"), 2, 0))

    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)

    if renderer.window is None:
        if args.gif:
            print("Saving gif... ",end='', flush=True)
            write_gif(np.array(frames), args.gif+".gif", fps=10)
            print("Done.")
        break
