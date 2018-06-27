#!/usr/bin/env python3

import argparse
import gym
import time

try:
    import gym_minigrid
except ImportError:
    pass

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)

# Define agent

save_dir = utils.get_save_dir(args.model)
agent = utils.Agent(save_dir, env.observation_space, args.argmax)

# Run the agent

done = True

while True:
    if done:
        obs = env.reset()
        print("Instr:", obs["mission"])

    time.sleep(args.pause)
    renderer = env.render("human")

    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)
    agent.analyze_feedback(reward, done)

    if renderer.window is None:
        break