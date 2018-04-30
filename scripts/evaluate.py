#!/usr/bin/env python3

import argparse
import gym
import gym_minigrid
import time
import datetime
import torch

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=2,
                    help="random seed (default: 2)")
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

# Initialize logs

log = {"num_frames": [], "return": []}

# Run the agent

start_time = time.time()

for _ in range(args.episodes):
    obs = env.reset()
    state = torch.zeros(1, acmodel.state_size)
    done = False

    num_frames = 0
    returnn = 0

    while not(done):
        preprocessed_obs = obss_preprocessor([obs])
        action, state = acmodel.get_action(preprocessed_obs, state, deterministic=args.deterministic)
        obs, reward, done, _ = env.step(action.item())
        
        num_frames += 1
        returnn += reward
    
    log["num_frames"].append(num_frames)
    log["return"].append(returnn)

end_time = time.time()

# Print logs

num_frames = sum(log["num_frames"])
fps = num_frames/(end_time - start_time)
ellapsed_time = int(end_time - start_time)

print("F {} | FPS {:.0f} | D {} | R:x̄σmM {:.2f} {:.2f} {:.2f} {:.2f} | F:x̄σmM {:.1f} {:.1f} {:.1f} {:.1f}"
      .format(num_frames, fps,
              datetime.timedelta(seconds=ellapsed_time),
              *utils.synthesize(log["return"]),
              *utils.synthesize(log["num_frames"])))