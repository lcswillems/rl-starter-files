#!/usr/bin/env python3

import argparse
import gym
import gym_minigrid
import time
import numpy as np

import torch_ac
import utils

# Parse arguments

parser = argparse.ArgumentParser(description="PyTorch RL example")
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on")
parser.add_argument("--model", default=None,
                    help="name of the pre-trained model")
parser.add_argument("--reset", action="store_true", default=False,
                    help="initialize model with random parameters")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--processes", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--total-frames", type=int, default=10**7,
                    help="number of frames of training (default: 10e6)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="interval between log display (default: 1)")
parser.add_argument("--save-interval", type=int, default=0,
                    help="interval between model saving (default: 0, 0 means no saving)")
parser.add_argument("--frames-per-update", type=int, default=None,
                    help="number of frames per agent during before updating parameters (default: 64)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=7e-4,
                    help="learning rate (default: 7e-4)")
parser.add_argument("--gae-tau", type=float, default=0.95,
                    help="tau coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-5,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-5)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer apha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=64,
                    help="batch size for PPO (default: 64, 0 means all)")
args = parser.parse_args()

# Set numpy and pytorch seeds

torch_ac.seed(args.seed)

# Generate environments

envs = []
for i in range(args.processes):
    env = gym.make(args.env)
    env.seed(args.seed + i)
    envs.append(env)

# Define model path

model_name = args.model or args.env+"_"+args.algo
model_path = utils.get_model_path(model_name)

# Define actor-critic model

from_path = None if args.reset else model_path
obs_space = utils.preprocess_obs_space(envs[0].observation_space)
acmodel = utils.load_model(obs_space, envs[0].action_space, from_path)
if torch_ac.use_gpu:
    acmodel = acmodel.cuda()

# Define actor-critic algo

if args.algo == "a2c":
    algo = torch_ac.A2CAlgo(envs, acmodel, args.frames_per_update, args.discount, args.lr, args.gae_tau,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.optim_alpha,
                            args.optim_eps, utils.preprocess_obss, utils.reshape_reward)
elif args.algo == "ppo":
    algo = torch_ac.PPOAlgo(envs, acmodel, args.frames_per_update, args.discount, args.lr, args.gae_tau,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.optim_eps,
                            args.clip_eps, args.epochs, args.batch_size, utils.preprocess_obss,
                            utils.reshape_reward)
else:
    raise ValueError

# Train model

total_num_frames = 0
i = 0

while total_num_frames < args.total_frames:
    # Update parameters

    start_time = time.time()
    log = algo.update_parameters()
    end_time = time.time()
    
    update_num_frames = log["total_num_frames"]
    total_num_frames += update_num_frames
    i += 1

    # Print logs

    if i % args.log_interval == 0:
        fps = update_num_frames/(end_time - start_time)

        print("tF {:06} | FPS {:.0f} | rR:x̄σmM {: .1f} {: .1f} {: .1f} {: .1f} | F:x̄σmM {:.1f} {:.1f} {:.1f} {:.1f} | H {:.3f} | vL {:.3f} | aL {: .3f}".
            format(total_num_frames, fps,
                   *utils.synthesize(log["reshaped_return"]),
                   *utils.synthesize(log["num_frames"]),
                   log["entropy"], log["value_loss"], log["action_loss"]))

    # Save model

    if args.save_interval > 0 and i % args.save_interval == 0:
        utils.save_model(acmodel, model_path)