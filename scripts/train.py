#!/usr/bin/env python3

import argparse
import gym
import gym_minigrid
import time
import datetime
import numpy as np
import sys
import torch
import torch_ac

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the pre-trained model (default: ENV_ALGO)")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--processes", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--total-frames", type=int, default=10**7,
                    help="number of frames of training (default: 10e7)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="interval between log display (default: 1)")
parser.add_argument("--save-interval", type=int, default=0,
                    help="interval between model saving (default: 0, 0 means no saving)")
parser.add_argument("--frames-per-update", type=int, default=None,
                    help="number of frames per agent before updating parameters (default: 64)")
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
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256, 0 means all)")
args = parser.parse_args()

# Set numpy and pytorch seeds

torch_ac.seed(args.seed)

# Generate environments

envs = []
for i in range(args.processes):
    env = gym.make(args.env)
    env.seed(args.seed + i)
    envs.append(env)

# Define model name

model_name = args.model or args.env+"_"+args.algo

# Define obss preprocessor

obss_preprocessor = utils.ObssPreprocessor(model_name, envs[0].observation_space)

# Define actor-critic model

acmodel = utils.load_model(obss_preprocessor.obs_space, envs[0].action_space, model_name)
if torch.cuda.is_available():
    acmodel.cuda()

# Define actor-critic algo

if args.algo == "a2c":
    algo = torch_ac.A2CAlgo(envs, acmodel, args.frames_per_update, args.discount, args.lr, args.gae_tau,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.optim_alpha,
                            args.optim_eps, obss_preprocessor)
elif args.algo == "ppo":
    algo = torch_ac.PPOAlgo(envs, acmodel, args.frames_per_update, args.discount, args.lr, args.gae_tau,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.optim_eps,
                            args.clip_eps, args.epochs, args.batch_size, obss_preprocessor)
else:
    raise ValueError

# Define logger and log command and model

suffix = datetime.datetime.now().strftime("%y%m%d%H%M%S")
logger = utils.Logger(model_name+"_"+suffix)
logger.log(" ".join(sys.argv), to_print=False)
logger.log(acmodel)

# Train model

total_num_frames = 0
total_start_time = time.time()
i = 0

while total_num_frames < args.total_frames:
    # Update parameters

    update_start_time = time.time()
    log = algo.update_parameters()
    update_end_time = time.time()
    
    update_num_frames = log["total_num_frames"]
    total_num_frames += update_num_frames
    i += 1

    # Print logs

    if i % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = update_num_frames/(update_end_time - update_start_time)

        logger.log(
            "U {} | tF {:06} | FPS {:04.0f} | D {} | rR:x̄σmM {: .2f} {: .2f} {: .2f} {: .2f} | F:x̄σmM {:.1f} {:.1f} {:.1f} {:.1f} | H {:.3f} | vL {:.3f} | aL {: .3f}"
            .format(i, total_num_frames, fps,
                    datetime.timedelta(seconds=total_ellapsed_time),
                    *utils.synthesize(log["reshaped_return"]),
                    *utils.synthesize(log["num_frames"]),
                    log["entropy"], log["value_loss"], log["action_loss"]))

    # Save model

    if args.save_interval > 0 and i % args.save_interval == 0:
        if torch.cuda.is_available():
            acmodel.cpu()
        utils.save_model(acmodel, model_name)
        if torch.cuda.is_available():
            acmodel.cuda()