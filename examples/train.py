import argparse
import gym
import gym_minigrid
import os
import pickle
import time
import numpy as np
import torch

from utils import get_envs, assets_dir
from models.policy import Policy
from models.value import Value
from ac_rl import a2c_step, ppo_step, use_gpu, seed

parser = argparse.ArgumentParser(description='PyTorch RL example')
parser.add_argument('--algo', required=True,
                    help='algorithm to use: a2c | ppo')
parser.add_argument('--env', required=True,
                    help='name of the environment to be run')
parser.add_argument('--model-path',
                    help='path of pre-trained model'),
parser.add_argument('--processes', type=int, default=16,
                    help='number of processes (default: 16)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--episodes', type=int, default=16,
                    help='number of episodes before update (default: 16)')
parser.add_argument('--steps', type=int, default=500,
                    help='number of train iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0,
                    help="interval between model saving (default: 0, 0 means no saving)")
parser.add_argument('--discount', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='learning rate (default: 7e-4)')
parser.add_argument('--gae-tau', type=float, default=0.95,
                    help='gae tau (default: 0.95, 1 means no gae)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy regularization factor (default: 0.01)')
parser.add_argument('--clip-eps', type=float, default=0.2,
                    help='clipping epsilon for PPO')
parser.add_argument('--epochs', type=int, default=4,
                    help='number of epochs for PPO (default: 4)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='batch size for PPO (default: 32, 0 means all)')
args = parser.parse_args()

"""set numpy and pytorch seeds"""
seed(args.seed)

"""generate environments"""
envs = get_envs(args.env, args.seed, args.processes)

"""define policy and value networks"""
if args.model_path is None:
    policy_net = Policy(envs[0].observation_space, envs[0].action_space)
    value_net = Value(envs[0].observation_space)
else:
    policy_net, value_net = pickle.load(open(args.model_path, "rb"))
if use_gpu:
    policy_net = policy_net.cuda()
    value_net = value_net.cuda()

"""define policy and value optimizers"""
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)

timestep = 0

for i in range(args.steps):
    """train networks"""
    start_time = time.time()
    if args.algo == "a2c":
        log = a2c_step(envs, args.episodes, args.discount, args.gae_tau, args.entropy_coef,
                       policy_net, value_net, policy_optimizer, value_optimizer)
    elif args.algo == "ppo":
        log = ppo_step(envs, args.episodes, args.discount, args.gae_tau, args.entropy_coef,
                       args.clip_eps, args.epochs, args.batch_size,
                       policy_net, value_net, policy_optimizer, value_optimizer)
    else:
        raise ValueError("Invalid algorithm")
    end_time = time.time()

    """print logs"""
    if i % args.log_interval == 0:
        total_num_steps = sum(log["num_steps"])
        timestep += total_num_steps
        fps = total_num_steps/(time.time() - start_time)

        print("Update {}, {} steps, {:.0f} FPS, mean/median returns {:.1f}/{:.1f}, min/max returns {:.1f}/{:.1f}, entropy {:.3f}, value loss {:.3f}, action loss {:.3f}".
            format(i, timestep, fps,
                   np.mean(log.returnn), np.median(log.returnn), min(log.returnn), max(log.returnn),
                   log.entropy, log.value_loss, log.action_loss))

    """save models"""
    if args.save_model_interval > 0 and i > 0 and i % args.save_model_interval == 0:
        if use_gpu:
            policy_net.cpu(), value_net.cpu()
        pickle.dump((policy_net, value_net),
                    open(os.path.join(assets_dir(), 'learned_models/{}_{}.pt'.format(args.env, args.algo)), 'wb'))
        if use_gpu:
            policy_net.cuda(), value_net.cuda()