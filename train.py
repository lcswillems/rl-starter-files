import argparse
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import time
import numpy as np
import torch

from utils import get_model_path, load_model, save_model
import ac_rl

parser = argparse.ArgumentParser(description='PyTorch RL example')
parser.add_argument('--algo', required=True,
                    help='algorithm to use: a2c | ppo')
parser.add_argument('--env', required=True,
                    help='name of the environment to be run')
parser.add_argument('--model', default=None,
                    help='name of the pre-trained model'),
parser.add_argument('--reset', action='store_true', default=False,
                    help='start from a new model')
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

assert args.algo in ["a2c", "ppo"]

"""set numpy and pytorch seeds"""
ac_rl.seed(args.seed)

"""generate environments"""
envs = []
for i in range(args.processes):
    env = gym.make(args.env)
    env.seed(args.seed + i)
    env = FlatObsWrapper(env)
    envs.append(env)

"""define model path"""
model_path = get_model_path(args.env, args.algo, args.model)

"""define policy and value networks"""
from_path = None if args.reset else model_path
policy_net, value_net = load_model(envs[0].observation_space, envs[0].action_space, from_path)

"""define policy and value optimizers"""
policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)

timestep = 0

for i in range(args.steps):
    """train networks"""
    start_time = time.time()
    if args.algo == "a2c":
        log = ac_rl.a2c_step(envs, args.episodes, args.discount, args.gae_tau, args.entropy_coef,
                       policy_net, value_net, policy_optimizer, value_optimizer)
    elif args.algo == "ppo":
        log = ac_rl.ppo_step(envs, args.episodes, args.discount, args.gae_tau, args.entropy_coef,
                       args.clip_eps, args.epochs, args.batch_size,
                       policy_net, value_net, policy_optimizer, value_optimizer)
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
        save_model(policy_net, value_net, model_path)