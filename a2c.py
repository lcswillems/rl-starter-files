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
from ac_rl.a2c import train
from ac_rl.utils import use_gpu

parser = argparse.ArgumentParser(description='PyTorch A2C example')
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
parser.add_argument('--train-iters', type=int, default=500,
                    help='number of train iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0,
                    help="interval between model saving (default: 0, 0 means no saving)")
parser.add_argument('--discount', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='learning rate (default: 7e-4)')
parser.add_argument('--gae-coef', type=float, default=0.95,
                    help='gae coefficient (default: 0.95, 1 means no gae)')
parser.add_argument('--entropy-reg', type=float, default=0.01,
                    help='entropy regularization factor (default: 0.01)')
args = parser.parse_args()

"""set numpy and pytorch seeds"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

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
# policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
# value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.lr)
policy_optimizer = torch.optim.RMSprop(policy_net.parameters(), args.lr, eps=1e-5, alpha=0.99)
value_optimizer = torch.optim.RMSprop(value_net.parameters(), args.lr, eps=1e-5, alpha=0.99)

timestep = 0

for i in range(args.train_iters):
    """train networks"""
    start_time = time.time()
    log = train(envs, args.episodes, args.discount, args.gae_coef, args.entropy_reg,
                policy_net, value_net, policy_optimizer, value_optimizer)
    end_time = time.time()

    """print logs"""
    if i % args.log_interval == 0:
        total_num_steps = sum(log["num_steps"])
        timestep += total_num_steps
        duration = time.time() - start_time

        print("Update {}, {} steps, {:.0f} FPS, min/max/median/mean returns {:.1f}/{:.1f}/{:.1f}/{:.1f}, entropy {:.3f}, value loss {:.3f}, action loss {:.3f}".
            format(i,
                   timestep,
                   total_num_steps/duration,
                   min(log["return"]),
                   max(log["return"]),
                   np.median(log["return"]),
                   np.mean(log["return"]),
                   log["entropy"],
                   log["value_loss"],
                   log["action_loss"]))

    """save networks"""
    if args.save_model_interval > 0 and i > 0 and i % args.save_model_interval == 0:
        if use_gpu:
            policy_net.cpu(), value_net.cpu()
        pickle.dump((policy_net, value_net),
                    open(os.path.join(assets_dir(), 'learned_models/{}_a2c.p'.format(args.env)), 'wb'))
        if use_gpu:
            policy_net.cuda(), value_net.cuda()