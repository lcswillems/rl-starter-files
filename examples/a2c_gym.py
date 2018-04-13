import argparse
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import os
import sys
import pickle
import time
import numpy as np
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.policy import Policy
from models.value import Value
from core.a2c import train
from core.common import use_gpu, estimate_advantages

parser = argparse.ArgumentParser(description='PyTorch A2C example')
parser.add_argument('--env', required=True,
                    help='name of the environment to run')
parser.add_argument('--model-path',
                    help='path of pre-trained model'),
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--num-threads', type=int, default=4,
                    help='number of threads (default: 4)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-episodes', type=int, default=4,
                    help='number of episodes per update (default: 4)')
parser.add_argument('--num-main-iter', type=int, default=500,
                    help='number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1,
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=0,
                    help="interval between saving model (default: 0, 0 means don't save)")
parser.add_argument('--discount', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='learning rate (default: 7e-4)')
parser.add_argument('--gae-coef', type=float, default=0.95,
                    help='gae parameter (default: 0.95, 1 means no gae)')
parser.add_argument('--entropy-reg', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
args = parser.parse_args()

"""set numpy and pytorch seeds"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)

"""generate environments"""
envs = get_envs(args.env, args.seed, args.num_threads)

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

for i in range(args.num_main_iter):
    """train networks"""
    start_time = time.time()
    log = train(envs, args.num_episodes, args.discount, args.gae_coef, args.entropy_reg,
                policy_net, value_net, policy_optimizer, value_optimizer)
    end_time = time.time()

    """print logs"""
    if i % args.log_interval == 0:
        total_num_steps = sum(log["num_steps"])
        timestep += total_num_steps
        duration = time.time() - start_time

        print("Update {}, {} steps, {:.0f} FPS, min/max/median returns {:.2f}/{:.2f}/{:.2f}, entropy {:.3f}, value loss {:.3f}, policy loss {:.3f}".
            format(i,
                   timestep,
                   total_num_steps/duration,
                   min(log["return"]),
                   max(log["return"]),
                   np.median(log["return"]),
                   log["entropy"],
                   log["value_loss"],
                   log["policy_loss"]))

    """save networks"""
    if args.save_model_interval > 0 and i > 0 and i % args.save_model_interval == 0:
        if use_gpu:
            policy_net.cpu(), value_net.cpu()
        pickle.dump((policy_net, value_net),
                    open(os.path.join(assets_dir(), 'learned_models/{}_a2c.p'.format(args.env)), 'wb'))
        if use_gpu:
            policy_net.cuda(), value_net.cuda()