# largely inspired by https://github.com/lcswillems/rl-starter-files

# import random 
# import time

from scripts.train import train
from utils.format import train_arg_parser

from model import ACModel
from phasicdoorkey import PhasicDoorKeyEnv
# from torch_ac import PPOAlgo

num_episodes = 100

# env_p1_l means the environment in phase 1 with the door locked, _u is unlocked
env_p1_l = PhasicDoorKeyEnv(phase=1, door_locked=True, size=7, max_steps=100, render_mode="rgb_array")
env_p1_u = PhasicDoorKeyEnv(phase=1, door_locked=False, size=7, max_steps=100, render_mode="rgb_array")
env_p2_l = PhasicDoorKeyEnv(phase=2, door_locked=True, size=7, max_steps=100, render_mode="rgb_array")
env_p2_u = PhasicDoorKeyEnv(phase=2, door_locked=False, size=7, max_steps=100, render_mode="rgb_array")
env_p3_l = PhasicDoorKeyEnv(phase=3, door_locked=True, size=7, max_steps=100, render_mode="rgb_array")
env_p3_u = PhasicDoorKeyEnv(phase=3, door_locked=False, size=7, max_steps=100, render_mode="rgb_array")

envs = [env_p1_l, env_p1_u, env_p2_l, env_p2_u, env_p3_l, env_p3_u]

# how they do it in original repo
# obs_space = {"image": envs[0].observation_space['image'].shape} 

# agent = ACModel(obs_space, envs[0].action_space, use_memory=False, use_text=False)

# train on first two envs
train_args = train_arg_parser("ppo", envs[:2], model="model_v0.0", frames=1e6)

train(train_args)