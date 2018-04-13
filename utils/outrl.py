from os import path
import torch
from gym_minigrid.wrappers import *

def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))

def get_envs(env_name, seed, nbs):
    envs = []
    for i in range(nbs):
        env = gym.make(env_name)
        env.seed(seed + i)
        env = FlatObsWrapper(env)
        envs.append(env)
    return envs