import gymnasium as gym
# from Minigridcustom.minigrid.minigrid_env import MiniGridEnv
# from Minigridcustom.minigrid.envs.custom import CustomEnv



def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env
