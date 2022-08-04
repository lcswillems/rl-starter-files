import gym
import gym_minigrid


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env.reset(seed=seed)
    return env
