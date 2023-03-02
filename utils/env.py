import gymnasium as gym
import matplotlib.pyplot as plt

def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env

def plot_env(env, name="plotted env"):
    env.reset()
    img = env.render()

    # Plot the rendered image
    plt.imshow(img)
    plt.title(f"Phase {name}")
    plt.show()