from phasicdoorkey import PhasicDoorKeyEnv
import matplotlib.pyplot as plt

from scripts.train import train
from utils.format import train_arg_parser

from utils.storage import generate_gif
from utils.env import plot_env

def main():
    version = "v0.8"

    # only have short gif with 100 steps max
    env_p1 = PhasicDoorKeyEnv(phase=1, size=7, max_steps=100, render_mode="rgb_array")
    env_p2 = PhasicDoorKeyEnv(phase=2, size=7, max_steps=100, render_mode="rgb_array")
    env_p3 = PhasicDoorKeyEnv(phase=3, size=7, max_steps=100, render_mode="rgb_array")

    show_envs = [env_p1, env_p2, env_p3]
    for i, env in enumerate(show_envs):
        plot_env(env, str(i+1))

    envs = []
    for i in range(3):
        environments = [PhasicDoorKeyEnv(phase=i+1, size=7) for _ in range(100)]
        envs.append(environments)

    # PHASE 1
    print("\n\nPHASE 1")
    train_args = train_arg_parser("ppo", envs[0], model=f"model_{version}", frames=1e6)
    train(train_args)

    generate_gif(env_p1, phase=1, version=version)

    # PHASE 2
    print("\n\nPHASE 2")
    train_args = train_arg_parser("ppo", envs[1], model=f"model_{version}", frames=1e6)
    train(train_args)

    generate_gif(env_p2, phase=2, version=version)

    # PHASE 3
    # the agent does not train anymore 
    print("\n\nPHASE 3")
    # this is the code for if you want to only train on phase 3 to see how the agent performs
    # train_args = train_arg_parser("ppo", envs[2], model=f"model_{version}", frames=1e6)
    # train(train_args)
    generate_gif(env_p3, phase=3, version=version)


if __name__ == "__main__":
    main()