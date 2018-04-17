# PyTorch A2C and PPO

A fast, robust, readable and multi-process PyTorch implementation of:

- [Synchronous A3C (A2C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)

inspired from 3 repositories:

1. [pytorch-a2c-ppo-acktr](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr)
2. [Pytorch RL](https://github.com/Khrylx/PyTorch-RL)
3. [OpenAI Baselines](https://github.com/openai/baselines)

## Features

- Discrete action space
- Entropy regularization
- Very fast (2400 FPS for A2C against 2100 for repo 1 and 1700 for repo 2)
- CUDA (x10 faster than CPU implementation)
- Multiprocessing for collecting agent's trajectories in multiple environments simultaneously

## Installation

Clone this repository and install the other dependencies with pip3:

```
git clone https://github.com/lcswillems/pytorch-a2c-ppo
cd pytorch-a2c-ppo
pip3 install -e .
```

## Important note before using

If you have a GPU, PyTorch will create additional threads when performing computations which can damage the performance of multiprocessing. This problem is most serious with Linux, where multiprocessing can be even slower than a single thread. You may have to set the OMP_NUM_THREADS to 1:

```
export OMP_NUM_THREADS=1
```

## Uses

[MinGrid environments](https://github.com/maximecb/gym-minigrid) are used in the following examples.

### Training

`scripts/train.py` enables you to load a model, train it with the actor-critic algorithms and save it.

2 arguments are required:
- `--algo ALGO`: name of the actor-critic algorithm.
- `--env ENV`: name of the environment to train on.

and a bunch of optional arguments are available among which:
- `--model MODEL`: name of the model, used for loading and saving it. If not specified, it is the `_`-concatenation of the environment name and algorithm name.
- `--frames-per-update FRAMES_PER_UPDATE`: number of frames per agent before updating parameters.
- ... (see more in `train.py`)

Here is an example of command:
```
scripts/train.py --algo a2c --env MiniGrid-DoorKey-5x5-v0 --seed 12 --processes 8 --save-interval 10 --update-frames 50
```

### Enjoying

`scripts/enjoy.py` enables you to visualize your trained model acting.

2 arguments are required:
- `--env ENV`: name of the environment to act on.
- `--model MODEL`: name of the trained model.

and several optional arguments are available (see more in `enjoy.py`).

Here is an example of command:
```
scripts/enjoy.py --env MiniGrid-DoorKey-8x8-v0 --model DoorKey
```

<p align="center"><img src="README-images/enjoy-doorkey.gif"></p>

## Todo (will be realised in next few days)

- Dictionnary observations
- Recurrent policy
- Reward shaping