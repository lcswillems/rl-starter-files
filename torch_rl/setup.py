from setuptools import setup

setup(
    name="torch_rl",
    version="1.0.0",
    keywords="reinforcement learning, actor-critic, a2c, ppo, multi-processes, gpu",
    packages=["torch_rl"],
    install_requires=[
        "numpy>=1.13.0",
        "torch>=0.4.0"
    ]
)