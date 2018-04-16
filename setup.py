from setuptools import setup

setup(
    name='torch_ac',
    version='0.0.1',
    keywords='actor-critic, a2c, ppo, multi-processes, gpu',
    packages=['torch_ac'],
    install_requires=[
        'numpy>=1.13.0',
        'torch>=0.3.1'
    ]
)