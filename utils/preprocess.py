import os
import json
import numpy
import re
import torch
import torch_rl
import gym

import utils

def get_obss_preprocessor(env_id, obs_space, model_dir):
    # Check if it is a MiniGrid environment
    if re.match("MiniGrid-.*", env_id):
        obs_space = {"image": obs_space.spaces['image'].shape, "instr": 100}

        vocab = Vocabulary(model_dir, obs_space["instr"])
        def preprocess_obss(obss, device=None):
            return torch_rl.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "instr": preprocess_instrs([obs["mission"] for obs in obss], vocab, device=device)
            })

    # Check if the obs_space is of type Box([X, Y, 3])
    elif isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3 and obs_space.shape[2] == 3:
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_rl.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device)
            })

    else:
        raise "Unknown observation space: " + obs_space

    return obs_space, preprocess_obss

def preprocess_images(images, device=None):
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)

def preprocess_instrs(instrs, vocab, device=None):
    var_indexed_instrs = []
    max_instr_len = 0

    for instr in instrs:
        tokens = re.findall("([a-z]+)", instr.lower())
        var_indexed_instr = numpy.array([vocab[token] for token in tokens])
        var_indexed_instrs.append(var_indexed_instr)
        max_instr_len = max(len(var_indexed_instr), max_instr_len)

    indexed_instrs = numpy.zeros((len(instrs), max_instr_len))

    for i, indexed_instr in enumerate(var_indexed_instrs):
        indexed_instrs[i, :len(indexed_instr)] = indexed_instr

    return torch.tensor(indexed_instrs, device=device, dtype=torch.long)

class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, model_dir, max_size):
        self.path = utils.get_vocab_path(model_dir)
        self.max_size = max_size
        self.vocab = {}
        if os.path.exists(self.path):
            self.vocab = json.load(open(self.path))

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def save(self):
        utils.create_folders_if_necessary(self.path)
        json.dump(self.vocab, open(self.path, "w"))