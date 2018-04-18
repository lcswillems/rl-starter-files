import os
import json
import torch
from torch.autograd import Variable
import numpy as np
import re

import utils

def get_vocab_path():
    return os.path.join(utils.storage_dir(), "vocab.json")

def load_vocab():
    path = get_vocab_path()
    if os.path.exists(path):
        return json.load(open(path))
    return {}

def save_vocab(vocab):
    json.dump(vocab, open(get_vocab_path(), "w"))

vocab = load_vocab()
vocab_has_changed = False
vocab_max_size = 20

def preprocess_obs_space(obs_space):
    obs_space = {
        "image": int(np.prod(obs_space.spaces["image"].shape)),
        "instr": vocab_max_size
    }
    return obs_space

def preprocess_instr(instr, vocab):
    global vocab_has_changed, vocab_max_size

    tokens = re.findall("([a-z]+)", instr.lower())
    instr = []

    for token in tokens:
        if not(token in vocab.keys()):
            if len(vocab) >= vocab_max_size:
                raise ValueError("Vocabulary maximum capacity reached")
            vocab[token] = len(vocab) + 1
            vocab_has_changed = True
        instr.append(vocab[token])
        
    return instr, vocab

def preprocess_obss(obss, volatile):
    # Preprocessing images

    np_image = np.array([np.array(obs["image"]).reshape(-1) for obs in obss])
    image = torch.from_numpy(np_image).float()

    # Preprocessing instructions

    global vocab, vocab_has_changed

    instr = []
    max_instr_len = 0
    
    for obs in obss:
        instr_, vocab = preprocess_instr(obs["mission"], vocab)
        instr.append(instr_)
        max_instr_len = max(len(instr_), max_instr_len)
    
    if vocab_has_changed:
        save_vocab(vocab)
        vocab_has_changed = False
    
    np_instr = np.zeros((max_instr_len, len(obss), vocab_max_size))

    for i, instr_ in enumerate(instr):
        hot_instr_ = np.zeros((len(instr_), vocab_max_size))
        hot_instr_[np.arange(len(instr_)), instr_] = 1
        np_instr[0:len(instr_),i,:] = hot_instr_
    
    instr = torch.from_numpy(np_instr).float()

    # Define the observation the model will receive

    obs = {
        "image": Variable(image, volatile=volatile),
        "instr": Variable(instr, volatile=volatile)
    }

    return obs

def reshape_reward(obs, action, reward):
    if reward > 0:
        return reward*10
    return -0.1