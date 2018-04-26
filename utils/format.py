import os
import json
import torch
import numpy
import re

import utils

def get_vocab_path(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name, "vocab.json")

class Vocabulary:
    def __init__(self, model_name):
        self.path = get_vocab_path(model_name)
        utils.create_folders_if_necessary(self.path)
        self.max_size = 100
        self.vocab = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def __getitem__(self, token):
        if not(token in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
            self.save()
        
        return self.vocab[token]

    def save(self):
        json.dump(self.vocab, open(self.path, "w"))

class ObssPreprocessor:
    def __init__(self, model_name, obs_space):
        self.vocab = Vocabulary(model_name)
        self.obs_space = {
            "image": 147,
            "instr": self.vocab.max_size
        }

    def __call__(self, obss, use_gpu=False):
        # Preprocessing images

        np_image = numpy.array([numpy.array(obs["image"]).reshape(-1) for obs in obss])
        image = torch.tensor(np_image).float()
        if use_gpu:
            image = image.cuda()

        # Preprocessing instructions

        instr = []
        max_instr_len = 0
        
        for obs in obss:
            tokens = re.findall("([a-z]+)", obs["mission"].lower())
            instr_ = [self.vocab[token] for token in tokens]
            instr.append(instr_)
            max_instr_len = max(len(instr_), max_instr_len)
        
        np_instr = numpy.zeros((max_instr_len, len(obss), self.vocab.max_size))

        for i, instr_ in enumerate(instr):
            hot_instr_ = numpy.zeros((len(instr_), self.vocab.max_size))
            hot_instr_[numpy.arange(len(instr_)), instr_] = 1
            np_instr[:len(instr_),i,:] = hot_instr_
        
        instr = torch.tensor(np_instr).float()
        if use_gpu:
            instr = instr.cuda()

        # Define the observation the model will receive

        obs = {
            "image": image,
            "instr": instr
        }

        return obs