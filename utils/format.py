# import os
# import json
import numpy
import re
import torch
import torch_ac
import gymnasium as gym

# import utils
def horse():
    pass

def get_obss_preprocessor(obs_space):
    # Check if obs_space is an image space
    if isinstance(obs_space, gym.spaces.Box):
        obs_space = {"image": obs_space.shape}

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images(obss, device=device)
            })

    # Check if it is a MiniGrid observation space
    elif isinstance(obs_space, gym.spaces.Dict) and "image" in obs_space.spaces.keys():
        obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}

        vocab = Vocabulary(obs_space["text"])

        def preprocess_obss(obss, device=None):
            return torch_ac.DictList({
                "image": preprocess_images([obs["image"] for obs in obss], device=device),
                "text": preprocess_texts([obs["mission"] for obs in obss], vocab, device=device)
            })

        preprocess_obss.vocab = vocab

    else:
        raise ValueError("Unknown observation space: " + str(obs_space))

    return obs_space, preprocess_obss


def preprocess_images(images, device=None):
    # Bug of Pytorch: very slow if not first converted to numpy array
    images = numpy.array(images)
    return torch.tensor(images, device=device, dtype=torch.float)


def preprocess_texts(texts, vocab, device=None):
    var_indexed_texts = []
    max_text_len = 0

    for text in texts:
        tokens = re.findall("([a-z]+)", text.lower())
        var_indexed_text = numpy.array([vocab[token] for token in tokens])
        var_indexed_texts.append(var_indexed_text)
        max_text_len = max(len(var_indexed_text), max_text_len)

    indexed_texts = numpy.zeros((len(texts), max_text_len))

    for i, indexed_text in enumerate(var_indexed_texts):
        indexed_texts[i, :len(indexed_text)] = indexed_text

    return torch.tensor(indexed_texts, device=device, dtype=torch.long)


class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

def train_arg_parser(algo, envs, model: str = None, seed: int = 1,
               log_interval: int = 1, save_interval: int = 10, procs: int = 16,
               frames: int = 10**7, epochs: int = 4, batch_size: int = 256,
               frames_per_proc: int = None, discount: float = 0.99, lr: float = 0.001,
               gae_lambda: float = 0.95, entropy_coef: float = 0.01,
               value_loss_coef: float = 0.5, max_grad_norm: float = 0.5,
               optim_eps: float = 1e-8, optim_alpha: float = 0.99,
               clip_eps: float = 0.2, recurrence: int = 1, text: bool = False):
    
    args = {
        'algo': algo,
        'envs': envs,
        'model': model,
        'seed': seed,
        'log_interval': log_interval,
        'save_interval': save_interval,
        'procs': procs,
        'frames': frames,
        'epochs': epochs,
        'batch_size': batch_size,
        'frames_per_proc': frames_per_proc,
        'discount': discount,
        'lr': lr,
        'gae_lambda': gae_lambda,
        'entropy_coef': entropy_coef,
        'value_loss_coef': value_loss_coef,
        'max_grad_norm': max_grad_norm,
        'optim_eps': optim_eps,
        'optim_alpha': optim_alpha,
        'clip_eps': clip_eps,
        'recurrence': recurrence,
        'text': text,
    }
    
    return args


def visualize_arg_parser(envs, model, seed=0, shift=0, 
                         argmax=False, pause=0.1, gif=None, episodes: int = 1e6, 
                         memory=False, text=False):
        
    args = {
        'envs': envs,
        'model': model,
        'seed': seed,
        "shift": shift,
        "argmax": argmax,
        "pause": pause,
        "gif": gif,
        "episodes": episodes,
        "memory": memory,
        'text': text,
    }
    return args