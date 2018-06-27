import random
import os
import numpy
import torch

def get_storage_dir():
    if "TORCH_RL_STORAGE" in os.environ:
        return os.environ["TORCH_RL_STORAGE"]
    return "storage"

def get_save_dir(save_name):
    return os.path.join(get_storage_dir(), save_name)

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not(os.path.isdir(dirname)):
        os.makedirs(dirname)

def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def synthesize(array):
    return {
        "mean": numpy.mean(array),
        "std": numpy.std(array),
        "min": numpy.amin(array),
        "max": numpy.amax(array)
    }