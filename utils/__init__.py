import os
import random
import numpy
import torch

def get_storage_dir():
    if "TORCH_RL_STORAGE" in os.environ:
        return os.environ["TORCH_RL_STORAGE"]
    return "storage"

def get_run_dir(run_name):
    return os.path.join(get_storage_dir(), run_name)

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

from utils.agent import Agent
from utils.format import ObssPreprocessor
from utils.log import synthesize, get_logger
from utils.model import model_exists, load_model, save_model, load_status, save_status