import os

def storage_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage")

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not(os.path.isdir(dirname)):
        os.makedirs(dirname)

from utils.format import preprocess_obs_space, preprocess_obss, reshape_reward
from utils.log import Logger, synthesize
from utils.model import get_model_path, load_model, save_model