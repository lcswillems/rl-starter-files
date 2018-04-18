import os

def storage_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "storage")

from utils.model import get_model_path, load_model, save_model
from utils.obs import vocab_max_size, preprocess_obss, preprocess_obs_space