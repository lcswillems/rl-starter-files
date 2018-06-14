import os
import torch

import utils

def get_model_path(run_dir):
    return os.path.join(run_dir, "model.pt")

def load_model(run_dir, raise_not_found=True):
    path = get_model_path(run_dir)
    try:
        model = torch.load(path)
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))

def save_model(model, run_dir):
    path = get_model_path(run_dir)
    utils.create_folders_if_necessary(path)
    torch.save(model, path)