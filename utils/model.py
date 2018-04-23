import os
import torch

from model import ACModel
import utils

def get_model_path(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name+".pt")

def load_model(observation_space, action_space, path):
    acmodel = ACModel(observation_space, action_space)
    if path is not None and os.path.exists(path):
        acmodel.load_state_dict(torch.load(path))
    acmodel.eval()
    return acmodel

def save_model(acmodel, path):
    utils.create_folders_if_necessary(path)
    torch.save(acmodel.state_dict(), path)