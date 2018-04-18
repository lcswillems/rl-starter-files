import os
import torch

from model import ACModel
import utils

def get_model_path(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name+".pt")

def load_model(observation_space, action_space, from_path):
    acmodel = ACModel(observation_space, action_space)
    if from_path != None and os.path.exists(from_path):
        acmodel.load_state_dict(torch.load(from_path))
    return acmodel

def save_model(acmodel, to_path):
    dirname = os.path.dirname(to_path)
    if not(os.path.isdir(dirname)):
        os.makedirs(dirname)
    torch.save(acmodel.state_dict(), to_path)