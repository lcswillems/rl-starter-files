import os
import torch

from model import ACModel
import utils

def get_model_dir(model_name):
    return os.path.join(utils.get_storage_dir(), "models", model_name)

def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")

def create_model(observation_space, action_space):
    model = ACModel(observation_space, action_space)
    model.eval()
    return model

def load_model(model_name):
    path = get_model_path(model_name)
    model = torch.load(path)
    model.eval()
    return model

def save_model(model, model_name):
    path = get_model_path(model_name)
    utils.create_folders_if_necessary(path)
    torch.save(model, path)