import os
import torch
import json

import utils

def get_model_path(run_dir):
    return os.path.join(run_dir, "model.pt")

def model_exists(run_dir):
    path = get_model_path(run_dir)
    return os.path.exists(path)

def load_model(run_dir):
    path = get_model_path(run_dir)
    model = torch.load(path)
    model.eval()
    return model

def save_model(model, run_dir):
    path = get_model_path(run_dir)
    utils.create_folders_if_necessary(path)
    torch.save(model, path)

def get_status_path(run_dir):
    return os.path.join(run_dir, "status.json")

def load_status(run_dir):
    path = get_status_path(run_dir)
    with open(path) as file:
        return json.load(file)

def save_status(status, run_dir):
    path = get_status_path(run_dir)
    utils.create_folders_if_necessary(path)
    with open(path, "w") as file:
        json.dump(status, file)