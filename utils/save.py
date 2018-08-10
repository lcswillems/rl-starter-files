import csv
import os
import torch
import json
import logging
import sys

import utils

def get_model_path(save_dir):
    return os.path.join(save_dir, "model.pt")

def model_exists(save_dir):
    path = get_model_path(save_dir)
    return os.path.exists(path)

def load_model(save_dir):
    path = get_model_path(save_dir)
    model = torch.load(path)
    model.eval()
    return model

def save_model(model, save_dir):
    path = get_model_path(save_dir)
    utils.create_folders_if_necessary(path)
    torch.save(model, path)

def get_status_path(save_dir):
    return os.path.join(save_dir, "status.json")

def load_status(save_dir):
    path = get_status_path(save_dir)
    with open(path) as file:
        return json.load(file)

def save_status(status, save_dir):
    path = get_status_path(save_dir)
    utils.create_folders_if_necessary(path)
    with open(path, "w") as file:
        json.dump(status, file)

def get_log_path(save_dir):
    return os.path.join(save_dir, "log.log")

def get_logger(save_dir):
    path = get_log_path(save_dir)
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()

def get_vocab_path(save_dir):
    return os.path.join(save_dir, "vocab.json")

def get_csv_path(save_dir):
    return os.path.join(save_dir, "log.csv")

def get_csv_writer(save_dir):
    csv_path = get_csv_path(save_dir)
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)