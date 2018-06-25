import csv
import os
import torch
import json
import logging
import sys

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

def get_log_path(run_dir):
    return os.path.join(run_dir, "log.log")

def get_logger(run_dir):
    path = get_log_path(run_dir)
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

def get_vocab_path(run_dir):
    return os.path.join(run_dir, "vocab.json")

def get_csv_path(run_dir):
    return os.path.join(run_dir, "log.csv")

def get_csv_writer(run_dir):
    csv_path = get_csv_path(run_dir)
    utils.create_folders_if_necessary(csv_path)
    file = open(csv_path, "a")
    return csv.writer(file)