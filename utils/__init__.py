import os

def storage_dir():
    return "storage"

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not(os.path.isdir(dirname)):
        os.makedirs(dirname)

from utils.format import ObssPreprocessor
from utils.log import Logger, synthesize
from utils.model import load_model, save_model