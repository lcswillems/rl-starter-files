import os
import sys
import numpy
import logging

import utils

def synthesize(array):
    return {
        "mean": numpy.mean(array),
        "std": numpy.std(array),
        "min": numpy.amin(array),
        "max": numpy.amax(array)
    }

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