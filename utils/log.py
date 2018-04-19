import numpy as np

def synthesize(array):
    return np.mean(array), np.std(array), np.amin(array), np.amax(array)