import numpy as np
import torch

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def concat_to_dict(d_to_concat, d):
    for key in d.keys():
        if not(key in d_to_concat.keys()):
            d_to_concat[key] = []
        d_to_concat[key] += d[key]

def append_to_dict(d_to_append, d):
    for key in d.keys():
        if not(key in d_to_append.keys()):
            d_to_append[key] = []
        d_to_append[key].append(d[key])

def split_dict_into_torch_batches(d, batch_size):
    first_value = d[list(d.keys())[0]]
    indexes = np.random.permutation(first_value.shape[0])
    indexes = torch.from_numpy(indexes)

    batches = []

    for i in range(0, len(indexes), batch_size):
        batch = type(d)()
        for key in d.keys():
            batch[key] = d[key][indexes[i:i+batch_size]]
        batches.append(batch)
    
    return batches