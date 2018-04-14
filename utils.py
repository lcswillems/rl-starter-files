import os
import torch
import pickle

import ac_rl
from models.policy import Policy
from models.value import Value

def storage_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')

def get_model_path(model_name):
    return os.path.join(storage_dir(), 'models', model_name+".pt")

def load_model(observation_space, action_space, from_path):
    if from_path == None or not(os.path.exists(from_path)):
        policy_net = Policy(observation_space, action_space)
        value_net = Value(observation_space)
    else:
        policy_net, value_net = pickle.load(open(from_path, "rb"))

    if ac_rl.use_gpu:
        policy_net = policy_net.cuda()
        value_net = value_net.cuda()

    return policy_net, value_net

def save_model(policy_net, value_net, to_path):
    if ac_rl.use_gpu:
        policy_net.cpu(), value_net.cpu()
    
    dirname = os.path.dirname(to_path)
    if not(dirname):
        os.makedirs(dirname)

    pickle.dump((policy_net, value_net), open(to_path, 'wb'))

    if ac_rl.use_gpu:
        policy_net.cuda(), value_net.cuda()