from multiprocessing import Queue, Process
from utils.inrl import *
import torch
from torch.autograd import Variable
import numpy as np

use_gpu = torch.cuda.is_available()

def collect_trajectories(envs, policy_net, num_episodes):
    if use_gpu:
        policy_net.cpu()
    
    queue = Queue()

    def start(pid):
        if pid == 0:
            collect_trajectory(pid, queue, envs[pid], policy_net)
        else:
            p = Process(target=collect_trajectory, args=(pid, queue, envs[pid], policy_net))
            p.start()

    transs = {}
    log = {}

    for i in range(min(len(envs), num_episodes)):
        start(i)
    
    for _ in range(num_episodes):
        pid, worker_transs, worker_log = queue.get()
        concat_to_dict(transs, worker_transs)
        append_to_dict(log, worker_log)
        if i+1 < num_episodes:
            start(pid)
            i += 1
    
    if use_gpu:
        policy_net.cuda()
    
    return transs, log

def collect_trajectory(pid, queue, env, policy_net):
    torch.randn(pid+1, )

    done = False
    obs = env.reset()

    transs = {}
    num_steps = 0
    returnn = 0

    while not(done):
        obs_var = Variable(torch.from_numpy(obs).float().unsqueeze(0), volatile=True)
        action = policy_net.get_sampled_action(obs_var)[0].numpy()
        next_obs, reward, done, _ = env.step(action)
        mask = 0 if done else 1
        
        trans = {"obs": obs, "action": action, "next_obs": next_obs, "reward": reward, "mask": mask}
        append_to_dict(transs, trans)
        
        obs = next_obs

        num_steps += 1
        returnn += reward
        
    log = {"num_steps": num_steps, "return": returnn}

    queue.put([pid, transs, log])

def estimate_advantages(rewards, masks, values, discount, gae_coef):
    if use_gpu:
        rewards, masks, values = rewards.cpu(), masks.cpu(), values.cpu()

    advantages = torch.zeros(rewards.size(0), 1).float()

    prev_value = 0
    prev_advantage = 0

    for i in reversed(range(rewards.size(0))):
        delta = rewards[i] + discount * prev_value * masks[i] - values[i]
        advantages[i] = delta + discount * gae_coef * prev_advantage * masks[i]

        prev_value = values[i, 0]
        prev_advantage = advantages[i, 0]

    returns = values + advantages

    if use_gpu:
        advantages, returns = advantages.cuda(), returns.cuda()
        
    return advantages, returns