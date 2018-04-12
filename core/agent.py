import multiprocessing
from utils.memory import Memory
from utils.general import *
import torch
from torch.autograd import Variable
import time
import numpy as np


def collect_samples(pid, queue, env, policy, render, min_batch_size):
    torch.randn(pid, )
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = float('+inf')
    max_reward = float('-inf')
    num_episodes = 0

    while num_steps < min_batch_size:
        done = False
        obs = env.reset()
        reward_episode = 0

        while not(done):
            obs_var = Variable(torch.from_numpy(obs).float().unsqueeze(0), volatile=True)
            action = policy.select_action(obs_var)[0].numpy()
            next_obs, reward, done, _ = env.step(action)
            mask = 0 if done else 1
            memory.push(obs, action, next_obs, reward, mask)
            obs = next_obs

            if render:
                env.render()

            reward_episode += reward
            num_steps += 1
        
        num_episodes += 1
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)
        total_reward += reward_episode
        log['num_steps'] = num_steps
        log['num_episodes'] = num_episodes
        log['total_reward'] = total_reward
        log['avg_reward'] = total_reward / num_episodes
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    return log


class Agent:
    def __init__(self, env_factory, policy, render=False, num_threads=1):
        self.env_factory = env_factory
        self.policy = policy
        self.render = render
        self.num_threads = num_threads
        self.env_list = []
        for i in range(num_threads):
            self.env_list.append(self.env_factory(i))

    def collect_samples(self, min_batch_size):
        t_start = time.time()
        if use_gpu:
            self.policy.cpu()
        thread_batch_size = min_batch_size // self.num_threads)
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env_list[i + 1], self.policy,
                           False, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env_list[0], self.policy,
                                      self.render, thread_batch_size)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.concat(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        if use_gpu:
            self.policy.cuda()
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
