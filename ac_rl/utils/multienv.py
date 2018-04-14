from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            conn.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            conn.send(ob)
        else:
            raise NotImplementedError

class MultiEnv(gym.Env):
    """
    An asynchronous multi-environment.
    """

    def __init__(self, envs):
        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.locals, self.remotes = zip(*[Pipe() for _ in self.envs])
        self.ps = [Process(target=worker, args=(remote, env))
                   for (remote, env) in zip(self.remotes, self.envs)]

        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.remotes:
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(('reset', None))
        obss = [local.recv() for local in self.locals]
        return obss

    def step(self, actions):
        for local, action in zip(self.locals, actions):
            local.send(('step', action))
        results = [local.recv() for local in self.locals]
        obss, rewards, dones, infos = zip(*results)
        return obss, rewards, dones, infos

    def render(self):
        raise NotImplementedError