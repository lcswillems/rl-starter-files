from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A multiprocess execution of environments."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        self.unique_env = len(self.envs) == 1
        if self.unique_env:
            self.env = self.envs[0]
        else:
            self.locals, self.remotes = zip(*[Pipe() for _ in self.envs])
            self.ps = [Process(target=worker, args=(remote, env))
                       for (remote, env) in zip(self.remotes, self.envs)]
            for p in self.ps:
                p.daemon = True
                p.start()
            for remote in self.remotes:
                remote.close()

    def reset(self):
        if self.unique_env:
            return [self.env.reset()]
        else:
            for local in self.locals:
                local.send(("reset", None))
            return [local.recv() for local in self.locals]

    def step(self, actions):
        if self.unique_env:
            obs, reward, done, info = self.env.step(actions[0])
            if done:
                obs = self.env.reset()
            return zip(*[(obs, reward, done, info)])
        else:
            for local, action in zip(self.locals, actions):
                local.send(("step", action))
            return zip(*[local.recv() for local in self.locals])

    def render(self):
        raise NotImplementedError