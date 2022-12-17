import numpy as np


class Buffer:
    is_prioritized = False
    weight = []
    capcity = 100000
    _size = 0
    obs,act,rew,done,obs_next = [[] for _ in range(5)]
    
    def __init__(self,capcity=100000) -> None:
        self.capcity = capcity

    def add(self, obs, act, rew, done, obs_next, info=None, wei=1):
        if self._size >= self.capcity:
            self.obs.pop(0)
            self.act.pop(0)
            self.rew.pop(0)
            self.done.pop(0)
            self.obs_next.pop(0)
            
            if self.is_prioritized:
                self.weight.pop(0)
                
            self._size -= 1
            
        self.obs.append(obs)
        self.act.append(act)
        self.rew.append(rew)
        self.done.append(done)
        self.obs_next.append(obs_next)
        
        if self.is_prioritized:
            self.weight.append(wei)
            
        self._size += 1

    def sample(self, batch_size):
        if self.is_prioritized:
            p = np.array(self.weights)
            p = p / np.sum(p)
            arg = np.random.choice(np.arange(self._size), batch_size, p=p)
            return np.array(self.obs)[arg], np.array(self.act)[arg], np.array(self.rew)[arg], np.array(self.done)[arg], np.array(self.obs_next)[arg]
        else:
            arg = np.random.choice(np.arange(self._size), batch_size)
            return np.array(self.obs)[arg], np.array(self.act)[arg], np.array(self.rew)[arg], np.array(self.done)[arg], np.array(self.obs_next)[arg]

    def __len__(self)->int:
        return self._size
    