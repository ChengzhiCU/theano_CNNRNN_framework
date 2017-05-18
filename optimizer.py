import theano.tensor as T
from utils import sharedX
import numpy as np


class SGDOptimizer(object):
    def __init__(self, learning_rate, weight_decay, momentum):
        self.lr = learning_rate
        self.wd = weight_decay
        self.mm = momentum

    def get_updates(self, cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            d = sharedX(p.get_value() * 0.0)
            new_d = self.mm * d - self.lr * (g + self.wd * p)
            updates.append((d, new_d))
            updates.append((p, p + new_d))

        return updates


class RMSpropOptimizer(object):
    def __init__(self, learning_rate, rho=0.9, eps=1e-8):
        # rho: decay_rate
        self.lr = learning_rate
        self.rho = rho
        self.eps = eps
        #self.cache = []

    def get_updates(self, cost, params):
        # Your codes here
        grads = T.grad(cost=cost,wrt = params)
        updates = []
        #cache = sharedX(params.get_value() * 0.0)
        for p,g in zip(params, grads):
            cache = sharedX(p.get_value() * 0.0)
            new_cache = self.rho * cache + (1 - self.rho) * g**2
            new_p = p - self.lr * g / (np.sqrt(new_cache) + self.eps)
            updates.append((cache, new_cache))
            updates.append((p, new_p))
        return updates






        '''
        if len(self.cache) == 0:
            self.cache = np.square(grads)
        else:
            self.cache = self.rho * self.cache + (1 - self.rho) * np.square(grads)
        cache_sqrt = np.sqrt(self.cache)
        for p,g,c in zip(params,grads,cache_sqrt):
            d = sharedX(p.get_value() * 0.0)
            new_d = self.lr * g / (c + self.eps)
            updates.append((d, new_d))
            updates.append((p, p - new_d))
        return updates
        '''

    




