import theano.tensor as T
from utils import sharedX
import numpy


class SGDOptimizer(object):
    def __init__(self, learning_rate, weight_decay=0.005, momentum=0.9):
        self.lr = learning_rate
        self.wd = weight_decay
        self.mm = momentum

    def get_updates(self, cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            d = sharedX(p.get_value() * 0.0)#####
            new_d = self.mm * d - self.lr * (g + self.wd * p)
            updates.append((d, new_d))
            updates.append((p, p + new_d))

        return updates


class AdagradOptimizer(object):
    def __init__(self, learning_rate, eps=1e-8):
        self.lr = learning_rate
        self.eps = eps
        self.cache = []

    def get_updates(self, cost, params):
        # Your codes here
        # hint: implementation idea
        #       cache += dx ** 2
        #       p = p - self.lr * dx / (sqrt(cache) + self.eps)
        grads = T.grad(cost=cost, wrt=params)
        if len(self.cache) == 0:
            self.cache = numpy.square(grads)
        else:
            self.cache = numpy.add(self.cache, numpy.square(grads))
        #self.cache =
        #self.cache.append(grads)
        updates = []
        print 'np.sum',self.cache
        cache_squSum = numpy.sqrt(self.cache)
        for p,g,c in zip(params,grads,cache_squSum):
            d = sharedX(p.get_value() * 0.0)
            new_d =  self.lr * g / (c + self.eps)
            updates.append((d, new_d))
            updates.append((p, p - new_d))

        return updates

