import numpy as np
from sklearn.datasets import make_spd_matrix
from scipy.special import binom

distNames = dict({0:'normal',
                 1:'poisson',
                 2:'uniform',
                 3:'lognormal',
                 4:'mixture'})

class partitions:
    def __init__(self, nclasses, nelements):
        nclasses = int(nclasses)
        nelements = int(nelements)
        first = [0] * nclasses
        first[0] = nelements
        self.nclasses = nclasses
        self.nelements = nelements
        self.state = first
    
    def __iter__(self):
        return self
    
    @staticmethod
    def _push(state):
        if state[0] == 0:
            return None
        new = [0] * len(state)
        new[0] = state[0] - 1
        new[1] = sum(state[1:]) + 1
        return new
    
    def _get_next(self, state):
        if len(state) == 2:
            return self._push(state)
        ns = self._get_next(state[1:])
        if ns is None:
            return self._push(state)
        else:
            return [state[0]] + ns
    
    def next(self):
        if self.state is None:
            raise StopIteration()
        state = self.state[:]
        self.state = self._get_next(self.state)
        return state
    
    def __next__(self):
        return self.next()
    
    def __len__(self):
        le = binom(self.nclasses + self.nelements - 1,
                   self.nclasses - 1)
        return int(le)

class sampler_class:
    def __init__(self, d=200, variance=1, nnorm=4):
        self.nnorm = nnorm
        self.d = int(d)
        self.variance = variance
        self.init_norm_mix(nnorm)
        
    def init_norm_mix(self, nnorm=4):
        # init normal mixture (hardest)
        weights = np.random.uniform(1, 10, size=nnorm)
        weights /= sum(weights)
        if self.d > 10:
            scale = [2]*10 + ([.1]*(self.d-10))
        else:
            scale = 2
        meanm = [np.random.normal(loc=0, scale=scale, size=self.d)
                 for i in range(nnorm-1)]
        meanm.append(-np.sum(meanm, axis=0))
        covm = [make_spd_matrix(self.d) for i in range(nnorm-1)]

        comb = self.incomplete_comb(weights, meanm, covm)
        fac = .9/max(np.linalg.eig(comb)[0])
        meanm = np.multiply(meanm, np.sqrt(fac))
        covm = np.multiply(covm, fac)

        comb = self.incomplete_comb(weights, meanm, covm)
        I = self.variance * np.eye(self.d)
        lastCov = (I - comb) / weights[-1]
        lastCov.shape = (1,) + lastCov.shape
        covm = np.append(covm, lastCov, axis=0)

        self.weights = weights.flatten()
        self.means = meanm
        self.covs = covm

    def incomplete_comb(self, weights, means, covs):
        # returnes the incomplete combined covariance
        nnorm = len(weights)
        mean = np.sum(weights[:, np.newaxis] * means, axis = 0)
        meancomb = np.stack([np.dot((means[i] - mean).reshape(-1, 1),
                                    (means[i] - mean).reshape(1, -1))
                                    for i in range(nnorm)])
        wr = weights.reshape(-1, 1, 1)
        comb = np.sum(wr[:-1] * covs, axis = 0)
        comb += np.sum(wr * meancomb, axis = 0)
        return comb
    
    def draw(self, dist):
        if dist == 0:
            return np.random.normal(loc=0, scale=np.sqrt(self.variance), size=self.d)
        elif dist == 1:
            return np.random.poisson(self.variance, size=self.d) - self.variance
        elif dist == 2:
            span = np.sqrt(12 * self.variance) / 2
            return np.random.uniform(-span, span, size=self.d)
        elif dist == 3:
            sigma = .2
            mean = (np.log(self.variance) - np.log(np.exp(sigma**2) - 1) - (sigma**2)) / 2
            diff = np.exp(mean + ((sigma**2)/2))
            return np.random.lognormal(mean, sigma, size=self.d) - diff
        elif dist == 4:
            # a mixtrure of normal distributions
            numb = np.random.choice(self.nnorm, p=self.weights)
            return np.random.multivariate_normal(mean=self.means[numb],
                                                 cov=self.covs[numb])