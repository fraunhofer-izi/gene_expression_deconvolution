import os
os.environ['OMP_NUM_THREADS'] = '1' # single thread openMP and BLAS
from multiprocessing import Pool
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import scale as doscale


def get_nL(gram, isSample=False):
    if isSample:
        gram = rbf_kernel(gram)
    n = gram.shape[0]
    nL = 1 / (n - 1)
    ndgram = gram[~np.eye(gram.shape[0], dtype=bool)]
    nL *= np.exp(ndgram).sum()
    grams = np.linalg.matrix_power(gram, 2)
    nL -= 2 * np.exp(np.diagonal(grams) / (2*n)).sum()
    dbs = np.linalg.det(np.eye(n) - (grams / (n**2)))
    nL += n / np.sqrt(dbs)
    return nL

def pool_f(dat):
    seed, hmean, hcov, n = dat
    np.random.seed(seed)
    samp = np.random.multivariate_normal(hmean, hcov, n)
    q = get_nL(rbf_kernel(samp))
    return q

class rkhs_gaussianity:
    def __init__(self, data, scale=True, B=1000, mean=None, cov=None):
        self.data = np.array(data).astype('float64')
        self._hmean_cash = mean
        self._hcov_cash = cov
        if scale:
            doscale(self.data, copy=False, axis=0)
            self._hmean_cash = None
            self._hcov_cash = None
        self.B = B
        return
    
    def cov(self):
        return np.cov(self.data)
    
    def gram(self):
        return rbf_kernel(self.data)
    
    def nL(self):
        return get_nL(self.gram())
    
    def hmean(self):
        if self._hmean_cash is None:
            self._hmean_cash = np.mean(self.data, axis=0)
        return self._hmean_cash
    
    def hcov(self):
        if self._hcov_cash is None:
            self._hcov_cash = np.cov(self.data, rowvar=False)
        return self._hcov_cash
    
    def qsamps(self, B=None):
        if B is not None:
            self.B = B
        if not hasattr(self, '_nsamp') or self._nsamp is None:
            self._nsamp = 0
            self._qsamp_cache = list()
        diff = self.B - self._nsamp
        tasks = [[i, self.hmean(), self.hcov(), self.data.shape[0]] for i in range(diff)]
        if diff > 0:
            with Pool() as pool:
                for q in pool.imap_unordered(pool_f, tasks):
                    self._qsamp_cache.append(q)
            self._nsamp = self._nsamp + diff
        else:
            self._qsamp_cache = self._qsamp_cache[0:self.B]
        return np.array(self._qsamp_cache)
    
    def pvalue(self):
        return sum(self.qsamps() > self.nL()) / self.B