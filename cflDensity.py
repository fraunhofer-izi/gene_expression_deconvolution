import pandas as pd
import numpy as np
from scipy.special import jv, loggamma, logsumexp
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize

def normf(nSamp, r, n, log=False):
    """normalization factor for bessel basis"""
    if r == 0:
        result = float('-inf')
    else:
        result = (n/2) * np.log(r / (2 * np.pi)) - np.log(nSamp)
    if not log: result = np.exp(result)
    return result

def beta(r, x1 = None, x2 = None, n = None, dist = None, log=False):
    """Returns the summands of the bassel formulated $\tilde f$ even if $x = x'$."""
    if n is None and np.isscalar(x1):
        n = 1
    elif n is None:
        n = x1.shape[1]
    if dist is None:
        dist = euclidean_distances(x1, x2)
    index = dist == 0
    with np.errstate(divide='ignore'):
        logVol = (n/2) * np.log(r/2) - loggamma(n/2 + 1)
        logTerms = np.where(index, logVol, np.log(jv(n/2, r * dist) + 0j))
        logScale = np.where(index, 0, (n/2)*np.log(dist))
    result = logTerms - logScale
    if not log: result = np.exp(result)
    return result

def getftd2(xs = None, x = None, r = None, n = None, nSamp = None, dist = None, log = False):
    if n is None and xs is None:
        raise Exception('Can not determine dimensionality.')
    elif n is None:
        n = xs.shape[1]
    if x is not None and (n != x.shape[1]):
        raise Exception('Both vector sets must have same dimensionality/number of columns.')
    if nSamp is None and x is None:
        nSamp = dist.shape[0]
    elif nSamp is None:
        nSamp = len(x)
    if r is None:
        r = getBestRad(x)
    if r == 0:
        result = float('-inf') * np.ones(len(xs))
    else:
        logTerms = beta(r, x1 = xs, x2 = x, n = n, dist = dist, log = True)
        logNf = normf(nSamp, r, n, log = True)
        nfLogTerms = logTerms + logNf
        result = logsumexp(nfLogTerms, axis = 1)
    if not log: result = np.real(np.exp(result))
    return np.real(result)

def crossDens(r, x = None, n = None, dist = None, nSamp = None, log = False, regularize = False):
    if n is None and x is None:
        raise Exception('Can not determine dimensionality.')
    elif n is None:
        n = x.shape[1]
    if dist is None:
        dist = euclidean_distances(x, x)
    if nSamp is None and x is None:
        nSamp = dist.shape[0]
    elif nSamp is None:
        nSamp = len(x)
    if r == 0:
        result = float('-inf') * np.ones(len(x))
    else:
        logTerms = beta(r = r, n = n, dist = dist, log = True)
        if regularize:
            diagonal = logTerms.diagonal() + (np.pi * 1j)
        else:
            diagonal = float('-inf')
        np.fill_diagonal(logTerms, diagonal)
        logNf = normf(nSamp - 1, r, n, log = True)
        nfLogTerms = logTerms + logNf
        result = logsumexp(nfLogTerms, axis = 1)
    if not log: result = np.real(np.exp(result))
    return result

def ll(r, x = None, dist = None, n = None, log = True):
    """returns log likelyhood of parameter r by leave one out cross validation"""
    if r == 0:
        prob = float('-inf')
    else:
        prob = logsumexp(crossDens(r, x = x, n = n, dist = dist, log = True, regularize = True))
    if not log: prob = np.real(np.exp(prob))
    return np.mean(prob)

def kld(r, x = None, dist = None, n = None, log = False):
    """returns log likelyhood of parameter r by leave one out cross validation"""
    if r == 0:
        prob = float('-inf')
    else:
        prob = np.sum(crossDens(r, x = x, n = n, dist = dist, log = True))
    if not log: prob = np.real(np.exp(prob))
    return np.mean(prob)

def getBestRad(x = None, dist = None, r0 = None, n = None):
    if n is None and x is None:
        raise Exception('Can not determine dimensionality.')
    elif n is None:
        n = x.shape[1]
    if dist is None:
        dist = euclidean_distances(x, x)
    if r0 is None:
        r0 = 40 / np.max(dist)
    mf = lambda r: -np.real(ll(np.absolute(r), dist = dist, n = n, log=True))
    result = minimize(mf, r0)
    radius = np.absolute(np.asscalar(result['x']))
    return radius