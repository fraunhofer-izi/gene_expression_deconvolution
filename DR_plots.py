import argparse
import matplotlib
if __name__ == "__main__":
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser(description = 'Correlation plot of before and after DR to specified dimension.')
    parser.add_argument('integers', metavar = 'N', type=int, nargs = '+',
                       help = 'The number of dimensions to project to. (multiple possible)')
    parser.add_argument("--res", type = int, default = 200,
                        help = 'Corrleation plot resolution.')
    args = parser.parse_args()

import os
os.environ['OMP_NUM_THREADS'] = '1' # single thread openMP and BLAS
from matplotlib import pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm_notebook, tqdm
from multiprocessing import Pool
from scipy.stats import gaussian_kde, spearmanr, pearsonr
from scipy.special import digamma
from scipy.optimize import minimize_scalar

with open(os.environ['degOutPath'] + 'other_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
pca = data['pca']
counts = data['counts']
n = counts.shape[1]

def beta(counts):
    alpha = counts + 1
    sub = np.log(np.sum(alpha, axis = 1, keepdims = True))
    nom = np.log(alpha)
    result = nom - sub
    return result
    
def inv_beta(log_alpha, seq_depth = 1e8, prior = 0):
    #bound = np.percentile(log_alpha, 20, axis = 1, keepdims = True)
    n = log_alpha.shape[1]
    alpha = np.exp(log_alpha)
    result = alpha*(seq_depth + n) - 1 - prior
    result = np.clip(result, 0, None)
    return result

def re_counts(log_p, seq_depth = 1e8):
    return np.exp(log_p) * seq_depth

def phi(X, ncomp):
    ind = range(ncomp)
    if pca.mean_ is not None:
        X = X - pca.mean_
    X_transformed = np.dot(X, pca.components_[ind, :].T)
    return X_transformed

def inv_phi(X):
    ind = range(X.shape[1])
    return np.dot(X, pca.components_[ind, :]) + pca.mean_

def dgamma(counts):
    seq_depth = np.sum(counts, axis = 1, keepdims = True)
    alpha = counts + 1
    sub = digamma(seq_depth + counts.shape[1])
    nom = digamma(alpha)
    result = nom - sub
    return result, seq_depth

def proj_O(log_counts):
    n = log_counts.shape[1]
    diff = log_counts.sum(axis = 1, keepdims = True) / n
    return log_counts - diff

def rev_O(o_counts):
    level = np.exp(o_counts).sum(axis = 1, keepdims = True)
    t = np.log(level)
    return o_counts - t

def _objective(l_count, dg, sub):
    alpha = np.exp(l_count) + 1
    nom = digamma(alpha)
    result = nom - sub
    return (result - dg)**2

def _single(dat):
    lalpha, sub = dat
    res = minimize_scalar(_objective, args = (lalpha, sub), tol = 1e-20)
    return np.round(np.exp(res.x))
    
def inv_dg(log_alpha, sub = 18, parallel = True):
    results = []
    N = np.repeat(1, 10).shape
    if len(sub) != N:
        sub = np.repeat(sub, N)
    iterator = range(log_alpha.shape[0])
    if parallel:
        iterator = tqdm_notebook(iterator)
    for samp in iterator:
        tasks = [(lalpha, sub[samp]) for lalpha in log_alpha[samp, :]]
        if parallel:
            with Pool() as pool:
                samp_res = pool.map(_single, tasks)
        else:
            samp_res = [_single(t) for t in tasks]
        results.append(np.array(samp_res))
    return np.stack(results, axis = 0)

def degenerate(X, ncomp, m=1e8):
    return inv_beta(rev_O(inv_phi(phi(beta(X), ncomp))), m)

def fast_deg(beta_counts, ncomp, m=1e8, prior=0):
    return inv_beta(rev_O(inv_phi(phi(beta_counts, ncomp))), m, prior)

def lin_deg(beta_counts, ncomp):
    return rev_O(inv_phi(phi(beta_counts, ncomp)))

def deplot(x, y, legend = True, ax = None, fig = None, res = 200):
    # fit an array of size [Ndim, Nsamples]
    data = np.vstack([x, y])
    kde = gaussian_kde(data)

    # evaluate on a regular grid
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    xgrid = np.linspace(minx, maxx, res)
    ygrid = np.linspace(miny, maxy, res)
    Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
    Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))

    # Plot the result as an image
    if ax is None:
        ax = plt
    if fig is None:
        fig = plt
    img = ax.imshow(Z.reshape(Xgrid.shape),
               origin='lower', aspect='auto',
               extent=[minx, maxx, miny, maxy],
               cmap='Blues')
    if legend:
        cb = fig.colorbar(img)
        cb.set_label("density")
    
    # draw diagonal
    vals = [max(minx, miny), min(maxx, maxy)]
    ax.plot(vals, vals, ls = '--', c = '.3')
    
    # add text
    spearman = spearmanr(x, y)[0]
    person = pearsonr(x, y)[0]
    text = ('Spearman Correlation = {:f}' +
            '\nPearson Correlation = {:f}').format(spearman, person)
    xpos = (maxx - minx) * .05 + minx
    ypos = - (maxy - miny) * .05 + maxy
    ax.text(xpos, ypos, text,
            verticalalignment = 'top')

def save_cor_plot(n):
    deg_counts = lin_deg(beta_counts, n + 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    deplot(beta_counts.flatten(), deg_counts.flatten(), ax = ax, res = args.res)
    plt.savefig('data/correlations/' + str(n) + '-comp.png')

if __name__ == "__main__":
    tasks = args.integers
    if len(tasks) == 1:
        save_cor_plot(tasks[0])
    elif len(tasks) > 1:
        with Pool() as pool:
            _ = list(tqdm(pool.imap(save_cor_plot, tasks), total=len(tasks)))