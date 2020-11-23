#!/usr/bin/env python
#SBATCH --job-name="invert digamma"
#SBATCH --out=.slurmlogs/slurm.dr_inv_digamma.%j_%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dominik.otto@izi.fraunhofer.de

import os
os.environ['OMP_NUM_THREADS'] = '1' # single thread openMP and BLAS
if 'SLURM_SUBMIT_DIR' in os.environ:
    os.chdir(os.environ['SLURM_SUBMIT_DIR'])
import numpy as np
import pickle
from models import marker_model as my_model
from tqdm import tqdm
from multiprocessing import Pool
from DR_plots import *
from rpy2.robjects import r, globalenv, numpy2ri
numpy2ri.activate()

dg, sub = dgamma(counts)
def pool_f(n):
    red = inv_phi(phi(dg, n * 10))
    deg_counts = inv_dg(red, sub, parallel = False)
    with open('data/deg_counts_sd/digamma_deg_' + str(n * 10) + '-comp.pkl', 'wb') as buff:
        pickle.dump(deg_counts, buff)
    rPath = 'data/deg_counts_sd/digamma_deg_' + str(n * 10) + '-comp.RDS'
    nr,nc = deg_counts.shape
    dg_r = r.matrix(deg_counts, nrow=nr, ncol=nc)
    r.saveRDS(dg_r, rPath)

tasks = set(range(np.floor(counts.shape[0] / 10).astype(int)))
tasks = tasks.union(range(10))

if __name__ == "__main__":
    with Pool() as pool:
        _ = list(tqdm(pool.imap_unordered(pool_f, tasks), total=len(tasks)))