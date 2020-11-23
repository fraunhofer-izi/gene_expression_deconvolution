"""
Copyright (C) 2019 Fraunhofer-Gesellschaft zur Foerderung der angewandten
Forschung e.V. acting on behalf of its Fraunhofer Institute for Cell Therapy
and Immunology (IZI).

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see http://www.gnu.org/licenses/.
"""
from ribmodel import ribmodel
if __name__ == "__main__":
    import os
    import tutil
    # let theano compile somewhere else
    compileDir = os.path.join(os.getenv('HOME'), '.theano_mvLocal')
    tutil.setFlag('base_compiledir', compileDir)
    tutil.setFlag('blas.ldflags', '"-L/usr/lib/ -lblas"')
import theano
import theano.tensor as tt
import numpy as np
import pymc3 as pm

class mv_model(ribmodel):
    def _make_model(self):
        
        from pymc3.distributions.transforms import interval
        # only the 1 % of highest expressed genes
        gc = np.sum(self.nCounts, axis = 1)
        zeros = np.any(np.int_(self.counts) == 0, axis = 1)
        nzCounts = self.counts[~zeros]
        ind = np.logical_and(~zeros, gc > np.percentile(gc, 99))
        self.feature_selection = ind
        subCounts = self.counts[ind]
        nsubCounts = self.nCounts[gc > np.percentile(gc, 99.9)]
        print('Data shape:')
        print(subCounts.shape)
        
        mCounts = np.int_(subCounts)

        multiNn = np.sum(mCounts, axis = 0)
        ldata = self.tau_log_E_p[:, ind]

        p_f = .95
        p_t = .95
        sparsity = 2 # ToDo: fit LKJCholeskyCov to corr distribution
        n = self.pheno['tcRes'].values[:, None]
        tc = self.pheno['tcEst'].values[:, None]
        cMean = np.mean(ldata, axis=0)
        #cMean.shape = cMean.shape + (1,)
        cSd = np.std(ldata, axis=0)
        #cSd.shape = cMean.shape
        #cSdMv = np.stack(np.diag(l) for l in cSd)
        # https://stats.stackexchange.com/questions/237847/what-are-the-properties-of-a-half-cauchy-distribution
        
        n_dim = mCounts.shape[0]
        n_samp = mCounts.shape[1]

        # nummerical padding
        numpad = 1e-5

        def pa2alpha(p_a):
            return (p_a + p_f - 1) / (p_t + p_f - 1)

        def alpha2pa(alpha):
            return (alpha * (p_t + p_f - 1)) - p_f + 1

        def mixCounts(x, alpha):
            return tt.sum(x * alpha, axis = 0)
        
        def mixSep(x_f, x_t, alpha):
            exp_f = tt.nnet.softmax(x_f)
            exp_t = tt.nnet.softmax(x_t)
            result = ((1 - alpha) * exp_f) + (alpha * exp_t)
            return result

        with pm.Model() as model:
            # bounds with nummerical padding
            p_a = pm.Beta('p_a', alpha=(n * tc) + 1, beta=(n * (1 - tc)) + 1,
                          transform=pm.distributions.transforms.Interval(1 - (p_f + numpad), (p_t + numpad)),
                          shape=(n_samp, 1), testval=alpha2pa(tc))
            alpha = pm.Deterministic('alpha', pa2alpha(p_a))

            mus_f = pm.Normal('mus_f', mu = cMean, sd = cSd, shape = n_dim, testval = cMean)
            mus_t = pm.Normal('mus_t', mu = cMean, sd = cSd, shape = n_dim, testval = cMean)
            sdd = pm.HalfNormal.dist(sd = cSd)
            packed_L_t = pm.LKJCholeskyCov('packed_L_t', n = n_dim,
                                           eta = sparsity, sd_dist = sdd)
            packed_L_f = pm.LKJCholeskyCov('packed_L_f', n = n_dim,
                                           eta = sparsity, sd_dist = sdd)
            chol_f = pm.expand_packed_triangular(n_dim, packed_L_f, lower=True)
            chol_t = pm.expand_packed_triangular(n_dim, packed_L_t, lower=True)
            x_f = pm.MvNormal('x_f', mu = mus_f, chol = chol_f, testval = ldata, shape = (n_samp, n_dim))
            x_t = pm.MvNormal('x_t', mu = mus_t, chol = chol_t, testval = ldata, shape = (n_samp, n_dim))

            x = pm.Deterministic('x', mixSep(x_f, x_t, alpha))

            obs = pm.Multinomial('obs', n=multiNn, p=x, observed=mCounts.T,
                                 dtype='int64', shape=mCounts.T.shape)
            
        return model

if __name__ == "__main__":
    import pickle    
    model = mv_model()
    model.trace(jobs = 1)
    with open('mv_model.pkl', 'wb') as buff:
        pickle.dump(model, buff)
