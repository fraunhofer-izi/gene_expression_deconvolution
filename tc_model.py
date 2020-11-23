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

class tc_model(ribmodel):
    def _make_model(self):
        
        tumorInd = self.pheno['Tumor'] == 1
        tumorTCs = self.pheno.loc[tumorInd, 'tcEst'].values
        tumorRes = self.pheno.loc[tumorInd, 'tcRes'].values
        nTumor = np.round(tumorTCs * tumorRes).astype(int)
        freeInd = self.pheno['Tumor'] == 0
        freeTCs = self.pheno.loc[freeInd, 'tcEst'].values
        freeRes = self.pheno.loc[freeInd, 'tcRes'].values
        nFree = np.round(freeTCs * freeRes).astype(int)
        mu = np.mean(list(tumorRes) + list(freeRes))
        sig = np.std(list(tumorRes) + list(freeRes))
        alpha_gamma = mu**2 / sig
        beta_gamma = mu / sig

        with pm.Model() as model:
            u = pm.Uniform('u', 0, 1, testval = .5, shape = 2)
            v = pm.Gamma('v', alpha = alpha_gamma, beta = beta_gamma,
                         testval = 100, shape = 2)
            alpha = pm.Deterministic('alpha', v * u)
            beta = pm.Deterministic('beta', v * (1 - u))
            p = pm.Beta('p', alpha = alpha, beta = beta, shape = 2)
            obsTumor = [pm.Binomial('obsTumor' + str(i), n = tumorRes[i], p = p[0],
                                     observed = nTumor[i])
                        for i in range(len(nTumor))]
            obsFree = [pm.Binomial('obsFree' + str(i), n = freeRes[i], p = p[1],
                                    observed = nFree[i])
                       for i in range(len(nFree))]
        return model
            
if __name__ == "__main__":
    import pickle    
    model = mv_model()
    model.trace(jobs = 1)
    with open('tc_model.pkl', 'wb') as buff:
        pickle.dump(model, buff)
