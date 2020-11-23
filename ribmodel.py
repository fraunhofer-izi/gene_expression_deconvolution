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
import numpy as np
import pandas as pd
import pymc3 as pm
from get_data import get_data, get_annoTab
from sklearn.decomposition import PCA
from tqdm import tqdm
from scipy.special import digamma
from property_deps import property_deps

# inherits from ABC for abstractmethod
class ribmodel:
    """model base class"""

    @property_deps('model', 'seqDepth', 'dirichlet_alpha', 'dcomp')
    def counts(self):
        """The raw counts."""
        counts, pheno = self.updateData()
        return counts

    @property_deps('counts')
    def feature_limit(self):
        return None

    @property_deps('E_log_p', 'E_p', 'pca')
    def dirichlet_alpha(self):
        return self.counts.values.T + 1

    @property_deps('log_E_p')
    def E_p(self):
        alpha = self.dirichlet_alpha
        E_p = alpha / alpha.sum(axis = 1, keepdims = True)
        return E_p

    @property_deps('tau_E_log_p')
    def E_log_p(self):
        alpha = self.dirichlet_alpha
        return digamma(alpha) - digamma(alpha.sum(axis = 1, keepdims = True))

    @property_deps('proj_counts', 'tau_log_E_p')
    def log_E_p(self):
        return np.log(self.E_p)

    @property_deps('model')
    def pheno(self):
        """The clinical/pheno data of the samples."""
        counts, pheno = self.updateData()
        return pheno

    @property_deps('nCounts')
    def seqDepth(self):
        """The total number of reads per sample."""
        return np.sum(self.counts, axis = 0)

    @property_deps()
    def nCounts(self):
        """Counts normalized by sequencing depth."""
        return self.counts / self.seqDepth

    @property_deps('pheno')
    def tumor_grid_resolution(self):
        """maximal relative grid distance relative to gap size"""
        return 1e-1

    @property_deps('pca')
    def pca_solver(self):
        return 'full'

    @property_deps('pca')
    def whiten(self):
        return True

    @property_deps('pca')
    def n_components(self):
        return None

    @property_deps('pca')
    def pca_data_type(self):
        return 'tau_log_E_p'

    @property_deps('proj_counts')
    def pca(self):
        if self.n_components is None and self.n_components!=0:
            pca = PCA(whiten=self.whiten, svd_solver=self.pca_solver)
        else:
            pca = PCA(whiten=self.whiten, svd_solver=self.pca_solver,
                     n_components=self.n_components)
        if self.pca_data_type == 'tau_log_E_p':
            return pca.fit(self.tau_log_E_p)
        elif self.pca_data_type == 'tau_E_log_p':
            return pca.fit(self.tau_E_log_p)
        else:
            raise Exception(f'Unknown data type {self.pca_data_type}.')

    @property
    def raw_counts(self):
        return self.counts.values.T

    @property_deps()
    def dcomp(self):
        """Number of components for reconstruction from projection."""
        return self.raw_counts.shape[0] - 1

    @property_deps()
    def proj_counts(self):
        return self.phi(self.log_E_p)

    @property_deps()
    def proj_whiten_counts(self):
        return self.phi(self.log_E_p, whiten=True)

    @property_deps()
    def tau_log_E_p(self):
        return proj_O(self.log_E_p)

    @property_deps()
    def tau_E_log_p(self):
        return proj_O(self.E_log_p)

    @property_deps('counts')
    def only_tumor(self):
        return True

    @property_deps('counts')
    def only_control(self):
        return False

    @property
    def deg_counts(self):
        X = proj_counts[:, 0:self.dcomp]
        return inv_beta(rev_O(self.inv_phi(X)))

    def phi(self, X, ncomp=None, whiten=False):
        if self.pca.mean_ is not None:
            X = X - self.pca.mean_
        if (ncomp is not None and
            self.pca.n_components_ is not None and
            self.n_components!=0 and
            ncomp < self.pca.n_components_):
                ind = range(ncomp)
                X_transformed = np.dot(X, self.pca.components_[ind, :].T)
                if whiten:
                    X_transformed /= np.sqrt(self.pca.explained_variance_[ind])
        else:
            X_transformed = np.dot(X, self.pca.components_.T)
            if whiten:
                X_transformed /= np.sqrt(self.pca.explained_variance_)
        return X_transformed

    def inv_phi(self, X):
        ind = range(X.shape[1])
        return np.dot(X, self.pca.components_[ind, :]) + self.pca.mean_

    def degenerate(self, X, ncomp, m=1e8):
        return inv_beta(rev_O(self.inv_phi(self.phi(beta(X), ncomp))), m)

    def fast_deg(beta_counts, ncomp, m=1e8):
        return inv_beta(rev_O(self.inv_phi(self.phi(beta_counts, ncomp))), m)

    def lin_deg(beta_counts, ncomp):
        return rev_O(self.inv_phi(self.phi(beta_counts, ncomp)))

    def updateData(self):
        """Loads and sets the counts and pheno properties."""
        counts, pheno = get_data(cohort = self.cohort, assembly = self.assembly,
                                 normalization = self.normalization, filter=self.filter,
                                 counter = self.counter, filterFile = self.filterFile,
                                 sync=True)
        if self.only_tumor:
            # remove sample without Tumor information
            hasValue = [t in [0, 1] for t in pheno['Tumor']]
            hasValue = hasValue | np.isfinite(pheno['tumorContent'])
            pheno = pheno[hasValue]
            counts = counts.loc[:, hasValue]
        self.counts = counts
        if self.feature_limit is not None and self.feature_limit != 0:
            masses = np.sum(counts, axis=1)
            index = masses.argsort()[-self.feature_limit:]
            masses[np.sort(index)]
            counts = counts.iloc[np.sort(index), :]
        self.pheno = pheno
        if self.only_tumor:
            self.calculateTCRes()
        return counts, pheno

    def check_consistancy(self):
        if all(counts.columns.values == pheno.index.values):
            return True
        else:
            return False

    class consistancy_error(Exception):
        def __str__(self):
            return "There is a inconsistancy between the count and pheno data."

    def calculateTCRes(self):
        """calculate the resolution of tumor content estimates per cohort."""
        pheno = self.pheno
        # make tumor content estimate
        pheno['tcEst'] = pheno['tumorContent'] / 100
        # get cohort wise resolution
        missing = np.isnan(pheno['tumorContent'])
        for c in pheno['Cohort'].unique():
            index = (pheno['Cohort'] == c) & (~missing)
            n = 0
            while True:
                n += 1
                shouldBeInt = n * pheno.loc[index, 'tcEst'].values
                upper_bound = np.floor(shouldBeInt) + self.tumor_grid_resolution
                lower_bound = np.ceil(shouldBeInt) - self.tumor_grid_resolution
                colse = (shouldBeInt <= upper_bound) | (lower_bound <= shouldBeInt)
                if all(colse):
                    break
            pheno.loc[index, 'tcRes'] = n
        # fill missings
        pheno.loc[missing, 'tcEst'] = pheno.loc[missing, 'Tumor']
        pheno.loc[missing, 'tcRes'] = 1
        self.pheno = pheno

    @property_deps()
    def model(self):
        return self._make_model()

    def _make_model(self):
        """Returns the pymc3 model"""
        pass

    @property_deps('step')
    def step_type(self):
        if not hasattr(self, '_step_type'):
            self._step_type = 'NUTS'
        return 'NUTS'

    @property_deps()
    def step(self):
        with self.model:
            step = getattr(pm, self.step_type)()
            pm.init_nuts()
        return step

    trace = None

    def sample(self, sample = 1000, **kwargs):
        model = self.model
        print('Sampling...')
        self.trace = pm.sample(sample, trace = self.trace,
                               step = self.step, model = model, **kwargs)
        return self.trace

    @property_deps('counts', 'pheno')
    def cohort(self):
        return 'pcap'

    @property_deps('counts', 'pheno', 'annotation_mapping', 'meta_result')
    def assembly(self):
        return 'gencodeV27'

    @property_deps()
    def meta_result(self):
        return get_annoTab(self.meta_result_file, self.assembly)

    @property_deps('annotation_mapping')
    def meta_result_file(self):
        return None

    @property_deps()
    def annotation_mapping(self):
        data = pd.read_csv(self.annotation_mapping_csv,
                           sep = ';', memory_map = True, low_memory = False)
        result = data
        for col in data.columns:
            if self.assembly.lower() in col.lower():
                result = data.set_index(col)
        return result

    @property_deps('annotation_mapping')
    def annotation_mapping_csv(self):
        return None

    @property_deps('counts', 'pheno')
    def normalization(self):
        return 'raw'

    @property_deps('counts', 'pheno')
    def counter(self):
        return 'kalGene'

    @property_deps('counts')
    def filterFile(self):
        return None

    @property_deps('counts')
    def filter(self):
        return 'none'

    @property_deps('events')
    def event_cohorts(self):
        '''Cohorts considered for events.'''
        return ['UKD1', 'PRAD', 'OSR1', 'UKD4', 'OSR2']

    @property_deps('events')
    def event_type(self):
        '''A column in pheno table indicating the event.'''
        return 'cep'

    @property_deps('events')
    def event_time(self):
        '''Name of column in pheno table with time to event.'''
        return 'cept'

    @property_deps('model')
    def events(self):
        cohorts = pd.Series(self.event_cohorts).str.upper()
        ind = self.pheno['CohortAbb'].str.upper().isin(cohorts)
        DoDs = np.isclose(self.pheno[self.event_type].astype(np.float), 1)
        DoDs &= ~self.pheno[self.event_time].isna()
        DoDs &= np.isclose(self.pheno['blacklisted'], 0)
        events = list()
        for event in np.flatnonzero(ind & DoDs):
            own_time = self.pheno[self.event_time][event]
            sample = self.pheno.index[event]
            patient = self.pheno['PatientID'][event]
            others = ind & (self.pheno[self.event_time] >= own_time)
            mark = others.copy()
            mark[event] = False
            among = np.flatnonzero(~mark[others])[0]
            events.append({
                'index':event,
                'sample': self.pheno.index[event],
                'patient': self.pheno['PatientID'][event],
                'time':own_time,
                'type':self.event_type,
                'mask':others,
                'index_among':among
            })
        return events

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def proj_O(log_counts):
    """Projection to the O^n plane (orthogonal plane to the 1-vector)."""
    n = log_counts.shape[1]
    diff = log_counts.sum(axis = 1, keepdims = True) / n
    return log_counts - diff

def rev_O(o_counts):
    """The tau-function. Maps the O^n plane to log(n-simplex)."""
    level = np.exp(o_counts).sum(axis = 1, keepdims = True)
    t = np.log(level)
    return o_counts - t

def beta(counts):
    seq_depth = np.sum(counts, axis = 1, keepdims = True)
    alpha = counts + 1
    sub = np.log(seq_depth + counts.shape[1])
    nom = np.log(alpha)
    result = nom - sub
    return result, seq_depth

def dgamma(counts):
    seq_depth = np.sum(counts, axis = 1, keepdims = True)
    alpha = counts + 1
    sub = digamma(seq_depth + counts.shape[1])
    nom = digamma(alpha)
    result = nom - sub
    return result, seq_depth

def inv_beta(log_alpha, seq_depth = 1e8):
    #bound = np.percentile(log_alpha, 20, axis = 1, keepdims = True)
    n = log_alpha.shape[1]
    alpha = np.exp(log_alpha)
    result = alpha*(seq_depth + n) - 1
    result = np.clip(result, 0, None)
    return result

def re_counts(log_p, seq_depth = 1e8):
    return np.exp(log_p) * seq_depth
