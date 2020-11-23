#! /usr/bin/env python

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

if __name__ == "__main__":
    import os
    import tutil
    import argparse

    desc = "Infer parameter distributions for a dimensionally reduced model."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "tag",
        metavar="name",
        type=str,
        nargs="?",
        help="tag the result with a name",
        default="nameless",
    )
    parser.add_argument(
        "--ncomp",
        help="number of components " "(use maximum if 0, default=275)",
        type=int,
        default=275,
        metavar="integer",
    )
    parser.add_argument(
        "--nModes",
        help="number of modes for tumor expression "
        "characterisation (default=10).",
        type=int,
        default=10,
        metavar="integer",
    )
    parser.add_argument(
        "--nFeatures",
        help="only use n highes expressed "
        "features (use all if 0, default=0)",
        type=int,
        default=0,
        metavar="integer",
    )
    parser.add_argument(
        "--sampDev", help="sample deviations", action="store_true"
    )
    parser.add_argument(
        "--noDev", help="do not add any deviation", action="store_true"
    )
    parser.add_argument(
        "--kmInit",
        help="Use k-means clustering to init "
        "charachterization gaussion mixtrure means.",
        action="store_true",
    )
    parser.add_argument(
        "--sameKernels",
        help="use the same kernels for tumor " "and non-tumor characterizer",
        action="store_true",
    )
    parser.add_argument(
        "--svgd",
        help="use SVGD instead of ADVI with `int` " "particles",
        type=float,
        default=0.0,
        metavar="int",
    )
    parser.add_argument(
        "--saveModel", help="save model with all the data", action="store_true"
    )
    parser.add_argument(
        "--seqFac",
        help="factor for sequencing depth " "(default=1)",
        type=float,
        default=1,
        metavar="float",
    )
    parser.add_argument(
        "--DirichletFactor",
        help="Factor for Dirichlet prior "
        "of gaussian mixture. Larger values force the mixture to use all "
        "kernels with more equal weights (default=1).",
        type=float,
        default=1,
        metavar="float",
    )
    parser.add_argument(
        "--relax",
        help="factor for standard variance to relax "
        "charcterization mean prior (default=1)",
        type=float,
        default=5,
        metavar="float",
    )
    parser.add_argument(
        "--maxIter",
        help="maximum number of iterations " "(default=1e6)",
        type=float,
        default=1e6,
        metavar="int",
    )
    parser.add_argument(
        "--startFile",
        help="hdf5 file with initial means",
        type=str,
        default=None,
        metavar="path",
    )
    parser.add_argument(
        "--cohort",
        help="The sample cohort to use, " "passed to get_data (default=pcap).",
        type=str,
        default="pcap",
        metavar="cohort",
    )
    parser.add_argument(
        "--counter",
        help="the tool the cohort experssions "
        "were counted with, passed to get_data (default=kalGene)",
        type=str,
        default="kalGene",
        metavar="counter",
    )
    parser.add_argument(
        "--assembly",
        help="the assembly the cohort "
        "experssions were counted on (default=gencodeV27)",
        type=str,
        default="gencodeV27",
        metavar="assembly",
    )
    parser.add_argument(
        "--filterFile",
        help="a text file listing the features " "to use",
        type=str,
        default=None,
        metavar="filterFile",
    )
    parser.add_argument(
        "--nmc",
        help="number of MC samples per advi iteration "
        "(default is the PyMC3 default which at the time of writing is 1)",
        type=int,
        default=None,
        metavar="integer",
    )
    parser.add_argument(
        "--optimizer",
        help="the stochastic optimizer to use " "(default=adam)",
        type=str,
        default="adam",
        metavar="optimizer",
    )
    parser.add_argument(
        "--hazard",
        help="Choose a hazard moden "
        "(e.g. none, cox, mk=multi_kernel, default=none).",
        type=str,
        default="none",
        metavar="hazard_model",
    )
    parser.add_argument(
        "--eventType",
        help="Type of events to consider "
        "in the hazard model. Must be a column with boolean values "
        "in the pheno data (default=cep).",
        type=str,
        default="cep",
        metavar="event_type",
    )
    parser.add_argument(
        "--eventTime",
        help="Time of events to consider "
        "in the hazard model. Must be a column with float values "
        "in the pheno data (default=cept).",
        type=str,
        default="cept",
        metavar="event_type",
    )
    parser.add_argument(
        "--withinCohort",
        help="Only compare events within "
        "the same cohort in the hazard model.",
        action="store_true",
    )
    parser.add_argument(
        "--hazardCohorts",
        nargs="+",
        metavar="cohort",
        help="Cohort appreviations as in the pheno data. "
        "Selects for which cohorts survival data should be used, "
        "(e.g. ukd1 prad, default=UKD1 PRAD OSR1 UKD4 OSR2)",
        default=["UKD1", "PRAD", "OSR1", "UKD4", "OSR2"],
    )
    parser.add_argument(
        "--learnrate",
        help="learning rate of the stochastic "
        "optimizer (default is the PyMC3 default)",
        type=float,
        default=None,
        metavar="float",
    )
    parser.add_argument(
        "--nwin",
        help="window size for the stochastic "
        "optimizer that uses windows (default=50)",
        type=int,
        default=50,
        metavar="int",
    )
    parser.add_argument(
        "--pf",
        help="probability for pathologist to correctly "
        "identify non tumor tissue (default=0.95)",
        type=float,
        default=0.95,
        metavar="float",
    )
    parser.add_argument(
        "--pt",
        help="probability for pathologist to correctly "
        "identify tumor tissue (default=0.95:)",
        type=float,
        default=0.95,
        metavar="float",
    )
    parser.add_argument(
        "--resScale",
        help="scale the resolution of the tumor "
        "content estimate (default=1.0)",
        type=float,
        default=1.0,
        metavar="float",
    )
    parser.add_argument(
        "--useMultinomial",
        help="Use the multinomial dist for "
        "data instead of Dirichlet potential. Not recommendet since data "
        "is rounded to integer values.",
        action="store_true",
    )
    args = parser.parse_args()
    # let theano compile somewhere else
    compileDir = os.path.join(
        os.getenv("HOME"), ".theano_dr_{}".format(args.tag)
    )
    tutil.setFlag("base_compiledir", compileDir)
    tutil.setFlag("blas.ldflags", "'-L/usr/lib/ -lblas'")
    print("Running DR model with arguments:")
    print(args)
    import warnings
    import logging

    logging.warn(
        "Ignoring FutureWarnings since the deprciaded theano throws a lot."
    )
    warnings.simplefilter(action="ignore", category=FutureWarning)
    print("Importing python packages...")
from ribmodel import ribmodel
import theano
from pymc3.theanof import floatX
from stickbreaking import *
import theano.tensor as tt
import numpy as np
import pymc3 as pm
from property_deps import property_deps
import _pickle as pickle
import h5py
from sklearn.cluster import KMeans


class mc_model(ribmodel):
    @property_deps("model")
    def p_f(self):
        return 0.95

    @property_deps("model")
    def p_t(self):
        return 0.95

    @property_deps("model")
    def res_scale(self):
        return 1.0

    @property_deps("model")
    def sample_deviation(self):
        return False

    @property_deps("model")
    def kmeansInit(self):
        return True

    @property_deps("model")
    def no_deviations(self):
        return False

    @property_deps("model")
    def seq_depth_factor(self):
        return 1

    @property_deps("model")
    def hazard_model(self):
        return "none"

    def _make_model(self):
        pca = self.pca
        mCounts = np.int_(self.counts * self.seq_depth_factor)
        n_dim = pca.n_components_
        n_modes = self.n_modes
        n_samp = mCounts.shape[1]
        n_features = mCounts.shape[0]
        if self.kmeansInit:
            sd_factor = 2 / n_modes
        else:
            sd_factor = 2

        print("Defining model constants...")
        if pca.whiten:
            rot = np.sqrt(pca.explained_variance_[:, None]) * pca.components_
            rot = theano.shared(floatX(rot))
            cSd = floatX(1)
            tcov = np.eye(n_dim)[np.tril_indices(n_dim)] * sd_factor
        else:
            rot = theano.shared(floatX(pca.components_))
            cSd = floatX(np.sqrt(pca.explained_variance_))
            tcov = (
                np.diag(pca.explained_variance_)[np.tril_indices(n_dim)]
                * sd_factor
            )
        shift = theano.shared(
            floatX(pca.mean_[None, :]), broadcastable=(True, False)
        )

        multiNn = np.sum(mCounts, axis=0)
        print("Counts shape:")
        print(mCounts.shape)
        lcounts = floatX(self.pca.transform(self.tau_log_E_p))
        print("Latent counts shape:")
        print(lcounts.shape)
        high_tumor = self.pheno["tcEst"] > 0.8
        low_tumor = self.pheno["tcEst"] < 0.2
        if self.kmeansInit:
            km = KMeans(
                n_clusters=n_modes, random_state=0, tol=1e-10, max_iter=100
            )
            mus_tumor = km.fit(lcounts[high_tumor, :]).cluster_centers_
            mus_free = km.fit(lcounts[low_tumor, :]).cluster_centers_
        else:
            mus_tumor = np.repeat(
                np.mean(lcounts[high_tumor, :], axis=0)[None, :], 10, axis=0
            )
            mus_free = np.repeat(
                np.mean(lcounts[low_tumor, :], axis=0)[None, :], 10, axis=0
            )
        mus_tumor = floatX(mus_tumor)
        mus_free = floatX(mus_free)
        try:
            chol_tumor = floatX(
                np.linalg.cholesky(np.cov(lcounts[high_tumor, :].T))
            )
            chol_tumor = chol_tumor[np.tril_indices(n_dim)] * sd_factor
        except np.linalg.LinAlgError:
            print(
                "Seems we have to few HIGH tumor content samples to infer a starting covariance."
            )
            chol_tumor = tcov
        try:
            chol_free = floatX(
                np.linalg.cholesky(np.cov(lcounts[low_tumor, :].T))
            )
            chol_free = chol_free[np.tril_indices(n_dim)] * sd_factor
        except np.linalg.LinAlgError:
            print(
                "Seems we have to few LOW tumor content samples to infer a starting covariance."
            )
            chol_free = tcov
        md = self.tau_log_E_p - pca.mean_[None, :]
        dev = md - np.dot(np.dot(md, pca.components_.T), pca.components_)
        dev_std = np.std(dev, axis=0)
        dev_mean = np.mean(dev, axis=0)
        if self.no_deviations is True:
            dev_f = dev_t = None
        else:
            dev_f = dev_t = theano.shared(floatX(dev))

        p_f = floatX(self.p_f)
        p_t = floatX(self.p_t)
        sparsity = floatX(1)
        n = floatX(self.pheno["tcRes"].values[:, None] * self.res_scale)
        tc = floatX(self.pheno["tcEst"].values[:, None])
        lb = floatX(1 - p_f)
        ub = floatX(p_t)
        padding = 1e-1 * (ub - lb)
        pa_start = ((n * tc) + 1) / (n + 2)
        pa_start = np.where(pa_start < lb, lb + padding, pa_start)
        pa_start = np.where(pa_start > ub, ub - padding, pa_start)
        pa_start = floatX(pa_start)

        def inverse_pca(X):
            return pm.math.dot(X, rot) + shift

        def pa2alpha(p_a):
            return (p_a + p_f - 1) / (p_t + p_f - 1)

        def alpha2pa(alpha):
            return (alpha * (p_t + p_f - 1)) - p_f + 1

        def mixSep(x_f, x_t, alpha, dev_f, dev_t):
            exp_f = inverse_pca(x_f)
            exp_t = inverse_pca(x_t)
            if dev_f is not None:
                exp_f += dev_f
            if dev_t is not None:
                exp_t += dev_t
            exp_f = tt.nnet.softmax(exp_f)
            exp_t = tt.nnet.softmax(exp_t)
            result = ((1 - alpha) * exp_f) + (alpha * exp_t)
            return result

        print("Making model...")
        with pm.Model() as model:
            # bounds with nummerical padding
            p_a = pm.Beta(
                "p_a",
                alpha=floatX((n * tc) + 1),
                beta=floatX((n * (1 - tc)) + 1),
                transform=pm.distributions.transforms.Interval(lb, ub),
                shape=(n_samp, 1),
                testval=pa_start,
            )
            alpha = pm.Deterministic("alpha", pa2alpha(p_a))
            sdd = pm.HalfNormal.dist(sd=cSd * self.relax_prior)

            x_f_comps = list()
            for i in range(n_modes):
                mus_f = pm.Normal(
                    "mus_f_{}".format(i),
                    mu=0,
                    sd=cSd * self.relax_prior,
                    shape=n_dim,
                    testval=mus_free[i, :],
                )
                packed_L_f = pm.LKJCholeskyCov(
                    "packed_L_f_{}".format(i),
                    n=n_dim,
                    eta=sparsity,
                    sd_dist=sdd,
                    testval=chol_free,
                )
                chol_f = pm.expand_packed_triangular(
                    n_dim, packed_L_f, lower=True
                )
                x_f_comps.append(
                    pm.MvNormal.dist(
                        mu=mus_f, chol=chol_f, shape=(n_samp, n_dim)
                    )
                )
            if n_modes > 1:
                w_f = pm.Dirichlet(
                    "w_f", a=np.ones(n_modes) * self.dirichlet_prior
                )
                x_f = pm.Mixture(
                    "x_f",
                    w=w_f,
                    comp_dists=x_f_comps,
                    shape=(n_samp, n_dim),
                    testval=lcounts,
                )
            else:
                x_f = pm.MvNormal(
                    "x_f", mu=mus_f, chol=chol_f, shape=(n_samp, n_dim)
                )

            if self.same_kernels:
                x_t_comps = x_f_comps
            else:
                x_t_comps = list()
                for i in range(n_modes):
                    mus_t = pm.Normal(
                        "mus_t_{}".format(i),
                        mu=0,
                        sd=cSd * self.relax_prior,
                        shape=n_dim,
                        testval=mus_tumor[i, :],
                    )
                    packed_L_t = pm.LKJCholeskyCov(
                        "packed_L_t_{}".format(i),
                        n=n_dim,
                        eta=sparsity,
                        sd_dist=sdd,
                        testval=chol_tumor,
                    )
                    chol_t = pm.expand_packed_triangular(
                        n_dim, packed_L_t, lower=True
                    )
                    x_t_comps.append(
                        pm.MvNormal.dist(
                            mu=mus_t, chol=chol_t, shape=(n_samp, n_dim)
                        )
                    )
            if n_modes > 1:
                w_t = pm.Dirichlet(
                    "w_t", a=np.ones(n_modes) * self.dirichlet_prior
                )
                x_t = pm.Mixture(
                    "x_t",
                    w=w_t,
                    comp_dists=x_t_comps,
                    shape=(n_samp, n_dim),
                    testval=lcounts,
                )
            else:
                x_t = pm.MvNormal(
                    "x_t", mu=mus_t, chol=chol_t, shape=(n_samp, n_dim)
                )

            if self.sample_deviation is True:
                dev_f = pm.Normal(
                    "dev_f",
                    mu=dev_mean,
                    sigma=dev_std,
                    shape=(n_samp, n_features),
                    testval=dev,
                )
                dev_t = pm.Normal(
                    "dev_t",
                    mu=dev_mean,
                    sigma=dev_std,
                    shape=(n_samp, n_features),
                    testval=dev,
                )

            if self.hazard_model == "cox":
                b = pm.Normal("logHR", mu=0, sigma=1, shape=(2 * n_dim, 1))
                for ev in self.events:
                    ind = ev["mask"].values
                    obs = np.array(ev["index_among"])
                    expressions = tt.concatenate(
                        [x_t[ind, :], x_f[ind, :]], axis=1
                    )
                    hazard = tt.exp(tt.dot(expressions, b)).T
                    evp = pm.Categorical(
                        "event_{}".format(ev["sample"]), hazard, observed=obs
                    )
            elif self.hazard_model == "mk":
                # This in not implemented and aims to model hazard with a gaussian mixture
                b = pm.Normal("kernel_weights", mu=0, sigma=1, shape=(10,))
                pass

            x = pm.Deterministic("x", mixSep(x_f, x_t, alpha, dev_f, dev_t))
            if self.use_multinomial:
                obs = pm.Multinomial(
                    "obs", n=multiNn, p=x, observed=mCounts.T, dtype="int64"
                )
            else:
                dist = pm.Dirichlet.dist(mCounts.T + 1)
                pot = pm.Potential("obs", dist.logp(x))
        return model


class SaveCallback:
    def __init__(
        self, file_base, every=100, verbose=True, learn_rate=None, note=dict()
    ):
        self.loss_hist_file = "{}_loss_hist.txt".format(file_base)
        try:
            os.remove(self.loss_hist_file)
        except NameError:
            pass
        except FileNotFoundError:
            pass
        self.params_file = "{}_params.hdf5".format(file_base)
        self.every = every
        self.learn_rate = learn_rate
        self.note = note
        self.first = True

        if verbose is True:
            print("Loss history is tracked in {}".format(self.loss_hist_file))
            print(
                "Parameters are updated every {} iterations in {}".format(
                    self.every, self.params_file
                )
            )

    def __call__(self, approx, loss, i):
        if self.learn_rate is not None:
            # upadte learing rate
            pass
        if self.first or not i % self.every:
            self.first = False
            if loss is not None and len(loss) >= self.every:
                with open(self.loss_hist_file, "a") as myfile:
                    for l in loss[-self.every :]:
                        myfile.write("{}\n".format(l))
            params = approx.params[0].eval()
            if isinstance(approx, pm.variational.approximations.MeanField):
                means = approx.mean.get_value()
                stds = approx.std.eval()
            elif isinstance(approx, pm.variational.approximations.Empirical):
                means = np.mean(params, axis=0)
                stds = np.std(params, axis=0)
            else:
                message = (
                    "Save method not implemented for approximations of type {}"
                )
                raise NotImplementedError(message.format(type(approx)))
            rmap = approx.bij.rmap
            with h5py.File(self.params_file, "w") as f:
                unnamed = f.create_group("unnamed")
                unnamed.create_dataset("means", data=means)
                unnamed.create_dataset("stds", data=stds)
                meang = f.create_group("means")
                for key, values in rmap(means).items():
                    meang.create_dataset(key, data=values)
                stdg = f.create_group("stds")
                for key, values in rmap(stds).items():
                    stdg.create_dataset(key, data=values)
                if isinstance(approx, pm.variational.approximations.MeanField):
                    unnamed.create_dataset("rhos", data=params)
                    rhog = f.create_group("rhos")
                    for key, values in rmap(params).items():
                        rhog.create_dataset(key, data=values)
                elif isinstance(
                    approx, pm.variational.approximations.Empirical
                ):
                    svgd = f.create_group("svgd")
                    svgd.create_dataset("params", data=params)
                if self.note:
                    note = f.create_group("note")
                    for key, value in self.note.items():
                        if value is None:
                            note.create_dataset(key, data="")
                        elif isinstance(value, list):
                            asciiList = [
                                n.encode("ascii", "ignore") for n in value
                            ]
                            note.create_dataset(
                                key, (len(asciiList), 1), "S10", asciiList
                            )
                        else:
                            note.create_dataset(key, data=value)


def get_params(file):
    def get_content(data):
        if isinstance(data, h5py.Dataset):
            return np.array(data)
        elif isinstance(data, h5py.Group) or isinstance(data, h5py.File):
            result = dict()
            for key, values in data.items():
                result[key] = get_content(values)
            return result
        else:
            raise TypeError("Only groups and datasets are supported.")

    with h5py.File(file, "r") as f:
        return get_content(f)


def _test_ars():
    if args.nModes < 1:
        raise ValueError(
            "The integer given through --nModes musst be at least 1."
        )


if __name__ == "__main__":
    _test_ars()
    if os.environ.get("SLURM_JOB_ID"):
        print("Slurm job id: %s" % os.environ.get("SLURM_JOB_ID"))
    if np.isclose(args.svgd, 0):
        model_type = "advi"
    else:
        model_type = "svgd"
    if args.startFile is not None and model_type == "advi":
        print("Using {} to initialize advi...".format(args.startFile))
        params = get_params(args.startFile)
        if "rhos" in params.keys():
            start = {"mu": params["means"], "rho": params["rhos"]}
        else:
            rhos = {
                key: np.log(np.expm1(values))
                for key, values in params["stds"].items()
            }
            start = {"mu": params["means"], "rho": rhos}
    elif args.startFile is not None:
        print("Using {} to initialize svgd...".format(args.startFile))
        params = get_params(args.startFile)
    print("Getting data...")
    model = mc_model(
        n_components=args.ncomp,
        sample_deviation=args.sampDev,
        seq_depth_factor=args.seqFac,
        no_deviations=args.noDev,
        n_modes=args.nModes,
        feature_limit=args.nFeatures,
        cohort=args.cohort,
        counter=args.counter,
        assembly=args.assembly,
        filterFile=args.filterFile,
        p_f=args.pf,
        p_t=args.pt,
        use_multinomial=args.useMultinomial,
        relax_prior=args.relax,
        hazard_model=args.hazard,
        res_scale=args.resScale,
        same_kernels=args.sameKernels,
        event_cohort=args.hazardCohorts,
        dirichlet_prior=args.DirichletFactor,
        kmeansInit=args.kmInit,
        event_type=args.eventType,
        event_time=args.eventTime,
        event_within_cohor=args.withinCohort,
    )
    pmodel = model.model
    if args.startFile is None:
        print("Testing...")
        print(pmodel.check_test_point())
    if args.saveModel is True:
        print("Saving the model...")
        with open("data/mc_model_{}.pkl".format(args.tag), "wb") as buff:
            pickle.dump(model, buff)
    print("Saving the pca...")
    with open("data/mc_model_{}_pca.pkl".format(args.tag), "wb") as buff:
        pickle.dump(model.pca, buff)
    print("Fitting...")
    save = SaveCallback(
        file_base="data/mc_model_{}".format(args.tag), note=vars(args)
    )
    save.note["gene_ids"] = list(model.counts.index)
    save.note["sample_ids"] = list(model.counts.columns)
    if args.learnrate:
        if args.optimizer == "adam":
            obj_optimizer = pm.adam(learning_rate=args.learnrate)
        elif args.optimizer == "adagrad_window":
            obj_optimizer = pm.adagrad_window(
                learning_rate=args.learnrate, n_win=args.nwin
            )
        elif args.optimizer == "nesterov_momentum":
            obj_optimizer = pm.nesterov_momentum(learning_rate=args.learnrate)
        elif args.optimizer == "adagrad":
            obj_optimizer = pm.adagrad_window(learning_rate=args.learnrate)
        elif args.optimizer == "momentum":
            obj_optimizer = pm.momentum(learning_rate=args.learnrate)
        else:
            raise ValueError(
                f'The given optimizer "{args.optimizer}" is unknown.'
            )
    else:
        if args.optimizer == "adam":
            obj_optimizer = pm.adam()
        elif args.optimizer == "adagrad_window":
            obj_optimizer = pm.adagrad_window(n_win=args.nwin)
        elif args.optimizer == "nesterov_momentum":
            obj_optimizer = pm.nesterov_momentum()
        elif args.optimizer == "adagrad":
            obj_optimizer = pm.adagrad_window()
        elif args.optimizer == "momentum":
            obj_optimizer = pm.momentum()
        else:
            raise ValueError(
                f'The given optimizer "{args.optimizer}" is unknown.'
            )
    if model_type == "advi":
        vinfer = pm.ADVI(model=pmodel, obj_optimizer=obj_optimizer)
        if args.startFile is not None:
            bij2 = vinfer.approx.groups[0].bij
            try:
                vinfer.approx.params[0].set_value(bij2.map(start["mu"]))
                vinfer.approx.params[1].set_value(bij2.map(start["rho"]))
            except KeyError:
                print(
                    "Found discrapency between start values and model parmeters."
                )
                rmap = vinfer.approx.bij.rmap
                model_means = rmap(vinfer.approx.params[0].eval())
                model_rhos = rmap(vinfer.approx.params[1].eval())
                un_initiated = set(model_means) - set(start["mu"])
                if un_initiated:
                    print("No start values for: {}".format(list(un_initiated)))
                no_params = set(start["mu"]) - set(model_means)
                if no_params:
                    print(
                        "No model parameters for start values: {}".format(
                            list(no_params)
                        )
                    )
                initialize = set(start["mu"]).intersection(model_means)
                if not initialize:
                    raise Exception(
                        "Could not find any start value names in the model."
                    )
                for var in initialize:
                    model_means[var] = start["mu"][var]
                    model_rhos[var] = start["rho"][var]
                vinfer.approx.params[0].set_value(bij2.map(model_means))
                vinfer.approx.params[1].set_value(bij2.map(model_rhos))
    else:
        vinfer = pm.SVGD(model=pmodel, n_particles=int(args.svgd))
        if args.startFile is not None:
            start = None
            if "svgd" in params.keys():
                candidate = params["svgd"]["params"]
                if candidate.shape == svgd.approx.params[0].shape.eval():
                    print("Using exact svgd params to init...")
                    start = candidate
            if start is None:
                print("Drawing particles to init svgd...")
                bij2 = vinfer.approx.groups[0].bij
                means = bij2.map(params["means"])
                stds = bij2.map(params["stds"])
                start = np.random.normal(
                    means, stds, size=(int(args.svgd), len(means))
                )
            vinfer.approx.params[0].set_value(start)
    with pmodel:
        aprox = vinfer.fit(
            n=int(args.maxIter),
            callbacks=[save],
            obj_n_mc=args.nmc,
            obj_optimizer=obj_optimizer,
            progressbar=True,
        )
    if args.saveModel is True:

        print(f"Attempting to save {model_type}...")
        with open(f"data/mc_model_{args.tag}_{model_type}.pkl", "wb") as buff:
            pickle.dump(aprox, buff)
