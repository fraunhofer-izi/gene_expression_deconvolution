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
import pickle
import re
import itertools

import h5py
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotnine as pn

from scipy.ndimage.filters import gaussian_filter1d
from scipy.special import softmax
from scipy.stats import norm

from rpy2.robjects import r

import warnings
import multiprocess as mp
from tqdm.auto import tqdm

import pymc3 as pm
from stickbreaking import StickBreaking_legacy, StickBreaking2

from gmmot import GW2
from lifelines import KaplanMeierFitter, CoxPHFitter, statistics
from lifelines.plotting import add_at_risk_counts
import gseapy as gp
from gseapy.parser import Biomart

from mc_model import mc_model


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
        fc = get_content(f)
    return fc


def get_counts(
    inFile,
    pcaFile=None,
    logscale=True,
    tumor_free=True,
    add_dev=True,
    xt_df=None,
):
    if pcaFile is None:
        pcaFile = inFile.replace("_params.hdf5", "_pca.pkl")
    dat = get_params(inFile)
    with open(pcaFile, "rb") as buff:
        pca = pickle.load(buff)
    xt = pca.inverse_transform(dat["means"]["x_t"])
    if add_dev is True and "dev_t" in dat["means"]:
        xt += dat["means"]["dev_t"]
    if logscale is False:
        xt = softmax(xt, axis=1)
    if xt_df is None:
        xt_df = mc_model(feature_limit=xt.shape[1]).counts
    xt_df.loc[:, :] = xt.T
    if tumor_free is False:
        return xt_df
    xf = pca.inverse_transform(dat["means"]["x_f"])
    if add_dev is True and "dev_f" in dat["means"]:
        xf += dat["means"]["dev_f"]
    if logscale is False:
        xf = softmax(xf, axis=1)
    xf_df = xt_df.copy()
    xf_df.loc[:, :] = xf.T
    return xt_df, xf_df


def save_counts(
    inFile,
    tag="taul",
    pcaFile=None,
    outFile=None,
    logscale=True,
    model=None,
    pathOut="/mnt/fhgfs_ribdata/user_worktmp/dominik.otto/PCa-2016",
    counts_df=None,
    tenplate=None,
):
    if outFile is None:
        outFile = inFile.replace("_params.hdf5", "_tumor_counts.csv")
    if counts_df is None:
        xt_df = get_counts(
            inFile,
            pcaFile=pcaFile,
            logscale=logscale,
            tumor_free=False,
            add_dev=False,
            xt_df=tenplate,
        )
    else:
        xt_df = counts_df
    xt_df.to_csv(outFile)
    if model is not None:
        # save as R data per cohort
        assembly = model.assembly
        pheno = model.pheno
        cohorts = pheno.CohortAbb.unique()
        for cohort in tqdm(cohorts, desc="cohorts"):
            print(f"Saving {cohort} ...")
            samp_names = pheno.index[pheno.CohortAbb == cohort]
            Counts = xt_df.loc[:, samp_names]
            outFile = (
                f"{pathOut}/"
                f"{tag}-normalized-none-{cohort}-{assembly}-kalGene-counts.csv"
            )
            print(f"... as {outFile}")
            Counts.to_csv(outFile)
            outFile = (
                f"{pathOut}/"
                f"{tag}-normalized-none-{cohort}-{assembly}-"
                f"kalGene-counts.RData"
            )
            print(f"... as {outFile}")
            r.assign("Counts", Counts)
            r(f"save(Counts, file='{outFile}')")


def get_means(inFile, pcaFile=None, logscale=True):
    if pcaFile is None:
        pcaFile = inFile.replace("_params.hdf5", "_pca.pkl")
    dat = get_params(inFile)
    with open(pcaFile, "rb") as buff:
        pca = pickle.load(buff)
    xt = pca.inverse_transform(dat["means"]["mus_t"])
    xf = pca.inverse_transform(dat["means"]["mus_f"])
    if logscale is False:
        xt = softmax(xt, axis=1)
        xf = softmax(xf, axis=1)
    return xt, xf


def multiplot(files, smooth=100, alpha=0.6, loss_padd=None):
    if not isinstance(files, dict):
        files = [files]

    def load_hist(entry):
        name, file = entry
        try:
            hist = np.loadtxt(file)
        except OSError:
            warn = "{} could not be loaded with np.loadtext({})."
            warnings.warn(warn.format(name, file), UserWarning)
            return name, None
        is_fine = np.isfinite(hist)
        if not any(is_fine):
            return name, None
        iters = np.where(is_fine)[0]
        hist = hist[is_fine]
        lb = min(hist)
        if loss_padd is not None and lb < 0:
            hist += loss_padd - lb
            lb = loss_padd
        ldf = pd.DataFrame(
            {"loss": hist, "iteration": iters, "model": [name] * len(hist)}
        )
        if smooth is not False:
            if lb > 0:
                ldf["sloss"] = np.exp(
                    gaussian_filter1d(np.log(hist), sigma=smooth)
                )
            else:
                ldf["sloss"] = gaussian_filter1d(hist, sigma=smooth)
        return name, ldf

    tasks = list(files.items())
    df = pd.DataFrame()
    with mp.Pool() as pool:
        for name, ldf in tqdm(
            pool.imap(load_hist, tasks), total=len(tasks), desc="models"
        ):
            if ldf is not None:
                df = df.append(ldf)

    def breaks(limits):
        ll = np.log10(limits)
        if (ll[1] - ll[0]) > 3:
            ll = np.round(ll)
            ex = np.linspace(ll[0], ll[1], 10)
            ex = np.round(ex)
        else:
            ex = np.linspace(ll[0], ll[1], 10)
        return 10.0 ** ex

    pl = (
        pn.ggplot(pn.aes("iteration", "loss", color="model"), df)
        + pn.geom_line(alpha=alpha)
        + pn.scale_y_log10()
        + pn.theme_minimal()
    )
    if smooth is not False:
        pl += pn.geom_line(pn.aes(y="sloss"), size=1, alpha=alpha)
    return pl, df


def pars2tumorContent(pars):
    p_f = p_t = 0.95
    trans = pm.distributions.transforms.Interval(1 - p_f, p_t)
    pa = trans.backward(pars["means"]["p_a_interval__"]).eval()
    alpha = (pa + p_f - 1) / (p_t + p_f - 1)
    return alpha


def getPheno(file, model=None):
    pars = get_params(file)
    if model is None:
        model = mc_model()
    pheno = model.pheno
    pheno["corrected tumor content"] = pars2tumorContent(pars)
    pheno["tumor content"] = pheno.tcEst
    pheno["Tumor"] = np.isclose(pheno.Tumor, 1)
    tumor_string = np.where(pheno.Tumor, "tumor", "free")
    pheno["missing tumor content"] = ~np.isfinite(pheno.tumorContent)
    pheno["Cohort - Tumor"] = [
        f"{co} {tu}" for co, tu in zip(model.pheno.Cohort, tumor_string)
    ]
    return model.pheno


def concordance(
    file,
    model=None,
    cohorts=["UKDP"],
    event_type="biochemicalRecurrence",
    event_time="bcrTime",
):
    if isinstance(cohorts, str):
        cohorts = [cohorts]
    pars = get_params(file)
    if "logHR" not in pars["means"]:
        raise TypeError(
            "The parameter in the file do not seem to contain hazard "
            "prediction."
        )
    if model is None:
        md = mc_model()
    else:
        md = model
    valid = np.logical_and(
        ~md.pheno.loc[:, [event_type, event_time]].isna().any(axis=1),
        md.pheno["blacklisted"] == 0,
    )
    chs = pd.Series(cohorts).str.upper()
    ind = np.logical_and(md.pheno["CohortAbb"].str.upper().isin(chs), valid)
    expressions = np.concatenate(
        [pars["means"]["x_t"][ind, :], pars["means"]["x_f"][ind, :]], axis=1
    )
    score = np.dot(expressions, pars["means"]["logHR"])

    time = md.pheno.loc[ind, event_time].values
    time_diffs = time[:, None] - time[None, :]

    event = md.pheno.loc[ind, event_type].values
    event1 = np.logical_and(time_diffs < 0, event[:, None])
    event2 = np.logical_and(time_diffs > 0, event[None, :])
    valid = np.tril(np.logical_or(event1, event2))

    score_diffs = score - score.T
    concordant = (score_diffs * time_diffs) < 0

    n_valid = np.squeeze(valid).sum()
    n_concordant = np.squeeze(concordant[valid]).sum()

    concordance = n_concordant / n_valid
    print(
        "{} of {} ({:.2%}) comparable pairs are concordant.".format(
            n_concordant, n_valid, concordance
        )
    )
    return concordance


def log_HR_plot(inFile, label_unit=10, log_scale_color=True):
    par = get_params(inFile)
    pca_components = par["means"]["logHR"].shape[0] >> 1
    components = range(pca_components)
    tf_components = slice(pca_components, 2 * pca_components)

    logHR_df = pd.DataFrame(index=[f"{i+1}" for i in components])
    logHR_df["tumor logHR"] = par["means"]["logHR"][components, 0]
    logHR_df["non-tumor logHR"] = par["means"]["logHR"][tf_components, 0]
    logHR_df["component"] = components
    logHR_df["label"] = [
        logHR_df.index[i] if i <= label_unit else "" for i in components
    ]
    logHR_df["tumor logHR sd"] = par["stds"]["logHR"][components, 0]
    logHR_df["non-tumor logHR sd"] = par["stds"]["logHR"][tf_components, 0]
    logHR_df["tumor Z"] = logHR_df["tumor logHR"] / logHR_df["tumor logHR sd"]
    logHR_df["non-tumor Z"] = (
        logHR_df["non-tumor logHR"] / logHR_df["tumor logHR sd"]
    )
    logHR_df["tumor p-value"] = norm.sf(abs(logHR_df["tumor Z"])) * 2
    logHR_df["non-tumor p-value"] = norm.sf(abs(logHR_df["non-tumor Z"])) * 2
    logHR_df["tumor -log10(p-value)"] = -np.log10(logHR_df["tumor p-value"])
    logHR_df["non-tumor -log10(p-value)"] = -np.log10(
        logHR_df["non-tumor p-value"]
    )

    lb = min(logHR_df["non-tumor logHR"].min(), logHR_df["tumor logHR"].min())
    ub = max(logHR_df["non-tumor logHR"].max(), logHR_df["tumor logHR"].max())
    pl = (
        pn.ggplot(
            pn.aes(
                "non-tumor logHR",
                "tumor logHR",
                color="non-tumor p-value",
                fill="tumor p-value",
                label="label",
            ),
            logHR_df,
        )
        + pn.xlim(lb, ub)
        + pn.ylim(lb, ub)
        + pn.geom_abline()
        + pn.geom_point()
        + pn.theme_minimal()
        + pn.geom_text(ha="left", va="bottom", color="black")
    )
    if log_scale_color:
        pl += pn.scale_color_cmap(trans="log")
        pl += pn.scale_fill_cmap(trans="log")

    lb = min(
        logHR_df["non-tumor -log10(p-value)"].min(),
        logHR_df["tumor -log10(p-value)"].min(),
    )
    ub = max(
        logHR_df["non-tumor -log10(p-value)"].max(),
        logHR_df["tumor -log10(p-value)"].max(),
    )
    pl_p = (
        pn.ggplot(
            pn.aes(
                "non-tumor -log10(p-value)",
                "tumor -log10(p-value)",
                color="component",
                label="label",
            ),
            logHR_df,
        )
        + pn.geom_point()
        + pn.xlim(lb, ub)
        + pn.ylim(lb, ub)
        + pn.theme_minimal()
        + pn.geom_text(ha="left", va="bottom", color="black")
    )
    return pl, pl_p, logHR_df


def gene_log_HR_plot(inFile, pcaFile=None, model=None):
    # get logHRs
    par = get_params(inFile)
    pca_components = par["means"]["logHR"].shape[0] >> 1
    components = range(pca_components)
    tf_components = slice(pca_components, 2 * pca_components)

    t_logHR = par["means"]["logHR"][components, 0]
    tf_logHR = par["means"]["logHR"][tf_components, 0]

    t_logHR_sd = par["stds"]["logHR"][components, 0]
    tf_logHR_sd = par["stds"]["logHR"][tf_components, 0]

    # get pca
    if pcaFile is None:
        pcaFile = inFile.replace("_params.hdf5", "_pca.pkl")
    with open(pcaFile, "rb") as buff:
        pca = pickle.load(buff)

    # prep dataframe
    n_genes = pca.components_.shape[1]
    if model is None:
        logHR_df = pd.DataFrame(index=[f"{i+1}" for i in range(n_genes)])
    else:
        logHR_df = pd.DataFrame(index=model.counts.index)
    logHR_df["tumor logHR"] = pca.inverse_transform(t_logHR)
    logHR_df["non-tumor logHR"] = pca.inverse_transform(tf_logHR)
    logHR_df["tumor logHR sd"] = np.sqrt(
        np.sum((pca.components_ * t_logHR_sd[:, None]) ** 2, axis=0)
    )
    logHR_df["non-tumor logHR sd"] = np.sqrt(
        np.sum((pca.components_ * tf_logHR_sd[:, None]) ** 2, axis=0)
    )
    logHR_df["tumor Z"] = logHR_df["tumor logHR"] / logHR_df["tumor logHR sd"]
    logHR_df["non-tumor Z"] = (
        logHR_df["non-tumor logHR"] / logHR_df["tumor logHR sd"]
    )
    logHR_df["tumor p-value"] = norm.sf(abs(logHR_df["tumor Z"])) * 2
    logHR_df["non-tumor p-value"] = norm.sf(abs(logHR_df["non-tumor Z"])) * 2

    # make plot
    lb = min(logHR_df["non-tumor logHR"].min(), logHR_df["tumor logHR"].min())
    ub = max(logHR_df["non-tumor logHR"].max(), logHR_df["tumor logHR"].max())
    pl = (
        pn.ggplot(pn.aes("non-tumor logHR", "tumor logHR"), logHR_df)
        + pn.xlim(lb, ub)
        + pn.ylim(lb, ub)
        + pn.theme_minimal()
        + pn.geom_point(alpha=0.3, color="red")
        + pn.geom_abline()
    )
    return pl, logHR_df


def expression_plot(
    inFile,
    cp1=1,
    cp2=2,
    model=None,
    draw_distribution=True,
    draw_points=True,
    max_kernel_alpha=0.5,
    color="expression",
):
    par = get_params(inFile)
    pl = (
        pn.ggplot(pn.aes(f"CP {cp1}", f"CP {cp2}", color=color))
        + pn.theme_minimal()
    )
    df = None
    kdf = None
    if draw_points:
        if model is None:
            index = [
                f"sample {i+1}" for i in range(par["means"]["x_t"].shape[0])
            ]
            if color != "expression":
                raise Exception(
                    "A model must be passed to color other that by expression."
                )
        else:
            index = model.counts.columns
        columns = [f"CP {i+1}" for i in range(par["means"]["x_t"].shape[1])]
        df_t = pd.DataFrame(par["means"]["x_t"], index=index, columns=columns)
        df_t["expression"] = "tumor"
        df_tf = pd.DataFrame(par["means"]["x_f"], index=index, columns=columns)
        df_tf["expression"] = "non-tumor"
        df = pd.concat([df_t, df_tf])
        if model is not None:
            df = df.merge(
                model.pheno, "left", left_index=True, right_index=True
            )
        pl += pn.geom_point(data=df, alpha=0.3)
    if draw_distribution:
        n_kernel = 0
        for var in sorted(par["means"]):
            n_kernel += "mus_f" in var

        if "altStick" in par["note"] and not par["note"]["altStick"]:
            tf = StickBreaking_legacy()
        else:
            tf = StickBreaking2()
        elipses = list()
        elipse_t = np.linspace(0, 2 * np.pi, 100)

        for tissue_type in ["t", "f"]:
            weights = tf.backward(
                par["means"][f"w_{tissue_type}_stickbreaking__"]
            ).eval()
            n_dim = par["means"][f"x_{tissue_type}"].shape[1]
            for kernel in range(n_kernel):
                # get covariance elipse parameters
                packed_cov = par["means"][
                    f"packed_L_{tissue_type}_{kernel}_cholesky-cov-packed__"
                ]
                lower = pm.expand_packed_triangular(
                    n_dim, packed_cov, lower=True
                ).eval()
                cov = np.dot(lower, lower.T)[[cp1 - 1, cp2 - 1], :][
                    :, [cp1 - 1, cp2 - 1]
                ]
                var, U = np.linalg.eig(cov)
                theta = np.arccos(np.abs(U[0, 0]))

                # parametrize elipse
                width = 2 * np.sqrt(5.991 * var[0])
                hight = 2 * np.sqrt(5.991 * var[1])

                density = weights[kernel] / width * hight

                x = width * np.cos(elipse_t)
                y = hight * np.sin(elipse_t)

                # rotation
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))
                path = np.dot(R, np.array([x, y]))

                # position
                pos = par["means"][f"mus_{tissue_type}_{kernel}"]
                path += pos[[cp1 - 1, cp2 - 1]][:, None]

                # make data frame
                path_df = pd.DataFrame(
                    {f"CP {cp1}": path[0, :], f"CP {cp2}": path[1, :]}
                )
                path_df["kernel"] = kernel
                path_df["density"] = density
                path_df["expression"] = (
                    "tumor" if tissue_type == "t" else "non-tumor"
                )
                path_df["expression-kernel"] = (
                    f"tumor {kernel}"
                    if tissue_type == "t"
                    else f"non-tumor {kernel}"
                )
                elipses.append(path_df)
        kdf = pd.concat(elipses)
        density_scale = max_kernel_alpha / kdf["density"].max()
        kdf["density"] *= density_scale
        pl += pn.geom_polygon(
            pn.aes(
                fill="expression", group="expression-kernel", alpha="density"
            ),
            data=kdf,
        )
        pl += pn.scale_alpha_continuous(range=(0, max_kernel_alpha))
    return pl, df, kdf


def find_optimal_projection(inFile):
    """
    The function calculates the "A Wasserstein-type distance"
    (s. https://arxiv.org/pdf/1907.05254.pdf)
    between the gaussian mixture distributions characterizing
    tumor and non-tumor tissue for each selection of 2 components.
    It returns the w components with the aximal statistical distance
    between the two distribution for visualization purposes,
    e.g., expression_plot.
    """
    par = get_params(inFile)

    n_components = par["means"]["mus_f_0"].shape[0]
    n_kernel = 0
    for var in sorted(par["means"]):
        n_kernel += "mus_f" in var

    if "altStick" in par["note"] and not par["note"]["altStick"]:
        tf = StickBreaking_legacy()
    else:
        tf = StickBreaking2()

    weights = dict()
    means = dict()
    covs = dict()
    for tissue_type in ["t", "f"]:
        weights[tissue_type] = tf.backward(
            par["means"][f"w_{tissue_type}_stickbreaking__"]
        ).eval()
        means[tissue_type] = list()
        covs[tissue_type] = list()
        for kernel in range(n_kernel):
            means[tissue_type].append(
                par["means"][f"mus_{tissue_type}_{kernel}"]
            )
            # get covariance elipse parameters
            packed_cov = par["means"][
                f"packed_L_{tissue_type}_{kernel}_cholesky-cov-packed__"
            ]
            lower = pm.expand_packed_triangular(
                n_components, packed_cov, lower=True
            ).eval()
            cov = np.dot(lower, lower.T)
            covs[tissue_type].append(cov)
        means[tissue_type] = np.stack(means[tissue_type])
        covs[tissue_type] = np.stack(covs[tissue_type])

    def get_distance(pair):
        cp1, cp2 = pair
        mean_t = means["t"][:, [cp1, cp2]]
        mean_f = means["f"][:, [cp1, cp2]]
        cov_t = covs["t"][:, [cp1, cp2], :][:, :, [cp1, cp2]]
        cov_f = covs["f"][:, [cp1, cp2], :][:, :, [cp1, cp2]]
        _, distance = GW2(
            weights["t"], weights["f"], mean_t, mean_f, cov_t, cov_f
        )
        return cp1, cp2, distance

    pairs = itertools.combinations(range(n_components), 2)
    total_pairs = int((n_components ** 2 - n_components) / 2)
    results = list()
    with mp.Pool() as pool:
        for cp1, cp2, distance in tqdm(
            pool.imap(get_distance, pairs),
            total=total_pairs,
            desc="projections",
        ):
            results.append({"cp1": cp1 + 1, "cp2": cp2 + 1, "GW2": distance})
    result_df = pd.DataFrame(results, index=range(total_pairs))
    max_dist = result_df["GW2"].argmax()
    max_cp1 = result_df["cp1"][max_dist]
    max_cp2 = result_df["cp2"][max_dist]
    return max_cp1, max_cp2, result_df


def kernel_stats(inFile, log_scale=True):
    par = get_params(inFile)

    n_kernel = 0
    for var in sorted(par["means"]):
        n_kernel += "mus_f" in var

    tf = pm.distributions.transforms.StickBreaking()

    dfs = list()
    for tissue_type in ["t", "f"]:
        weights = tf.backward(
            par["means"][f"w_{tissue_type}_stickbreaking__"]
        ).eval()
        n_dim = par["means"][f"x_{tissue_type}"].shape[1]
        volumes = list()
        for kernel in range(n_kernel):
            # get covariance elipse parameters
            packed_cov = par["means"][
                f"packed_L_{tissue_type}_{kernel}_cholesky-cov-packed__"
            ]
            lower = pm.expand_packed_triangular(
                n_dim, packed_cov, lower=True
            ).eval()
            cov = np.dot(lower, lower.T)
            volume = np.linalg.det(cov)
            volumes.append(volume)
        type_df = pd.DataFrame(
            {
                "tissue": "tumor" if tissue_type == "t" else "non-tumor",
                "weight": weights,
                "volume": volumes,
            },
            index=[f"kernel {i}" for i in range(n_kernel)],
        )
        dfs.append(type_df)
    df = pd.concat(dfs)
    pl = (
        pn.ggplot(pn.aes("volume", "weight", color="tissue"), df)
        + pn.geom_point()
    )
    if log_scale:
        pl += pn.scale_y_log10()
        pl += pn.scale_x_log10()
    pl += pn.theme_minimal()
    return pl, df


def kaplan_meier(
    file,
    model=None,
    cohorts=["UKDP"],
    event_type="biochemicalRecurrence",
    event_time="bcrTime",
    figsize=(9, 6),
):
    if isinstance(cohorts, str):
        cohorts = [cohorts]
    if model is None:
        md = mc_model()
    else:
        md = model

    valid = np.logical_and(
        ~md.pheno.loc[:, [event_type, event_time]].isna().any(axis=1),
        md.pheno["blacklisted"] == 0,
    )
    chs = pd.Series(cohorts).str.upper()
    ind = np.logical_and(md.pheno["CohortAbb"].str.upper().isin(chs), valid)
    mpheno = md.pheno.loc[ind, :].copy()

    if file.endswith(".tsv"):
        # this is a score file
        score_df = pd.read_csv(file, delimiter="\t", index_col="ID")
        score = score_df.loc[mpheno.index, "score"]
        ind[ind] = ~score.isna()
        score = score[~score.isna()].values
        mpheno = md.pheno.loc[ind, :].copy()
    else:
        pars = get_params(file)
        if "logHR" not in pars["means"]:
            raise TypeError(
                "The parameter in the file do not seem to contain hazard "
                "prediction."
            )
        expressions = np.concatenate(
            [pars["means"]["x_t"][ind, :], pars["means"]["x_f"][ind, :]],
            axis=1,
        )
        score = np.dot(expressions, pars["means"]["logHR"])[:, 0]

    event = md.pheno.loc[ind, event_type].values
    time = md.pheno.loc[ind, event_time].values / 365.25

    # Grouping
    threshold = np.median(score)
    grouping = score > threshold
    g1 = grouping
    g2 = ~grouping

    # Kaplan Mayer Plot
    kmfh = KaplanMeierFitter()
    kmfh.fit(time[g1], event[g1], label="High Hazard")
    figure = kmfh.plot(figsize=figsize)
    kmfl = KaplanMeierFitter()
    kmfl.fit(time[g2], event[g2], label="Low Hazard")
    figure = kmfl.plot(ax=figure)
    plt.xlabel("years")
    add_at_risk_counts(kmfh, kmfl, ax=figure)

    # Cox Regression
    mpheno["score"] = score
    cph = CoxPHFitter()
    cph.fit(
        mpheno, duration_col=event_time, event_col=event_type, formula="score"
    )

    # logrank test
    logr = statistics.logrank_test(
        mpheno.loc[g1, event_time],
        mpheno.loc[g2, event_time],
        mpheno.loc[g1, event_type],
        mpheno.loc[g2, event_type],
    )

    print(
        "Cohorts: {}, event: {}, time: {}".format(
            cohorts, event_type, event_time
        )
    )
    print("Concordance: {:.2%}".format(cph.concordance_index_))
    print("Cox p-value: {}".format(cph.summary.loc["score", "p"]))
    print("Logrank p-value: {}".format(logr.p_value))

    return figure, cph, logr


def component_gene_enrichment(
    inFile,
    genes,
    component,
    quantile=0.1,
    only_ens=True,
    gene_sets=["KEGG_2019_Human"],
    pcaFile=None,
):
    print(
        "Selecting {} qantile of component {}...".format(quantile, component)
    )
    if only_ens:
        print("...using only ENSG-features...")
        selection = ["ENSG" in gene_id for gene_id in genes]
    else:
        selection = slice(None)
    # get pca
    if pcaFile is None:
        pcaFile = inFile.replace("_params.hdf5", "_pca.pkl")
    with open(pcaFile, "rb") as buff:
        pca = pickle.load(buff)
    comp = np.abs(pca.components_[component - 1, selection])
    return vec_enrich(comp, genes[selection], quantile, gene_sets)


def vec_enrich(vec, gene_ids, quantile, gene_sets):
    ind = np.quantile(vec, quantile) > vec
    print("... {} features selected...".format(sum(ind)))
    genes = gene_ids[ind]

    # remove ens id version
    genes = [re.sub("\\..*$", "", g) for g in genes]

    print("Mapping to gene names...")
    # map ens ids to gene symbols
    bm = Biomart()
    bm_result = bm.query(
        dataset="hsapiens_gene_ensembl",
        attributes=[
            "ensembl_gene_id",
            "external_gene_name",
            "entrezgene_id",
            "go_id",
        ],
        filters={"ensembl_gene_id": genes},
    )
    gene_symbols = list(bm_result["external_gene_name"].unique())

    print("Calculating enrichment...")
    enr = gp.enrichr(
        gene_list=gene_symbols,
        gene_sets=gene_sets,
        organism="Human",
        cutoff=0.05,
    )
    return enr


def gene_log_HR_enrichment(
    inFile, genes, gene_sets, tumor=True, pcaFile=None, quantile=0.05
):
    print("Enrichment for tumor log-HR...")
    # get logHRs
    par = get_params(inFile)
    pca_components = par["means"]["logHR"].shape[0] >> 1
    components = range(pca_components)
    tf_components = slice(pca_components, 2 * pca_components)

    t_logHR = par["means"]["logHR"][components, 0]
    tf_logHR = par["means"]["logHR"][tf_components, 0]

    # get pca
    if pcaFile is None:
        pcaFile = inFile.replace("_params.hdf5", "_pca.pkl")
    with open(pcaFile, "rb") as buff:
        pca = pickle.load(buff)

    if tumor:
        comp = np.abs(pca.inverse_transform(t_logHR))
    else:
        comp = np.abs(pca.inverse_transform(tf_logHR))
    return vec_enrich(comp, genes, quantile, gene_sets)
