#! /usr/bin/env python

import os
from shutil import copyfile
import time
from datetime import datetime
from warnings import warn
import argparse
import numpy as np
import pandas as pd

from scipy.ndimage.filters import gaussian_filter1d
from mc_model import mc_model
from model_inspection_tools import multiplot, kaplan_meier

desc = "Record the concordnace index of mc_model ADVI inference."
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
    "--cohorts",
    nargs="+",
    metavar="cohort",
    help="Cohort appreviations as in the pheno data. "
    "Selects for which cohorts survival data should be used, "
    "(e.g. ukd1 prad, default=UKDP)",
    default=["UKDP"],
)
parser.add_argument(
    "--eventType",
    help="Type of events to consider "
    "in the hazard model. Must be a column with boolean values "
    "in the pheno data (default=biochemicalRecurrence).",
    type=str,
    default="biochemicalRecurrence",
    metavar="event_type",
)
parser.add_argument(
    "--eventTime",
    help="Time of events to consider "
    "in the hazard model. Must be a column with float values "
    "in the pheno data (default=bcrTime).",
    type=str,
    default="bcrTime",
    metavar="event_type",
)
parser.add_argument(
    "--params",
    help="Parameter file.hdf5.",
    type=str,
    default=None,
    metavar="file",
)
parser.add_argument(
    "--loss", help="Loss history.txt.", type=str, default=None, metavar="file"
)
args = parser.parse_args()


def load_hist(file, smooth=True, loss_padd=None):
    hist = np.loadtxt(file)
    is_fine = np.isfinite(hist)
    if not any(is_fine):
        return pd.DataFrame()
    iters = np.where(is_fine)[0]
    hist = hist[is_fine]
    lb = min(hist)
    if loss_padd is not None and lb < 0:
        hist += loss_padd - lb
        lb = loss_padd
    ldf = pd.DataFrame({"loss": hist, "iteration": iters})
    if smooth is not False:
        if lb > 0:
            ldf["sloss"] = np.exp(
                gaussian_filter1d(np.log(hist), sigma=smooth)
            )
        else:
            ldf["sloss"] = gaussian_filter1d(hist, sigma=smooth)
    return ldf


def watch_model(
    name,
    cohorts,
    param_file=None,
    loss_file=None,
    output_dir="data",
    event_type="biochemicalRecurrence",
    event_time="bcrTime",
    model=None,
):
    if model is None:
        md = mc_model(
            cohort="pcapp",
            counter="htseq",
            assembly="tophatV04",
            filterFile="/mnt/fhgfs_ribdata/user_worktmp/dominik.otto/PCa-2016/featureLists/PCaP2q8.txt",
        )
    else:
        md = model

    def now():
        return datetime.today().strftime("%Y-%m-%d-%H:%M:%S")

    output = os.path.join(output_dir, name)
    out_name = "-".join(cohorts) + "_" + event_type + "_" + event_time
    out_tsv = os.path.join(output, f"watch_{out_name}.tsv")
    print(f"[{now()}] Writing to {out_tsv}.")
    try:
        df = pd.read_csv(out_tsv, sep="\t")
    except OSError:
        df = pd.DataFrame(
            columns=[
                "iteration",
                "smooth_loss",
                "cohort",
                "cindex",
                "cox p-value",
                "logrank p-value",
            ]
        )
        df.to_csv(out_tsv, sep="\t", header=True)

    if param_file is None:
        param_file = os.path.join(output_dir, f"{name}_params.hdf5")
    param_bare = os.path.splitext(os.path.basename(param_file))[0]
    if loss_file is None:
        loss_file = os.path.join(output_dir, f"{name}_loss_hist.txt")
    print(f"[{now()}] Waiting for {loss_file}.")
    while not os.path.exists(loss_file):
            time.sleep(5)
    last_edit = None
    log_iteration = 0
    while True:
        log_iteration += 1
        print(f"[{now()}] Starting log iteration {log_iteration}.")
        while True:
            try:
                stats = os.stat(param_file)
            except OSError:
                print(f"[{now()}] Waiting for {param_file}.")
                time.sleep(30)
                continue
            if last_edit != stats.st_mtime:
                break
            else:
                time.sleep(10)
        print(f"[{now()}] Update detected.")
        size = stats.st_size
        while True:
            time.sleep(1)
            stats = os.stat(param_file)
            if size == stats.st_size:
                break
            size = stats.st_size
        last_edit = stats.st_mtime
        print(f"[{now()}] Making statistics.")
        cph, logr = kaplan_meier(
            param_file,
            model=md,
            cohorts=cohorts,
            event_type=event_type,
            event_time=event_time,
            no_plot=True,
        )
        loss = load_hist(loss_file).iloc[-1, :]
        model_iteration = loss["iteration"] + 1
        sloss = loss["sloss"]
        cindex = cph.concordance_index_
        df = df.append(
            {
                "iteration": model_iteration,
                "smooth_loss": sloss,
                "cohort": cohorts,
                "cindex": cindex,
                "cox p-value": cph.summary.loc["score", "p"],
                "logrank p-value": logr.p_value,
            },
            ignore_index=True,
        )
        df.to_csv(out_tsv, sep="\t", header=True)
        out_copy = f"{param_bare}_{model_iteration}iter_{sloss}sloss_{cindex}cindex.hdf5"
        print(f"[{now()}] Copying parameters with concordance {cindex}.")
        copyfile(param_file, os.path.join(output, out_copy))


if __name__ == "__main__":
    print("Watching with args:")
    print(args)
    watch_model(
        args.tag,
        args.cohorts,
        param_file=args.params,
        loss_file=args.loss,
        event_type=args.eventType,
        event_time=args.eventTime,
    )
