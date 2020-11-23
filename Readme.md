# Introduction

This is a collection of code that was used for the Ph.D. thesis

    Computational Gene Expression Deconvolution

by Dominik J. Otto. This repo only contains the code used in the example
application "deconvolution of expression patterns". The code that was applied
in the DREAM Tumor Deconvolution challenge is collected in another repo.

# Goal

The goal of this project is to reconstruct tumor-specific gene expression
patterns from RNA-Seq samples of mixed tissue. To achieve this a probability
distribution of expression patterns is learned for tumor and for non-tumor
cells to characterize the respective tissue. This characterization is learned
implicitly during the deconvolution. Hence the deconvolution of expression
patterns and characterization affect one another.

# Method

The deconvolution is achieved by defining conditional and prior probabilities
that link the measured data and pathological tumor content estimate with latent
information. The latent information inferred is the true tumor cell content,
the true transcriptome decomposition of the sample, the tumor-specific
expression patterns, the non-tumor specific expression patterns, and the
described characterizations of tissues.

To achieve this, the PyMC3 implementation of Automatic Differential Variational
Inference (**ADVI**) is used.

# Input

The input data is collected with the `get_data` function defined in
`get_data.py`. New input must be spplied in the format that can be read
by the function or throgh modification of its code. Here is the docstring:
```
get_data(cohort, assembly='tophatV04', normalization='raw', filter='none',
         counter='htseq', sync=True, metaCounts=False, countDir='data/in',
         phenoFile='data/in.RData', filte rFile=None)
    Loads data from RData count files in `countDir`
    and pheno data from 'phenoDir'.

    Example
    -------
    >>> counts, pheno = get_data('ukd1')

    Parameters
    ----------
    cohort String
        Determines the cohort to load.
        (e.g. 'ukd1', 'ukd2', 'ukd4', 'osr1', 'osr2',
                          'ut1', 'prad', 'pcap')
    assembly String
        The assembly used to define the features
        that were counted. (e.g. 'tophatV04' or 'stringtieV05')
    normalization String
        Which normalization of the counts to use.
        (e.g. 'raw', 'cpm', 'tpm', 'vst', 'rlog', 'normTransform')
    filter String
        The filter applied befor normalizing the data.
        (e.g. 'none' or 'q9')
    counter String
        The tool and summarization to count the features.
        e.g. 'kalGene' stands for kallisto counts that where
        summarized to gene counts.
    sync Bool
        A boolean indicating whether counts and pheno data
        should be restricted to common samples and ordered the
        equally.
    metaCounts Bool
        A boolean indicating whether features that
        start with '__' (e.g. '__alignment_not_unique') should
        be included.
    countDir Path
        A director where the expression data is stored in files
        `<normalization>-none-<cohort>-<assembly>-counts.RData`.
    phenoFile Path
        Path to a file that contains the pheno data as
        R dataframe.
    filterFile Path
        A text file containing the names of features
        that should be used.

    Value
    -----
    The function returns the two objects `counts, pheno`.
    counts pandas.dataframe
        A pandas data frame containing the counts
        with samples as columns and features as rows.
    pheno pandas.dataframe
        A pandas data frame containing the phenotypic
        data with samples as rows.
```

The script is specialized to load data as distributed in our file system and
from its native R-dataframes that are converted to python pandas dataframes. If
other file sources have to be used, the script must be modified.  While the
script can load different formats only one is required:

The **gene expression data** has to be presented as raw gene counts as of
HTSeqCount, kallisto or the like.  In the `get_data.py` this is the expression
matrix loaded when `normalization = "raw"` is passed. The file has to be
`RData`, located in the directory passed with `countDir='data/in'`, named
`raw-none-<cohort name>-<assembly name>-counts.RData`, and contain an
R dataframe with columns named after the samples and rows beeing the
genomic features or genes.

The **phenotypic data** has to be reference with `phenoFile`
and to be in a unified format. As curently implemented
it has to be an R dataframe with the columns
 - `ID` the sample id
 - `Cohort` the cohort name
 - `Tumor` a boolean value indicating tumor samples
 - `tumorContent` a float or `NA` giving the pathological estimate

If a survival analysis is intendet the **phenotypic data** must also contain
 - `CohortAbb` an abbreviation of the cohort name as used in the model in
   `event_cohorts`
 - `blacklisted` a boolean indication samples which survival data should be
   ignored
 - `PatientID` the patient idendification
 - `<event type>` as specified in the model in `event_type`
 - `<event time>` as specified in the model in `event_time`

# Run the Application

The main application is the deconvolution described in the dissertation.
We only describe this application here. Scripts for other purposes are only
listed in the [content section](#content).

To run the model execute `python3 mc_model.py <arguments> <name>`.
Available options are listed with `python3 mc_model.py --help`:
```
usage: mc_model.py [-h] [--ncomp integer] [--nModes integer]
                   [--nFeatures integer] [--sampDev] [--noDev] [--kmInit]
                   [--sameKernels] [--svgd int] [--saveModel] [--seqFac float]
                   [--DirichletFactor float] [--relax float] [--maxIter int]
                   [--startFile path] [--cohort cohort] [--counter counter]
                   [--assembly assembly] [--filterFile filterFile]
                   [--nmc integer] [--optimizer optimizer]
                   [--hazard hazard_model]
                   [--hazardCohorts cohort [cohort ...]] [--learnrate float]
                   [--nwin int] [--pf float] [--pt float] [--resScale float]
                   [--useMultinomial]
                   [name]

Infer parameter distributions for a dimensionally reduced model.

positional arguments:
  name                  tag the result with a name

optional arguments:
  -h, --help            show this help message and exit
  --ncomp integer       number of components (use maximum if 0, default=275)
  --nModes integer      number of modes for tumor expression characterisation
                        (default=10).
  --nFeatures integer   only use n highes expressed features (use all if 0,
                        default=0)
  --sampDev             sample deviations
  --noDev               do not add any deviation
  --kmInit              Use k-means clustering to init charachterization
                        gaussion mixtrure means.
  --sameKernels         use the same kernels for tumor and non-tumor
                        characterizer
  --svgd int            use SVGD instead of ADVI with `int` particles
  --saveModel           save model with all the data
  --seqFac float        factor for sequencing depth (default=1)
  --DirichletFactor float
                        Factor for Dirichlet prior of gaussian mixture. Larger
                        values force the mixture to use all kernels with more
                        equal weights (default=1).
  --relax float         factor for standard variance to relax charcterization
                        mean prior (default=1)
  --maxIter int         maximum number of iterations (default=1e6)
  --startFile path      hdf5 file with initial parameters
  --cohort cohort       the sample cohort to use, passed to get_data
                        (default=pcap)
  --counter counter     the tool the cohort experssions were counted with,
                        passed to get_data (default=kalGene)
  --assembly assembly   the assembly the cohort experssions were counted on
                        (default=gencodeV27)
  --filterFile filterFile
                        a text file listing the features to use
  --nmc integer         number of MC samples per advi iteration (default is
                        the PyMC3 default which at the time of writing is 1)
  --optimizer optimizer
                        the stochastic optimizer to use (default=adam)
  --hazard hazard_model
                        Choose a hazard moden (e.g. none, cox,
                        mk=multi_kernel, default=none).
  --hazardCohorts cohort [cohort ...]
                        Cohort appreviations as in the pheno data. Selects for
                        which cohorts survival data should be used, (e.g. ukd1
                        prad, default=UKD1 PRAD OSR1 UKD4 OSR2)
  --learnrate float     learning rate of the stochastic optimizer (default is
                        the PyMC3 default)
  --nwin int            window size for the stochastic optimizer that uses
                        windows (default=50)
  --pf float            probability for pathologist to correctly identify non
                        tumor tissue (default=0.95)
  --pt float            probability for pathologist to correctly identify
                        tumor tissue (default=0.95:)
  --resScale float      scale the resolution of the tumor content estimate
                        (default=1.0)
  --useMultinomial      Use the multinomial dist for data instead of Dirichlet
                        potential. Not recommendet since data is rounded to
                        integer values.
```

# Output

The output is written into a directory `data`. The files are
 - `<model name>_params.hdf5` the resulting latent parameters as HDF5 file.
 - `<model name>_loss_hist.txt` the history of the ELBO during ADVI as text
   file.
 - `<model name>_pca.pkl` data for the dimensional reduction as python pickle
   file.
 - `mc_model_<model name>.pkl` (optional if `--saveModel` is passed)
   the PyMC3 model with the initial parameters as python pickle file.

# Content

A short description per file.
 - `1-d-moments.ipynb` notebook investigating density approximation with the
    characteristic function in one dimension
 - `cflDensity.py` utilities for density estimations with the characteristic
   function
 - `DR_DESeq2.R` R script to investigate systematic differential expression
   after dimensional reduction
 - `DR_inv_digamma.py` script to dimensional reduce and reconstruc count data
   to investigate information loss due to dimensional reduction
 - `DR_plots.py` correlation plot of gene counts before and after dimensional
   reduction and reconstruction
 - `gaus_sampler.py` class to sample from gaussian mixture
 - `get_data.py` script to load gene count and pheno data for the deconvolution
    model
 - `gmmot.py` calculate minimal transport between gaussian mixtures
   (by Julie Delon)
 - `ID_DESeq2.R` script to investigate internal distortion after
   dimensional reduction
 - `logDect_cpu.py` implementation of the probability of mixtures in
   the dimensionally reduced space as Theano operation for CPU
 - `logDect_gpu.py` implementation of the probability of mixtures in
   the dimensionally reduced space as Theano operation for GPU
 - `mc_model.py` the main deconvolution model using gaussian mixtures for the
   tumor and non-tumor characteristics
 - `models.py` make all models available in the namespace
 - `m-d-moments.ipynb` notebook investigating density approximations with the
   characteristic function in two or more dimensions
 - `model_inspection_tools.py` collection of functions to plot and ispect
    a deconvolution model from `mc_model`
 - `mv_model.py` a deconvolution model using a single multivariate normal
   distribution for the tumor and non-tumor characteristics
 - `PyMC3_tcStat.ipynb` notebook to fit a beta distribution to tumor content
   estimate distribution
 - `record_cindex.py` script to log and observe model during inference
 - `ribmodel.py` base class for the deconvolution models
 - `rkhs_gaussianity.py` script to investigate gaussianity with reproducing
   kernel Hilbert spaces
 - `tc_model.py` a Bayesian model for the true tumor content based on the
   pathological estimate
 - `tutil.py` utilities to work with Theano on multiple nodes sharing the same
   environment and file system

# License

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

# Citation

[![DOI](https://zenodo.org/badge/315341535.svg)](https://zenodo.org/badge/latestdoi/315341535)

BibTex:
```
@software{dominik_j_otto_2020_4287079,
  author       = {Dominik J. Otto},
  title        = {fraunhofer-izi/gene\_expression\_deconvolution: v1.0},
  month        = nov,
  year         = 2020,
  publisher    = {Zenodo},
  version      = {v1.0},
  doi          = {10.5281/zenodo.4287079},
  url          = {https://doi.org/10.5281/zenodo.4287079} Titel anhand dieser DOI in Citavi-Projekt übernehmen
}
```
