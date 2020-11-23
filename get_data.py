import numpy as np
import pandas as pd
from rpy2.robjects import r, pandas2ri, globalenv
from itertools import compress
import os.path
import warnings

def get_data(cohort, assembly = 'tophatV04', normalization = 'raw', filter =
             'none', counter = 'htseq', sync = True, metaCounts = False,
             countDir = 'data/in', phenoFile = 'data/in.RData', filterFile = None):
    """
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
    """

    # check types
    stringArgs = [cohort, assembly, normalization, filter, counter]
    isString = [isinstance(arg, str) for arg in stringArgs]
    if (not all(isString)):
        raise TypeError("All arguments but `sync` and `metaCounts` musst be strings.")
    if not isinstance(sync, bool):
        raise TypeError("The argument `sync` musst be boolean value.")
    if not isinstance(metaCounts, bool):
        raise TypeError("The argument `metaCounts` musst be boolean value.")

    # check normalization
    norms = ['raw', 'tpm-normalized', 'vst-normalized', 'rlog-normalized',
             'normTransform-normalized', 'cpm-normalized']
    normInd = [n.startswith(normalization) for n in norms]
    nHits = sum(normInd)
    if (nHits < 1):
        raise ValueError("The argument `normalization` needs to be an abbreviation for one of " +
                         ', '.join(norms))
    elif (nHits > 2):
         raise ValueError("The abbreviation in the argument `normalization` is ambiguous.")
    else:
        normTerm = list(compress(norms, normInd))[0]

    if (normTerm == 'cpm-normalized'):
        normTerm = 'raw'
        doCPM = True
    else:
        doCPM = False

    if (counter == 'kallisto'):
        counter = 'kalGene'
    if (counter == 'htseq'):
        fnParts = [normTerm, filter, cohort, assembly]
    else:
        fnParts = [normTerm, filter, cohort, assembly, counter]

    # look for counts file
    cFileName = '-'.join(fnParts) + '-counts.RData'
    countsFile = os.path.join(countDir, cFileName)
    if (not os.path.isfile(countsFile)):
        raise IOError('There is no counts file for the given parameters: ' + countsFile)

    pandas2ri.activate()

    # look for pheno file
    if (sync and not os.path.isfile(phenoFile)):
        raise IOError('The pheno data file could not e found: ' + phenoFile)
    elif (not os.path.isfile(phenoFile)):
        warnings.warn('The pheno data file could not e found: ' + phenoFile)
        pheno = pd.DataFrame()
    else:
        r.load(phenoFile)
        r('ind <- sapply(pheno, mode) == "logical"')
        r('pheno[ind] = lapply(pheno[ind], as.numeric)') # only float arrays have `NaN`
        r('pheno <- subset(pheno, Cohort != "UKDP1")') # remove duplicate samples
        pheno = r('pheno').set_index('ID')

    # load counts file
    r.load(countsFile)
    r('colnames(Counts) <- make.names(colnames(Counts))')
    if filterFile is not None:
        if not os.path.isfile(filterFile):
            raise IOError(f'The given filter file `{filterFile}` does not exist.')
        r(f'features <- readLines("{filterFile}")')
        r('Counts <- subset(Counts, rownames(Counts) %in% features)')
    counts = pd.DataFrame(r('Counts'), index=r('rownames(Counts)'), columns=r('colnames(Counts)'))

    if (not metaCounts):
        normalFeatures = [not f.startswith('__') for f in counts.index.values]
        counts = counts.loc[normalFeatures, :]

    if (doCPM):
        counts = 1e6 * counts / counts.apply(sum, axis = 0)

    if (sync):
        commonSample = sorted(set(pheno.index).intersection(counts.columns))
        counts = counts[commonSample]
        pheno = pheno.loc[commonSample]

    return counts, pheno

def get_annoTab(path = None, assembly = None):
    """This method never worked because pandas2ri cannot import factors with <NA> as level and value."""
    pandas2ri.activate()
    globalenv['path'] = path
    r('vars = load(path)')
    r('mTab = get(vars[1])')
    annoTab = r('mTab')

    result = annoTab
    if assembly is not None:
        for col in annoTab.columns:
            if assembly.lower() in col.lower():
                result = annoTab.set_index(col)
    return result
