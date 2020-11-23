#!/usr/bin/env Rscript
#SBATCH --job-name="ID DESeq2"
#SBATCH --out=.slurmlogs/slurm.id_deseq2.%j_%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dominik.otto@izi.fraunhofer.de

# This script calculates the differntial experssion between
# tumor and non tumor samples within each cohort and in the
# degernate / dimensional reduced data with ncomponents.

if (!exists("ncomps")){
    args = commandArgs(trailingOnly = TRUE)
    if (length(args) < 1){
        stop(paste0("No numer of components specified.\n",
            "You can specify the number by providing the it behind ",
            "--args when calling the script from the command line ",
            "or setting the 'ncomps' variable in the R enviorment."))
    }else{
        ncomps = as.numeric(args[1])
    }
}
ncomps = ceiling(ncomps)
if (!is.finite(ncomps) || ncomps < 1) {
    stop("The number of components musst be an integer larger than 0.")
}

if (!exists("cohort") && length(args) > 1) {
    cohort = args[2]
} else if (!exists("cohort")) {
    cohort = NULL
}

message(paste("This is ncomps:", ncomps, "; for cohort: ", cohort))

suppressPackageStartupMessages(library(DESeq2))

inPath = "data/deg_counts_od"
outPath = "data/ID_DESeq2_od"
dir.create(outPath, showWarnings = FALSE, recursive = TRUE)

pheno = readRDS(paste0(inPath, "/pheno.RDS"))
counts = readRDS(paste0(inPath, "/counts.RDS"))

dg_path = paste0(inPath, "/", ncomps - 1, "-comp.RDS")
if (!file.exists(dg_path)) {
    message(paste("The file", dg_path, "does not exist. Stopping..."))
    quit()
}
raw_deg_counts = readRDS(dg_path)
deg_counts = data.frame(t(raw_deg_counts))
colnames(deg_counts) = colnames(counts)
rownames(deg_counts) = rownames(counts)

if (!is.null(cohort)) {
    ind = pheno$Cohort == cohort
    deg_counts = deg_counts[, ind]
    pheno = pheno[ind, ]
}

broken = apply(is.na(deg_counts), 1, any)
if (any(broken)) {
    message(paste0("There are ", sum(broken),
                   " features with NA values that are removed."))
    deg_counts = deg_counts[!broken, ]
}

noPheno = is.na(pheno$cep)
if (any(noPheno)) {
    message(paste0("There are ", sum(noPheno),
                   " samples with NA values that are removed."))
    deg_counts = deg_counts[, !noPheno]
    pheno = pheno[!noPheno, ]
}

dds_dat = DESeqDataSetFromMatrix(countData = round(deg_counts),
                                 colData = pheno,
                                 design = ~ cep)

dds = DESeq(dds_dat)

res = results(dds)

data = list(dds_data = dds_dat, dds = dds, dds_result = res)
saveRDS(data, paste0(outPath, "/DESeq2_results_", cohort, "_",
                     ncomps - 1, "-comp.RDS"))
