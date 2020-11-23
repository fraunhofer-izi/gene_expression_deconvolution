#' ---
#' jupyter:
#'   jupytext:
#'     text_representation:
#'       extension: .R
#'       format_name: spin
#'       format_version: '1.0'
#'       jupytext_version: 0.8.5
#'   kernelspec:
#'     display_name: R
#'     language: R
#'     name: ir
#'   language_info:
#'     codemirror_mode: r
#'     file_extension: .r
#'     mimetype: text/x-r-source
#'     name: R
#'     pygments_lexer: r
#'     version: 3.4.1
#' ---

#SBATCH --job-name="DESeq on DR"
#SBATCH --out=.slurmlogs/slurm.dr_deseq2.%j_%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=7000
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=dominik.otto@izi.fraunhofer.de

# This script uses DESeq2 and fdrtool in dimensional reduced
# data to estimate the share of differntially expressed features.

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
suppressPackageStartupMessages(library(fdrtool))

inPath = "data/deg_counts_prior"
outPath = "data/DR_DESeq2_prior"
dir.create(outPath, showWarnings = FALSE, recursive = TRUE)

pheno = readRDS(paste0(inPath, "/pheno.RDS"))
counts = readRDS(paste0(inPath, "/counts.RDS"))

dg_path = paste0(inPath, "/", ncomps - 1, "-comp.RDS")
if (!file.exists(dg_path)) {
    message(paste('The file', dg_path, 'does not exist. Stopping...'))
    quit()
}
raw_deg_counts = readRDS(dg_path)
deg_counts = data.frame(t(raw_deg_counts))
colnames(deg_counts) = paste0(colnames(counts), "-DR")
rownames(deg_counts) = rownames(counts)

deg_pheno = pheno
rownames(deg_pheno) = paste0(rownames(deg_pheno), "-DR")
pheno$DR = TRUE
deg_pheno$DR = FALSE

a_counts = cbind(counts, deg_counts)
a_pheno = rbind(pheno, deg_pheno)

if (!is.null(cohort)) {
    ind = a_pheno$Cohort == cohort
    a_counts = a_counts[, ind]
    a_pheno = a_pheno[ind, ]
}

broken = apply(is.na(a_counts), 1, any)
if (any(broken)) {
    message(paste0("There are ", sum(broken),
                   " features with NA values that are removed."))
    a_counts = a_counts[!broken, ]
}

dds_dat = DESeqDataSetFromMatrix(countData = round(a_counts),
                                 colData = a_pheno,
                                 design = ~ DR)

dds = DESeq(dds_dat)

res = results(dds)


ind = is.finite(res$pvalue)
pdf(paste0(outPath, "/fdrtool_plot_", cohort, "_", ncomps - 1, "-comp.pdf"))
fdr_result = fdrtool(res$pvalue[ind], statistic = "pvalue", verbose = FALSE)
dev.off()

share_diff_expr = max(fdr_result$qval, na.rm = TRUE)

write(share_diff_expr, paste0(outPath, "/share_diff_expr_",
                              cohort, "_", ncomps - 1, "-comp.txt"))

data = list(dds_data = dds_dat, dds = dds, dds_result = res,
            fdr_result = fdr_result)
saveRDS(data, paste0(outPath, "/DESeq2_results_", cohort, "_",
                     ncomps - 1, "-comp.RDS"))
