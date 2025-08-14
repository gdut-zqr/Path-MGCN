check_and_install <- function(pkgs, bioc=FALSE) {
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      message("Installing missing package: ", pkg)
      if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager", repos = "https://cran.r-project.org")
      if (bioc) {
        BiocManager::install(pkg, ask = FALSE, update = FALSE)
      } else {
        install.packages(pkg, repos = "https://cran.r-project.org")
      }
    }
  }
}

check_and_install(c("optparse", "data.table"), bioc = FALSE)
check_and_install(c("GSVA", "GSEABase", "BiocParallel"), bioc = TRUE)

suppressPackageStartupMessages({
  library(optparse)
  library(GSEABase)
  library(GSVA)
  library(data.table)
  library(BiocParallel)
})

opt_list <- list(
  make_option("--expr",    type = "character", help = "expression TSV"),
  make_option("--gmt",     type = "character", help = "gene-set GMT file"),
  make_option("--out",     type = "character", help = "output TSV"),
  make_option("--threads", type = "integer",   default = 50, help = "CPU cores [default %default]")
)
opt <- parse_args(OptionParser(option_list = opt_list))

if (is.null(opt$expr) || is.null(opt$gmt) || is.null(opt$out)) {
  stop("Must provide --expr, --gmt and --out", call. = FALSE)
}

expr_dt <- fread(opt$expr, data.table = FALSE)
rownames(expr_dt) <- expr_dt[[1]]
expr_mat <- as.matrix(expr_dt[ , -1, drop = FALSE])

gene_sets  <- getGmt(opt$gmt)
gsva_res <- NULL
try_old <- tryCatch({
  message("Trying old GSVA API...")
  gsva(expr = expr_mat,
       gset.idx.list = gene_sets,
       kcdf = "Poisson",
       parallel = opt$threads)
}, error = function(e) {
  message("Old API failed: ", conditionMessage(e))
  NULL
})

if (!is.null(try_old)) {
  gsva_res <- try_old
} else {
  message("Falling back to new GSVA API...")
  gsva_param <- gsvaParam(
    exprData = expr_mat,
    geneSets = gene_sets,
    kcdf     = "Poisson",
  )
  gsva_res <- gsva(gsva_param)
}
out_dt <- as.data.frame(t(gsva_res))
fwrite(out_dt, file = opt$out, sep = "\t", quote = FALSE, row.names = TRUE)
cat("GSVA finished, result written to", opt$out, "\n")
invisible(gc())
closeAllConnections()
Sys.sleep(1)
