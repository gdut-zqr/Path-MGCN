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
gsva_res <- gsva(
  expr = expr_mat,
  gset.idx.list = gene_sets,
  kcdf = 'Poisson',
  parallel = opt$threads,
) 

out_dt <- as.data.frame(t(gsva_res))
fwrite(out_dt, file = opt$out, sep = "\t", quote = FALSE, row.names = TRUE)

cat("GSVA finished, result written to", opt$out, "\n")
