source("/root/data/DBP_sa_bc/preprocess/utils.R")

base_dir <- "/root/data/DBP_sa_bc/data/raw/rna/stimulate/sim1"
cell_path <- pj(base_dir, "cells.csv")
count_path <- pj(base_dir, "counts.csv")
gene_path <- pj(base_dir, "genes.csv")


# load data
counts <- read.csv(file = count_path, row.names = 1)
cells <- read.csv(file = cell_path, row.names = 1)
obj <- gen_rna(counts)
obj$celltypes <- cells["Group"]
obj$batchs <- cells["Batch"]
rna_split <- SplitObject(obj, split.by = "batchs")

for (batch in unique(obj@meta.data$batchs)) {
    prt("Processing batch ", batch, " ...\n")
    output_dir <- pj(base_dir, tolower(batch), "seurat")
    mkdir(output_dir, remove_old = T)

    rna_counts <- rna_split[[batch]]$rna@counts

    # RNA
    rna <- gen_rna(rna_counts)
    VlnPlot(rna, c("nFeature_rna", "nCount_rna"), 
            pt.size = 0.001, ncol = 2) + NoLegend()
    ggsave(file="temp.png", width=12, height=6)
    rna
    # rna <- subset(rna, subset =
    #     nFeature_rna > 4800 & nFeature_rna < 6400 &
    #     nCount_rna >40000 & nCount_rna < 80000
    #     )
    
    # Get intersected cells satisfying QC metrics of all modalities
    cell_ids <- Reduce(intersect, list(colnames(rna)))
    rna <- subset(rna, cells = cell_ids)
    rna
   
    # preprocess and save data
    preprocess(output_dir, rna = rna)
}