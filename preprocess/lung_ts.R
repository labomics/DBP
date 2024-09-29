source("/root/data/DBP_sa_bc/preprocess/utils.R")
library(Matrix)

base_dir <- "/root/data/DBP_sa_bc/data/raw/rna/lung_ts"
obj <- readRDS(pj(base_dir, "lung_ts.rds"))
obj
unique(obj@meta.data$Donor)

rna_split <- SplitObject(obj, split.by = "Donor")

for (batch in unique(obj@meta.data$Donor)) {
    prt("Processing batch ", batch, " ...\n")
    output_dir <- pj(base_dir, tolower(batch), "seurat")
    mkdir(output_dir, remove_old = T)

    rna_counts <- rna_split[[batch]]$RNA@counts

    # RNA
    rna <- gen_rna(rna_counts)
    # VlnPlot(rna, c("nFeature_rna", "nCount_rna"), 
    #         pt.size = 0.001, ncol = 2) + NoLegend()
    # ggsave(file="temp.png", width=12, height=6)
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