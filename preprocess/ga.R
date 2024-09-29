source("/root/data/DBP_sa_bc/preprocess/utils.R")

base_dir <- "/root/data/DBP_sa_bc/data/raw/rna/ga_normal"
obj <- LoadH5Seurat(pj(base_dir, "obj_normal.h5seurat"))
# obj <- subset(x = obj, celltype == "Epithial.Cell")
obj
unique(obj@meta.data$batch)


output_dir <- pj(base_dir, "seurat")
mkdir(output_dir, remove_old = T)
rna_counts <- obj$rna@counts

# RNA
rna <- gen_rna(rna_counts)
VlnPlot(rna, c("nFeature_rna", "nCount_rna"), 
            pt.size = 0.001, ncol = 2) + NoLegend()
ggsave(file="temp.png", width=12, height=6)
rna
# rna <- subset(rna, subset =
#     nFeature_rna > 1780 & nFeature_rna < 1820 &
#     nCount_rna >300000 & nCount_rna < 390000
#     )
    
# Get intersected cells satisfying QC metrics of all modalities
cell_ids <- Reduce(intersect, list(colnames(rna)))
rna <- subset(rna, cells = cell_ids)
rna
   
# preprocess and save data
preprocess(output_dir, rna = rna)
