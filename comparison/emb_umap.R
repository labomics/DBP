# %% [markdown]
# # Visualizing embeddings obtained from LDVAE and scETM

# %%
source("/root/data/DBP_sa_bc/preprocess/utils.R")
setwd("/root/data/DBP_sa_bc/")
library(RColorBrewer)

parser <- ArgumentParser()
parser$add_argument("--task", type = "character", default = "wnn_rna")
parser$add_argument("--method", type = "character", default = "LDVAE")
parser$add_argument("--exp", type = "character", default = "e1")
parser$add_argument("--model", type = "character", default = "default")
parser$add_argument("--init_model", type = "character", default = "sp_00001899")
parser$add_argument("--K", type = "integer", default = "20")
o <- parser$parse_known_args()[[1]]

config <- parseTOML("configs/data.toml")[[o$task]]
subset_names <- basename(config$raw_data_dirs)
subset_ids <- sapply(seq_along(subset_names) - 1, toString)
input_dirs <- pj("result", o$task, o$exp, o$model, "predict", o$init_model, paste0("subset_", subset_ids))
pp_dir <- pj("data", "processed", o$task)
output_dir <- pj("result", "comparison", o$task, o$method, o$K)
mkdir(output_dir, remove_old = F)
# label_paths <- pj(config$raw_data_dirs, "label_seurat", "l1.csv")
label_paths <- pj(config$raw_data_dirs, "label", "meta.csv")

# K <- parseTOML("configs/model.toml")[["default"]]$dim_c
K <- o$K
l <- 7.5  # figure size
L <- 10   # figure size
m <- 0.5  # legend margin

# %% [markdown]
# ## Load preprossed data

# %%
rna_list <- list()
cell_name_list <- list()
label_list1 <- list()
label_list2 <- list()
subset_name_list <- list()
S <- length(subset_names)
for (i in seq_along(subset_names)) {
    subset_name <- subset_names[i]
    rna_dir  <- pj(input_dirs[i], "x", "rna")
    fnames <- dir(path = rna_dir, pattern = ".csv$")
    fnames <- str_sort(fnames, decreasing = F)

    rna_subset_list <- list()
    N <- length(fnames)
    for (n in seq_along(fnames)) {
        message(paste0("Loading Subset ", i, "/", S, ", File ", n, "/", N))
        rna_subset_list[[n]] <- read.csv(file.path(rna_dir, fnames[n]), header = F)
    }
    rna_list[[subset_name]] <- bind_rows(rna_subset_list)

    cell_name_list[[subset_name]] <- read.csv(pj(pp_dir, paste0("subset_", subset_ids[i]),
        "cell_names.csv"), header = T)[, 2]
        
    if ("lung_ts" %in% o$task){
        label_list1[[subset_name]] <- read.csv(label_paths[i], header = T)[, "Celltypes1"]
        label_list2[[subset_name]] <- read.csv(label_paths[i], header = T)[, "Celltypes_updated_July_2020"]
    }else if("wnn_rna" %in% o$task){
        label_list1[[subset_name]] <- read.csv(label_paths[i], header = T)[, "celltype.l1"]
        label_list2[[subset_name]] <- read.csv(label_paths[i], header = T)[, "celltype.l2"]
    }
    subset_name_list[[subset_name]] <- rep(subset_name, length(cell_name_list[[subset_name]]))
}

# %% [markdown]
# ## Create seurat object

# %%
cell_name <- do.call("c", unname(cell_name_list))

rna <- t(data.matrix(bind_rows(rna_list)))
colnames(rna) <- cell_name
rownames(rna) <- read.csv(pj(pp_dir, "feat", "feat_names_rna.csv"), header = T)[, 2]


# remove missing features
rna_mask_list <- list()
for (i in seq_along(subset_names)) {
    subset_name <- subset_names[i]
    rna_mask_list[[subset_name]] <- read.csv(pj(pp_dir, paste0("subset_", subset_ids[i]),
        "mask", "rna.csv"), header = T)[, -1]
}
rna_mask <- as.logical(apply(data.matrix(bind_rows(rna_mask_list)), 2, prod))
rna <- rna[rna_mask, ]


obj <- CreateSeuratObject(counts = rna, assay = "rna")
# obj@meta.data$celltype <- do.call("c", unname(label_list))

obj@meta.data$celltype1 <- do.call("c", unname(label_list1))
obj@meta.data$celltype2 <- do.call("c", unname(label_list2))
obj@meta.data$batch <- factor(x = do.call("c", unname(subset_name_list)), levels = subset_names)
table(obj@meta.data$batch)[unique(obj@meta.data$batch)]

obj <- subset(obj, subset = nCount_rna > 0)
obj

# %%
c <- read.csv(pj(output_dir, "embeddings.csv"), header = FALSE)
c <- data.matrix(c)
colnames(c) <- paste0("c_", seq_len(ncol(c)))
rownames(c) <- colnames(obj)
obj[["c"]] <- CreateDimReducObject(embeddings = c, key = "c_", assay = "rna")

# obj <- subset(obj, subset = nCount_rna_bc > 0)
obj

# %% [markdown]
# ## Visualization

# %%
obj <- RunUMAP(obj, reduction = 'c', dims = 1:K, reduction.name = "umap")
# SaveH5Seurat(obj, pj(output_dir, "obj.h5seurat"), overwrite = TRUE)

# %%
if ("wnn_rna" %in% o$task){
    batch_cols <- col_8
    celltype1_cols <- col_8
    celltype2_cols <- col_31
}else if("lung_ts" %in% o$task){
    batch_cols <- col_5
    celltype1_cols <- col_16
    celltype2_cols <- col_28
}

dim_plot(obj, w = L, h = L, reduction = "umap", no_axes = T,
    split.by = NULL, group.by = "batch", label = F, repel = T, 
    label.size = 4, pt.size = 0.1, cols = batch_cols, legend = F,
    save_path = pj(output_dir, paste(o$method, "merged_batch", sep = "_")))
     
dim_plot(obj, w = L, h = L, reduction = "umap", no_axes = T,
    split.by = NULL, group.by = "celltype1", label = F, repel = T, 
    label.size = 4, pt.size = 0.1, cols = celltype1_cols, legend = F,
    save_path = pj(output_dir, paste(o$method, "merged_label1", sep = "_")))

dim_plot(obj, w = L, h = L, reduction = "umap", no_axes = T,
    split.by = NULL, group.by = "celltype2", label = F, repel = T, 
    label.size = 4, pt.size = 0.1, cols = celltype2_cols, legend = F,
    save_path = pj(output_dir, paste(o$method, "merged_label2", sep = "_")))

dim_plot(obj, w = L*6, h = L, reduction = "umap", no_axes = T,
    split.by = "batch", group.by = "celltype1", label = F, repel = T, 
    label.size = 4, pt.size = 0.1, cols = celltype1_cols, legend = F,
    save_path = pj(output_dir, paste(o$method, "batch_split1", sep = "_"))) 

dim_plot(obj, w = L*6, h = L, reduction = "umap", no_axes = T,
    split.by = "batch", group.by = "celltype2", label = F, repel = T, 
    label.size = 4, pt.size = 0.1, cols = celltype2_cols, legend = F,
    save_path = pj(output_dir, paste(o$method, "batch_split2", sep = "_"))) 

# %%
# # convert the notebook to html
# system(paste0("jupyter nbconvert --to html comparison/", o$method, ".ipynb"))
# system(paste0("mv comparison/", o$method, ".html comparison/", o$task, "_", o$method, ".html"))


