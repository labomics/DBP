# %% [markdown]
# # Factorization using LDVAE

# %%
import os
os.chdir("/root/data/DBP_sa_bc/")
from os.path import join as pj
import argparse
import sys
sys.path.append("modules")
import utils
import numpy as np
import anndata as ad
import re
# import scanpy as sc
import os
import pandas as pd
import matplotlib.pyplot as plt

import scvi


# %%
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='wnn_rna')
parser.add_argument('--experiment', type=str, default='e1')
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--init_model', type=str, default='sp_00001899')
parser.add_argument('--method', type=str, default='LDVAE')
parser.add_argument('--K', type=int, default='50')
o, _ = parser.parse_known_args()  # for python interactive
# o = parser.parse_args()

K = o.K

# %%
result_dir = pj("result", "comparison", o.task, o.method, str(o.K))
data_dir = pj("data", "processed", o.task)
cfg_task = re.sub("_atlas|_generalize|_transfer|_ref_.*", "", o.task) # dogma_full
data_config = utils.load_toml("configs/data.toml")[cfg_task]
for k, v in data_config.items():
    vars(o)[k] = v
model_config = utils.load_toml("configs/model.toml")["default"]
if o.model != "default":
    model_config.update(utils.load_toml("configs/model.toml")[o.model])
for k, v in model_config.items():
    vars(o)[k] = v
o.s_joint, o.combs, *_ = utils.gen_all_batch_ids(o.s_joint, o.combs)
utils.mkdirs(result_dir, remove_old=False)

# %% [markdown]
# ## Load preprossed data

# %%
o.mods = ["rna"]
o.pred_dir = pj("result", o.task, o.experiment, o.model, "predict", o.init_model)
pred = utils.load_predicted(o, joint_latent=False, input=True, group_by = "subset")

# %%
# get counts and masks
counts = {"rna": []}
masks = {"rna": []}
s = {"rna": []}
for batch_id in pred.keys():
    for m in counts.keys():
        if m in pred[batch_id]["x"].keys():
            counts[m].append(pred[batch_id]["x"][m])
            s[m].append(pred[batch_id]["s"][m])
            mask_dir = pj(data_dir, "subset_"+str(batch_id), "mask")
            mask = np.array(utils.load_csv(pj(mask_dir, m+".csv"))[1][1:]).astype(bool)
            masks[m].append(mask)
        else:
            counts[m].append(None)

counts["nbatches"] = len(pred)

# %%
# feature intersection
for m in masks.keys():
    mask = np.array(masks[m]).prod(axis=0).astype(bool)
    for i, count in enumerate(counts[m]):
        if count is not None:
            counts[m][i] = count[:, mask]

# %%
if o.task == "wnn_rna":
    labels = []
    for raw_data_dir in o.raw_data_dirs:
        label = utils.load_csv(pj(raw_data_dir, "label", "meta.csv"))
        labels += utils.transpose_list(label)[10][1:]
    labels = np.array(labels)
    print(np.unique(labels))
elif o.task == "lung_ts":
    labels = []
    for raw_data_dir in o.raw_data_dirs:
        label = utils.load_csv(pj(raw_data_dir, "label", "meta.csv"))
        labels += utils.transpose_list(label)[13][1:]
    labels = np.array(labels)
    print(np.unique(labels))

# %% [markdown]
# ## Create AnnData

# %%
# ann_data = ad.AnnData(np.concatenate(np.array(counts["rna"]), axis=0))
ann_data = ad.AnnData(np.concatenate(counts["rna"]))
ann_data.obs["batch"] = np.concatenate(s["rna"]).astype(str)
ann_data.obs["batch"] = ann_data.obs["batch"]
ann_data.obs["cell_types"] = labels
ann_data.layers["counts"] = ann_data.X.copy()
ann_data.raw = ann_data
ann_data

# %% [markdown]
# ## Dimensionality reduction using LDVAE

# %%
scvi.model.LinearSCVI.setup_anndata(ann_data, batch_key="batch")
model = scvi.model.LinearSCVI(ann_data, n_latent=K)
# model.train(max_epochs=250, plan_kwargs={'lr':5e-3}, check_val_every_n_epoch=10, use_gpu=0)
model.train(use_gpu=0)

# %% [markdown]
# ## Save embedings

# %%

latent = model.get_latent_representation()
np.savetxt(pj(result_dir, 'embeddings.csv'), latent, delimiter=',')

# %%
# save results
ann_data.write(pj(result_dir, 'adata.h5ad'))

# %%
# # convert the notebook to html
# system(paste0("jupyter nbconvert --to html comparison/", o$method, ".ipynb"))
# system(paste0("mv comparison/", o$method, ".html comparison/", o$task, "_", o$method, ".html"))


