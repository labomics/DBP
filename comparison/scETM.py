# %% [markdown]
# # Batch correction and factorization using scETM

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
import pandas as pd
import re
from scETM import scETM, UnsupervisedTrainer, evaluate, prepare_for_transfer

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='wnn_rna')
parser.add_argument('--experiment', type=str, default='e1')
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--init_model', type=str, default='sp_00001899')
parser.add_argument('--method', type=str, default='scETM')
parser.add_argument('--K', type=int, default='20')
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
s = []
for batch_id in pred.keys():
    s.append(pred[batch_id]["s"]["rna"])
s = np.concatenate(s, axis=0)

# %%
# get counts and masks
counts = {"rna": []}
masks = {"rna": []}
for batch_id in pred.keys():
    for m in counts.keys():
        if m in pred[batch_id]["x"].keys():
            counts[m].append(pred[batch_id]["x"][m])
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

# %% [markdown]
# ## Create AnnData

# %%
if o.task == "wnn_rna":
    labels = []
    for raw_data_dir in o.raw_data_dirs:
        label = utils.load_csv(pj(raw_data_dir, "label", "meta.csv"))
        labels += utils.transpose_list(label)[10][1:]
    labels = np.array(labels)
    ann_data = ad.AnnData(np.concatenate(np.array(counts["rna"]), axis=0))
    ann_data.obs["batch_indices"] = s
    ann_data.obs["cell_types"] = labels
elif o.task == "lung_ts":
    labels = []
    for raw_data_dir in o.raw_data_dirs:
        label = utils.load_csv(pj(raw_data_dir, "label", "meta.csv"))
        labels += utils.transpose_list(label)[13][1:]
    labels = np.array(labels)
    ann_data = ad.AnnData(np.concatenate(np.array(counts["rna"]), axis=0))
    ann_data.obs["batch_indices"] = s
    ann_data.obs["cell_types"] = labels
ann_data

# %%
# ann_data = ad.AnnData(
#     X = np.concatenate(np.array(counts["rna"]), axis=0),
#     obs = meta
# )

# %% [markdown]
# ## Dimensionality reduction using scETM

# %%
obj_model = scETM(ann_data.n_vars, counts["nbatches"], n_topics = K)
trainer = UnsupervisedTrainer(obj_model, ann_data, test_ratio=0.1)
# trainer.train(n_epochs = 12000, eval_every = 1000, batch_col = "Batch", cell_type_col = "Cell type")
trainer.train()

# %% [markdown]
# ## Batch correction using scETM

# %%
obj_model.get_cell_embeddings_and_nll(ann_data, batch_col = "batch_indices")
ann_data

# %% [markdown]
# ## Save embeddings

# %%
embed = ann_data.obsm['theta']
np.savetxt(pj(result_dir, 'embeddings.csv'), embed, delimiter=',')

# %%
ann_data.write(pj(result_dir, 'adata.h5ad'))


