# %%
import os, sys

sys.path.append('/root/data/data/scsim-master')
from scsim import scsim

import numpy as np

# %%
## Create output directory structure
if not os.path.exists('/root/data/data/scsim-master/data/Simulations_6/'):
    # os.mkdir('/root/data/data/scsim-master/data/Simulations/')
    os.mkdir('/root/data/data/scsim-master/data/Simulations_6/deloc_0.75')
    # os.mkdir('/root/data/data/scsim-master/data/Simulations/deloc_0.75')
    # os.mkdir('/root/data/data/scsim-master/data/Simulations/deloc_0.00')
    # os.mkdir('/root/data/data/scsim-master/data/Simulations/runtime_evaluation')

# %%
ngenes=20000    
descale=1.0
ndoublets=0
# K=6  
K=20 
# nproggenes = 1000
nproggenes = 7000
proggroups = [1,2,3,4,5,6,7,8,9,10,11,12,13]
progcellfrac = .35
ncells = 60000
# ncells = 50000
deprob = .025
outdirbase = '/root/data/data/scsim-master/data/Simulations_6/deloc_%.2f/Seed_%d'

# %%
def save_df_to_npz(obj, filename):
    np.savez_compressed(filename, data=obj.values, index=obj.index.values, columns=obj.columns.values)

# %%
deloc=0.75
simseeds = [9485,13657]
# simseeds = [9485]

for seed in simseeds:
    outdir = outdirbase % (deloc, seed)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        
    simulator = scsim(ngenes=ngenes, ncells=ncells, ngroups=K, libloc=7.64, libscale=0.78,
                 mean_rate=7.68,mean_shape=0.34, expoutprob=0.00286,
                 expoutloc=6.15, expoutscale=0.49,
                 diffexpprob=deprob, diffexpdownprob=0., diffexploc=deloc, diffexpscale=descale,
                 bcv_dispersion=0.448, bcv_dof=22.087, ndoublets=ndoublets,
                 nproggenes=nproggenes, progdownprob=0., progdeloc=deloc,
                 progdescale=descale, progcellfrac=progcellfrac, proggoups=proggroups,
                 minprogusage=.1, maxprogusage=.7, seed=seed)
    simulator.simulate()

    save_df_to_npz(simulator.cellparams, '%s/cellparams' % outdir)
    save_df_to_npz(simulator.geneparams, '%s/geneparams' % outdir)
    save_df_to_npz(simulator.counts, '%s/counts' % outdir)


