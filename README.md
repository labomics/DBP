# DBP
DBP is a Python library for adaptive factor analysis of scRNA-seq data.

## Usage examples
### Generating training data for DBP
```bash
Rscript preprocess/pbmc.R 
Rscript preprocess/combine_subsets.R --task pbmc  && py preprocess/split_mat.py --task pbmc 
```

### Training DBP
```bash
CUDA_VISIBLE_DEVICES=0 py run.py --exp e0 --task pbmc
```
### Comparison methods
```bash
task=pbmc
init=sp_00001899
K=50
Rscript comparison/liger.r --exp e0 --init_model $init --K $K --task $task
```
### Quantitative evaluation
#### DBP
```bash
task=pbmc
init=sp_00001899
K=38
py eval/benchmark_batch_bio_break.py --task $task --init_model $init --K $K --method DBP
```
#### State-of-the-art methods
```bash
task=pbmc
init=sp_00001899
K=50
py eval/benchmark_batch_bio.py --task $task --init_model $init --K $K --method liger
```

## File description

| File or directory | Description                                                 |
| ------------------- | ------------------------------------------------------------- |
| `simulated datasets/`     | Script for generating simulated data sets |
| `analysis/`     | Scripts for downstream analysis |
| `comparison/`     | Scripts for algorithm comparison and qualitative evaluation |
| `configs/`        | Dataset configuration and DBP model configuration         |
| `eval/`           | Scripts for quantitative evaluation                         |
| `functions/`      | PyTorch functions for DBP                                 |
| `modules/`        | PyTorch models and dataloader for DBP                     |
| `preprocess/`     | Scripts for data preprocessing                              |
| `utils/`          | Commonly used functions                                     |
| `README.md`       | This file                                                   |
| `run.py`          | Script for DBP training and inference                     |

