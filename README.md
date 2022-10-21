# Introduction 

Supporting code and analyses for [A substructure-aware loss for feature attribution in drug discovery](https://chemrxiv.org/engage/chemrxiv/article-details/633a98bbea6a22542f06e149)

# Getting Started

## Installation process

We recommend the [conda](https://docs.conda.io/en/latest/miniconda.html) Python package manager, and while an GPU is technically not required to run the models and feature attribution methods reported here, it is heavily encouraged. Furthermore, the code has only been tested under Linux. Make a new environment with the provided `environment.yml` file:

```bash
conda env create --file=environment.yaml
conda activate molucn
```

## Prerequisites and data structure

To reproduce the results presented in the paper, download the following compressed data tarball from [here](https://figshare.com/articles/dataset/molucn-data/21215477) (~26GB, when uncompressed):

```bash
wget -O data.tar.gz "https://figshare.com/articles/dataset/molucn-data/21215477?file=37624043"
tar -xf data.tar.gz
```

As the original [benchmark study](https://github.com/josejimenezluna/xaibench_tf), the data provided here is composed of subdirectories organized by PDB identifier, contaning activity data for each target considered in the benchmark. Subfolders have the following structure:


```bash
(molucn):~/molucn/data/1D3G-BRE$ tree
.
├── 1D3G-BRE_heterodata_list.pt
├── 1D3G-BRE_seed_1337_info.txt
├── 1D3G-BRE_seed_1337_stats.csv
├── 1D3G-BRE_seed_1337_test.pt
└── 1D3G-BRE_seed_1337_train.pt
```

An explanation for each file in the subdirectories is provided below:

- `1D3G-BRE_heterodata_list.pt`: dataset with all pairs of ligands saved as `torch_geometric.data.HeteroData` objects with information containing the SMILES, grount-truth colorings, ligand activities, and molecule structures in `torch_geometric.data.Data` objects and the MCS boolean lists at different thresholds. 
- `1D3G-BRE_seed_1337_info.txt`: text file containing information on the congeneric series: number of different ligands/compounds, number of pairs, number of training and testing pairs after 1. splitting the compounds in training and testing sets, 2. keeping pairs with no overlap, 3. rebalancing the training and testing pairs to have a 80%/20% ratio.
- `1D3G-BRE_seed_1337_stats.csv`: summarizes the previous .txt file into a .csv file to facilitate information extraction.
- `1D3G-BRE_seed_1337_test.pt` and `1D3G-BRE_seed_1337_train.pt`: contains the test and train pairs, respectively, saved as `torch_geometric.data.HeteroData` objects obtained after the preprocessing and rebalancing pipelines.


All the `.pt` files can be read with the Python [dill](https://pypi.org/project/dill/) module.


## Build and Test

Given a specific target protein and a feature attribution, the `main.py` file trains a GNN model and generates node colorings using the explainability method selected for all 3 losses proposed in the study: MSE, MSE+AC and MSE+UCN.

The trained GNN models and their logs (metrics) will be saved under `models/` and `logs/` subdirectories in in the root directory of the repo. The atom coloring produced by the feature attribution methods are saved in `colors/`. Metrics measuring the performance of the different feature attribution techniques will be saved under `results/`.

## Example: Test on the data from 1 protein target

To train the GNN model and run a feature attribution for one target protein (e.g., 1D3G-BRE) run:

```bash
python molucn/main.py --target 1D3G-BRE --method [diff, gradinput, ig, cam, gradcam]
```

For the random forest and masking baseline:

```bash
python molucn/main_rf.py --target 1D3G-BRE
```


## Test on all 350 protein targets

To reproduce the results for the 350 protein targets:
- GNN-based methods :

```bash
bash main.sh {diff|gradinput|ig|cam|gradcam}
```


- RF masking:

```
bash main_rf.sh
```

## Citation

If you find this work or parts thereof useful, please consider citing:

```
@article{amara2022substructure,
  title={A substructure-aware loss for feature attribution in drug discovery},
  author={Amara, Kenza and Rodriguez-Perez, Raquel and Luna, Jos{\'e} Jim{\'e}nez},
  year={2022}
}
```
