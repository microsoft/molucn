# Introduction 

This is the official code for [A substructure-aware loss for feature attribution in drug discovery](). 
Compound property predictions is an important task for ML in drug discovery. The past few years, GNN models have been used to predict the activity of ligands towards protein targets. But there is a need for explainability: we want to know what atoms in the molecule are responsible for the compound property predictions.
We usually use feature attribution methods that color atoms, namely assign positive or negative weights to important atoms to account for their contribution in the compound property prediction. However, recent GNN explainability methods have proven less performant than the simple Random Forest masking strategy which estimates atom contribution as the difference in prediction after the bits of the atom in the molecule fingerprint are removed.       
We propose to improve the gradient-based GNN feature attribution methods by modifying the training optimization objective of GNNs to specifically account for common substructures of pairs of related compounds.



# Getting Started

1.	Installation process

We recommend the [conda](https://docs.conda.io/en/latest/miniconda.html) Python package manager, and while an GPU is technically not required to run the models and feature attribution methods reported here, it is heavily encouraged. Furthermore, the code has only been tested under Linux. Make a new environment with the provided environment.yml file:

`conda env create --name xaienv --file=environment.yaml`

`conda activate xaicode`

2. Load the data

Download the data with 350 protein targets split into training and testing pairs from [here]("https://figshare.com/download) (~26GB, when uncompressed):

`wget -O data.tar.gz "https://figshare.com/download?path=%2F&files=data.tar.gz"`

`tar -xf data.tar.gz`

The data is composed of subfolders, each contaning one congeneric series (for a target protein) considered in the benchmark. Subfolders have the following structure:


```
(xaienv):~/molucn/data/1D3G-BRE$ tree
.
├── 1D3G-BRE_heterodata_list.pt
├── 1D3G-BRE_seed_1337_info.txt
├── 1D3G-BRE_seed_1337_stats.csv
├── 1D3G-BRE_seed_1337_test.pt
└── 1D3G-BRE_seed_1337_train.pt
```

An explanation of each file is provided below:

- `1D3G-BRE_heterodata_list.pt`: dataset with all pairs of ligands saved as `torch_geometric.data.HeteroData` objects with information containing the smiles, the true colorings, the ligands activities, the molecule structures in a `torch_geometric.data.Data` objects and the mcs boolean list indicating if the pair common substructure represents at least 50, 55, 60, 65, 70, 75, 80, 85, 90, 95% of the atoms. 
- `1D3G-BRE_seed_1337_info.txt`: text file containing information on the congeneric series: number of different ligands/compounds, number of pairs, number of training and testing pairs after 1. splitting the compounds in training and testing sets, 2. keeping pairs with no overlap, 3. rebalancing the training and testing pairs to have a 80/20 ratio.
- `1D3G-BRE_seed_1337_stats.csv`: summarizes the previous .txt file into a .csv file to facilitate information extraction.
- `1D3G-BRE_seed_1337_test.pt`: contains the testing pairs saved as `torch_geometric.data.HeteroData` objects obtained after 1. test/train compounds split, 2. pairs that conatin only testing compounds, 3. rebalancing to get a 80/20 ratio training/testing pairs.
- `1D3G-BRE_seed_1337_train.pt`: contains the training pairs saved as `torch_geometric.data.HeteroData` objects obtained after 1. test/train compounds split, 2. pairs that conatin only training compounds, 3. rebalancing to get a 80/20 ratio training/testing pairs.

All the .pt files can be read with the Python [pickle](https://docs.python.org/3/library/pickle.html) module or its extension the Python [dill](https://pypi.org/project/dill/) module.

# Code architecture

All results reported in the manuscript can be reproduced with the accompanying code:


```
(xaienv):~/molucn/xaicode$ tree
.
├── __init__.py
├── dataset
│   ├── __init__.py
│   ├── featurization.py
│   ├── pair.py
│   └── prepare_data.py
├── evaluation
│   ├── __init__.py
│   ├── compare_bench_color.py
│   ├── explain.py
│   ├── explain_color.py
│   ├── explain_direction_global.py
│   └── explain_direction_local.py
├── feat_attribution
│   ├── __init__.py
│   ├── cam.py
│   ├── diff.py
│   ├── explainer_base.py
│   ├── gradcam.py
│   ├── gradinput.py
│   ├── ig.py
├── gnn
│   ├── __init__.py
│   ├── aggregation.py
│   ├── loss.py
│   ├── model.py
│   └── train.py
├── main.py
├── main_rf.py
└── utils
    ├── __init__.py
    ├── parser_utils.py
    ├── path.py
    ├── rf_utils.py
    ├── train_utils.py
    └── utils.py
```

# Build and Test

Given a specific target protein and a feature attribution, the `main.py` file trains a GNN model and generate node coloring using the explainability method selected for all 3 losses: MSE, MSE+AC and MSE+UCN.

The trained GNN models are saved in ./models. The logs (rmse and pcc scores on testing pairs) are saved in ./logs.
The atom coloring returned by the feature attribution method are saved in ./colors. The finale scores measuring the performance of feature attribution are in ./results.

## Test 1 protein target: 1D3G-BRE

To train the GNN model and run feature attribution for one target protein 1D3G-BRE, run:

`python xaicode/main.py --target 1D3G-BRE --method [diff, gradinput, ig, cam, gradcam]`

To run Random Forest and use it to assign feature importance, run:

`python xaicode/main_rf.py --target 1D3G-BRE`

## Test on 350 protein targets

To reproduce the results for the 350 protein targets:
- Gradient-based methods and GNN masking (diff):

`bash main.sh [diff, gradinput, ig, cam, gradcam]`

- RF masking (Sheridan baseline):

`bash main_rf.sh`
