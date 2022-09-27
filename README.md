# Introduction 

This is the official code for [A substructure-aware loss for feature attribution in drug discovery](). 

# Getting Started

1.	Installation process

`conda env create --name xaienv --file=environment.yaml`
`conda activate xaicode`

2. Load the data

`wget -O data.tar.gz "https://figshare.com/download?path=%2F&files=data.tar.gz"`
`tar -xf data.tar.gz`


```
(xaienv):~/molucn/data/1D3G-BRE$ tree
.
├── 1D3G-BRE_heterodata_list.pt
├── 1D3G-BRE_seed_1337_info.txt
├── 1D3G-BRE_seed_1337_stats.csv
├── 1D3G-BRE_seed_1337_test.pt
└── 1D3G-BRE_seed_1337_train.pt
```

# Code architecture

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


# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 

If you want to learn more about creating good readme files then refer the following [guidelines](https://docs.microsoft.com/en-us/azure/devops/repos/git/create-a-readme?view=azure-devops). You can also seek inspiration from the below readme files:
- [ASP.NET Core](https://github.com/aspnet/Home)
- [Visual Studio Code](https://github.com/Microsoft/vscode)
- [Chakra Core](https://github.com/Microsoft/ChakraCore)