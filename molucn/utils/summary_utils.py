# Read gpu results and save them in a single csv file

import pandas as pd
import numpy as np
import os

model_scores = pd.DataFrame(
            columns=[
                "target",
                "seed",
                "conv",
                "pool",
                "loss",
                "lambda1",
                "explainer",
                "rmse_test",
                "pcc_test",
            ],
        )


attr_scores = pd.DataFrame(
            columns=[
            "target",
            "seed",
            "conv",
            "pool",
            "loss",
            "lambda1",
            "explainer",
            "time",
            "acc_train",
            "acc_test",
            "f1_train",
            "f1_test",
            "global_dir_train",
            "global_dir_test",
            "local_dir_train",
            "local_dir_test",
            "mcs",
            "n_mcs_train",
            "n_mcs_test"
        ],
        )

par_path = "/home/kamara/molucn/"
dir_path = "/cluster/work/zhang/kamara/molucn/"

model_path = os.path.join(dir_path, "logs")
attr_path = os.path.join(dir_path, "results")


for expe in os.listdir(model_path):
    if expe[-15:-13] == "vx":
        print(expe)
        expe_path = os.path.join(model_path, expe)
        row = pd.read_csv(expe_path)
        model_scores = pd.concat([model_scores, row])


for expe in os.listdir(attr_path):
   if expe[-15:-13] == "vx":
        print(expe)
        expe_path = os.path.join(attr_path, expe)
        row = pd.read_csv(expe_path)
        attr_scores = pd.concat([attr_scores, row])

all_model_scores = pd.read_csv('./logs/mcs_model_scores_350.csv')
all_attr_scores = pd.read_csv('./results/mcs_attr_scores_350.csv')

model_scores = pd.concat([model_scores, all_model_scores])
attr_scores = pd.concat([attr_scores, all_attr_scores])
                        
total_model_targets = len(np.unique(model_scores['target']))
total_explained_targets = len(np.unique(attr_scores['target']))

model_scores.to_csv(f'./logs/all_mcs_model_scores_{total_model_targets}.csv', index=False)
attr_scores.to_csv(f'./results/all_mcs_attr_scores_{total_explained_targets}.csv', index=False)