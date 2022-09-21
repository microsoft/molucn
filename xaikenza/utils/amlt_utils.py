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


for folder in os.listdir('./amlt'):
    if folder.endswith('mcs'):
        folder_path = os.path.join('./amlt', folder)
        if os.path.isdir(folder_path):
            for expe in os.listdir(folder_path):
                print(expe)
                expe_path = os.path.join(folder_path, expe)
                assert os.path.isdir(expe_path)
                for file in os.listdir(expe_path):
                    if file.startswith('model'):
                        row = pd.read_csv(os.path.join(expe_path, file))
                        model_scores = pd.concat([model_scores, row])
                    elif file.startswith('attr'):
                        row = pd.read_csv(os.path.join(expe_path, file))
                        attr_scores = pd.concat([attr_scores, row])
                    
total_model_targets = len(np.unique(model_scores['target']))
total_explained_targets = len(np.unique(attr_scores['target']))

model_scores.to_csv(f'./logs/mcs_model_scores_{total_model_targets}.csv', index=False)
attr_scores.to_csv(f'./results/mcs_attr_scores_{total_explained_targets}.csv', index=False)