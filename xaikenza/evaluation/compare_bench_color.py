import json
import os
import os.path as osp

import numpy as np
import pandas as pd

from xaikenza.utils.parser_utils import overall_parser

NO_VALUES_AT = [28, 269, 346, 600, 608, 655]


def clean_score(score, selected_idx):
    score_bench = np.delete(score, NO_VALUES_AT)
    return score_bench[selected_idx]


def create_mapping(path="xaibench/benchmark"):
    mapping = dict()
    k = 0
    for folder in os.listdir(path):
        mapping[folder] = k
        k += 1
    return mapping


def get_selected_indices(mapping, list_targets):
    selected_idx = []
    for target in list_targets:
        selected_idx.append(mapping[target])
    return selected_idx


if __name__ == "__main__":
    
    parser = overall_parser()
    args = parser.parse_args()
    train_params = f"{args.conv}_{args.loss}_{args.pool}_{args.lambda1}"
    # Import benchmark scores
    ori_results = "xaibench/results"

    file = os.path.join(ori_results, "accs.pt")
    with open(file, "rb") as fp:
        xaibench_accs = pd.read_pickle(fp)

    file = os.path.join(ori_results, "f1s.pt")
    with open(file, "rb") as fp:
        xaibench_f1s = pd.read_pickle(fp)


    # Import computed scores
    try:
        with open(
            osp.join(args.result_path, f"results_{train_params}.json"), "r"
        ) as fp:
            res_pred = json.load(fp)
    except FileNotFoundError:
        print("No results generated yet. Please run code/explain.py first.")

    res_pred = pd.DataFrame(res_pred["results"])
    res_pred = res_pred.drop_duplicates(subset=["target"])
    list_targets = list(res_pred["target"])
    
    
    mapping = create_mapping()
    selected_idx = get_selected_indices(mapping, list_targets)

    accs_bench = list(
        clean_score(xaibench_accs["mpnn"]["GradInput"][0], selected_idx)
    )  # First MCS
    f1s_bench = list(
        clean_score(xaibench_f1s["mpnn"]["GradInput"][0], selected_idx)
    )  # First MCS
    
    accs_train_pred, f1s_train_pred = (
        res_pred["mean_acc_train"],
        res_pred["mean_f1_train"],
    )
    accs_test_pred, f1s_test_pred = res_pred["mean_acc_test"], res_pred["mean_f1_test"]
    targets, ids = res_pred['target'], [mapping[i] for i in res_pred['target']]

    # Compare scores
    assert len(accs_train_pred) == len(accs_bench)
    assert len(f1s_train_pred) == len(f1s_bench)
    assert len(accs_test_pred) == len(accs_bench)
    assert len(f1s_test_pred) == len(f1s_bench)

    x = np.array(
        [   targets,
            ids,
            accs_bench,
            f1s_bench,
            accs_train_pred,
            f1s_train_pred,
            accs_test_pred,
            f1s_test_pred,
        ]
    )
    df = pd.DataFrame(
        x.T,
        columns=[
            "target",
            "id",
            "accs_bench",
            "f1s_bench",
            "accs_train",
            "f1s_train",
            "accs_test",
            "f1s_test",
        ],
    )
    df.to_csv(osp.join(args.result_path, f"results_{train_params}.csv"), index=False)

  