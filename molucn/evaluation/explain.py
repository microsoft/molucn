# Read colors and generate the scores
import os
import os.path as osp
import torch
import dill
import numpy as np
import pandas as pd
from molucn.evaluation.explain_color import get_scores
from molucn.evaluation.explain_direction_global import get_global_directions
from molucn.evaluation.explain_direction_local import get_local_directions
from molucn.utils.parser_utils import overall_parser
from molucn.utils.utils import get_mcs, set_seed


def get_colors(pairs_list, explainer):
    colors = []
    for hetero_data in pairs_list:
        data_i, data_j = hetero_data["data_i"], hetero_data["data_j"]
        data_i.batch, data_j.batch = torch.zeros(
            data_i.x.size(0), dtype=torch.int64
        ), torch.zeros(data_j.x.size(0), dtype=torch.int64)
        color_pred_i, color_pred_j = (
            explainer.explain_graph(data_i),
            explainer.explain_graph(data_j),
        )
        colors.append([color_pred_i, color_pred_j])
    return colors


def get_scores_from_colors(args):

    set_seed(args.seed)
    train_params = (
        f"{args.conv}_{args.loss}_{args.pool}_{args.lambda1}_{args.explainer}"
    )
    # Check that data exists
    file_train = osp.join(
        args.data_path, f"{args.target}/{args.target}_seed_{args.seed}_train.pt"
    )
    file_test = osp.join(
        args.data_path, f"{args.target}/{args.target}_seed_{args.seed}_test.pt"
    )

    if not osp.exists(file_train) or not osp.exists(file_test):
        raise FileNotFoundError(
            "Data not found. Please try to - choose another protein target or - run code/pair.py with a new seed."
        )

    with open(file_train, "rb") as handle:
        train_dataset = dill.load(handle)

    with open(file_test, "rb") as handle:
        test_dataset = dill.load(handle)

    os.makedirs(osp.join(args.color_path, args.explainer, args.target), exist_ok=True)
    with open(
        osp.join(
            args.color_path,
            args.explainer,
            args.target,
            f"{args.target}_seed_{args.seed}_{train_params}_train.pt",
        ),
        "rb",
    ) as handle:
        train_colors = dill.load(handle)
    with open(
        osp.join(
            args.color_path,
            args.explainer,
            args.target,
            f"{args.target}_seed_{args.seed}_{train_params}_test.pt",
        ),
        "rb",
    ) as handle:
        test_colors = dill.load(handle)

    accs_train, f1s_train = get_scores(train_dataset, train_colors, set="train")
    accs_test, f1s_test = get_scores(test_dataset, test_colors, set="test")

    global_dir_train = get_global_directions(train_dataset, train_colors, set="train")
    global_dir_test = get_global_directions(test_dataset, test_colors, set="test")

    local_dir_train = get_local_directions(train_dataset, train_colors, set="train")
    local_dir_test = get_local_directions(test_dataset, test_colors, set="test")

    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(osp.join(args.result_path, args.target), exist_ok=True)
    global_res_path = osp.join(
        args.result_path,
        args.target,
        f"mcs_attr_scores_{train_params}_{args.target}.csv",
    )
    print("Saving scores at...", global_res_path)
    # with warnings.catch_warnings():
    # warnings.simplefilter("ignore", category=RuntimeWarning)

    res_dict = {
        "target": [args.target] * 10,
        "seed": [args.seed] * 10,
        "explainer": [args.explainer] * 10,
        "time": [np.nan] * 10,
        "acc_train": np.nanmean(accs_train, axis=0).tolist(),
        "acc_test": np.nanmean(accs_test, axis=0).tolist(),
        "f1_train": np.nanmean(f1s_train, axis=0).tolist(),
        "f1_test": np.nanmean(f1s_test, axis=0).tolist(),
        "global_dir_train": np.nanmean(global_dir_train, axis=0).tolist(),
        "global_dir_test": np.nanmean(global_dir_test, axis=0).tolist(),
        "local_dir_train": np.nanmean(local_dir_train, axis=0).tolist(),
        "local_dir_test": np.nanmean(local_dir_test, axis=0).tolist(),
        "mcs": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        "n_mcs_train": np.sum(get_mcs(train_dataset), axis=0).tolist(),
        "n_mcs_test": np.sum(get_mcs(test_dataset), axis=0).tolist(),
    }

    df = pd.DataFrame({key: pd.Series(value) for key, value in res_dict.items()})
    df.to_csv(global_res_path, index=False)
    print("scores: ", df)

    attr_path = osp.join(args.result_path, "attr_scores_350.csv")
    if os.path.isfile(attr_path):
        df_all = pd.read_csv(attr_path)
        df_all = pd.concat([df_all, df], axis=0)
    else:
        df_all = df

    df_all.to_csv(attr_path, index=False)


if __name__ == "__main__":
    parser = overall_parser()
    args = parser.parse_args()

    for explainer in os.listdir("./colors"):
        explainer_path = os.path.join("./colors", explainer)
        args.explainer = explainer
        for target in os.listdir(explainer_path):
            expe_path = os.path.join(explainer_path, target)
            args.target = target
            for file in os.listdir(expe_path):
                if file.endswith("train.pt"):
                    args.conv, args.loss, args.pool, args.lambda1 = tuple(
                        [eval(i) for i in file.split("_")[-6:-2]]
                    )
                    get_scores_from_colors(args)
