import argparse
import multiprocessing
import os
import os.path as osp
import time
import warnings

import dill
import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles
from sklearn.ensemble import RandomForestRegressor

from xaikenza.evaluation.explain_color import get_scores
from xaikenza.evaluation.explain_direction_global import get_global_directions
from xaikenza.evaluation.explain_direction_local import get_local_directions
from xaikenza.feat_attribution.gradinput import GradInput
from xaikenza.utils.path import COLOR_DIR, DATA_DIR, LOG_DIR, MODEL_DIR, RESULT_DIR
from xaikenza.utils.rf_utils import diff_mask, featurize_ecfp4
from xaikenza.utils.train_utils import DEVICE
from xaikenza.utils.utils import get_mcs, set_seed

os.environ["WANDB_SILENT"] = "true"
N_TREES = 1000
N_JOBS = int(os.getenv("LSB_DJOB_NUMPROC", multiprocessing.cpu_count()))
print("Number of Jobs: ", N_JOBS)

rmse = lambda x, y: np.sqrt(np.mean((x - y) ** 2))


def overall_parser():
    parser = argparse.ArgumentParser(description="Train GNN Model")

    parser.add_argument("--dest", type=str, default="/home/t-kenzaamara/internship2022")
    parser.add_argument(
        "--wandb",
        type=str,
        default="False",
        help="if set to True, the training curves are shown on wandb",
    )
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--seed", type=int, default=1337, help="seed")

    # Saving paths
    parser.add_argument(
        "--data_path", nargs="?", default=DATA_DIR, help="Input data path."
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        default=MODEL_DIR,
        help="path for saving trained model.",
    )
    parser.add_argument(
        "--log_path",
        nargs="?",
        default=LOG_DIR,
        help="path for saving gnn scores (rmse, pcc).",
    )
    parser.add_argument(
        "-color_path",
        nargs="?",
        default=COLOR_DIR,
        help="path for saving node colors.",
    )

    parser.add_argument(
        "--result_path",
        nargs="?",
        default=RESULT_DIR,
        help="path for saving the feature attribution scores (accs, f1s).",
    )

    # Choose protein target
    parser.add_argument(
        "--target", type=str, default="1D3G-BRE", help="Protein target."
    )

    # Train test val split
    parser.add_argument(
        "--test_set_size", type=float, default=0.2, help="test set size (ratio)"
    )
    parser.add_argument(
        "--val_set_size", type=float, default=0.1, help="validation set size (ratio)"
    )

    # Feature attribution method
    parser.add_argument(
        "--explainer", type=str, default="rf", help="Feature attribution method"
    )  # gradinput, ig

    return parser


def main_rf(args):
    set_seed(args.seed)
    train_params = f"None_None_None_None_{args.explainer}"
    # wandb.init(project=f'{train_params}_training', entity='k-amara', name=args.target)

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

    # Make a dataset that inherits from torch_geometric.data.Dataset

    # Ligands in testing set are NOT in training set!

    df_train, df_test = pd.DataFrame(columns=["smiles", "y"]), pd.DataFrame(
        columns=["smiles", "y"]
    )

    for hetero_data in train_dataset:
        df_train.loc[len(df_train.index)] = [
            hetero_data["data_i"].smiles,
            hetero_data["data_i"].y,
        ]
        df_train.loc[len(df_train.index)] = [
            hetero_data["data_j"].smiles,
            hetero_data["data_j"].y,
        ]
    for hetero_data in test_dataset:
        df_test.loc[len(df_test.index)] = [
            hetero_data["data_i"].smiles,
            hetero_data["data_i"].y,
        ]
        df_test.loc[len(df_test.index)] = [
            hetero_data["data_j"].smiles,
            hetero_data["data_j"].y,
        ]

    df_train = df_train.drop_duplicates(subset=["smiles"])
    df_test = df_test.drop_duplicates(subset=["smiles"])

    fps_train = np.vstack(
        [featurize_ecfp4(MolFromSmiles(sm)) for sm in df_train.smiles]
    )
    fps_test = np.vstack([featurize_ecfp4(MolFromSmiles(sm)) for sm in df_test.smiles])

    y_train = np.array([y.item() for y in df_train.y])
    y_test = np.array([y.item() for y in df_test.y])

    model_rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=N_JOBS)
    model_rf.fit(fps_train, y_train)

    # yhat_train = rf.predict(fps_train)
    yhat_test = model_rf.predict(fps_test)

    # rmse_train = rmse(df_train.y, yhat_train)
    # pcc_train = np.corrcoef((df_train.y, yhat_train))[0, 1]

    rmse_test = rmse(y_test, yhat_test)
    pcc_test = np.corrcoef((y_test, yhat_test))[0, 1]

    print("Final test rmse: {:.4f}".format(rmse_test))
    print("Final test pcc: {:.4f}".format(pcc_test))

    # Save GNN scores
    os.makedirs(args.log_path, exist_ok=True)
    global_res_path = osp.join(
        args.log_path, f"model_scores_rf_{train_params}_{args.target}.csv"
    )
    df = pd.DataFrame(
        [
            [
                args.target,
                args.seed,
                args.explainer,
                rmse_test,
                pcc_test,
            ]
        ],
        columns=[
            "target",
            "seed",
            "explainer",
            "rmse_test",
            "pcc_test",
        ],
    )
    df.to_csv(global_res_path, index=False)

    """###### Save trained GNN ######
    save_path = f"{args.target}_{train_params}.pt"
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(osp.join(args.model_path, args.target), exist_ok=True)
    torch.save(
        model.state_dict(), osp.join(args.model_path, args.target, save_path)
    )
    print("Model saved!\n")
    """
    ##### Feature Attribution ####

    # explain model
    t0 = time.time()

    def get_colors_rf(pairs_list, model, diff_fun):
        colors = []
        for hetero_data in pairs_list:
            data_i, data_j = hetero_data["data_i"], hetero_data["data_j"]
            color_pred_i, color_pred_j = (
                diff_fun(data_i.smiles, model.predict),
                diff_fun(data_j.smiles, model.predict),
            )
            colors.append([color_pred_i, color_pred_j])
        return colors

    t0 = time.time()
    train_colors = get_colors_rf(train_dataset, model=model_rf, diff_fun=diff_mask)
    time_explainer = (time.time() - t0) / len(train_dataset)
    print("Average time to generate 1 explanation: ", time_explainer)
    test_colors = get_colors_rf(test_dataset, model=model_rf, diff_fun=diff_mask)

    # Save colors
    os.makedirs(osp.join(args.color_path, args.explainer, args.target), exist_ok=True)
    with open(
        osp.join(
            args.color_path,
            args.explainer,
            args.target,
            f"{args.target}_seed_{args.seed}_{train_params}_train.pt",
        ),
        "wb",
    ) as handle:
        dill.dump(train_colors, handle)
    with open(
        osp.join(
            args.color_path,
            args.explainer,
            args.target,
            f"{args.target}_seed_{args.seed}_{train_params}_test.pt",
        ),
        "wb",
    ) as handle:
        dill.dump(test_colors, handle)

    accs_train, f1s_train = get_scores(train_dataset, train_colors, set="train")
    accs_test, f1s_test = get_scores(test_dataset, test_colors, set="test")

    global_dir_train = get_global_directions(train_dataset, train_colors, set="train")
    global_dir_test = get_global_directions(test_dataset, test_colors, set="test")

    local_dir_train = get_local_directions(train_dataset, train_colors, set="train")
    local_dir_test = get_local_directions(test_dataset, test_colors, set="test")

    os.makedirs(args.result_path, exist_ok=True)
    global_res_path = osp.join(
        args.result_path, f"attr_scores_{train_params}_{args.target}.csv"
    )
    print("Saving scores at...", global_res_path)
    # with warnings.catch_warnings():
    # warnings.simplefilter("ignore", category=RuntimeWarning)

    res_dict = {
        "target": [args.target] * 10,
        "seed": [args.seed] * 10,
        "explainer": [args.explainer] * 10,
        "time": [round(time_explainer, 4)] * 10,
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


if __name__ == "__main__":

    parser = overall_parser()
    args = parser.parse_args()
    main_rf(args)
