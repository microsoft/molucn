import os
import os.path as osp
import time

import dill
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader

from molucn.dataset.pair import get_num_features
from molucn.evaluation.explain_color import get_scores
from molucn.evaluation.explain_direction_global import get_global_directions
from molucn.evaluation.explain_direction_local import get_local_directions
from molucn.feat_attribution.cam import CAM
from molucn.feat_attribution.diff import Diff
from molucn.feat_attribution.gradcam import GradCAM
from molucn.feat_attribution.gradinput import GradInput
from molucn.feat_attribution.ig import IntegratedGradient
from molucn.gnn.model import GNN
from molucn.utils.parser_utils import overall_parser
from molucn.utils.train_utils import DEVICE, test_epoch, train_epoch 
from molucn.utils.utils import get_mcs, set_seed



def main(args):
    set_seed(args.seed)
    train_params = (
        f"{args.conv}_{args.loss}_{args.pool}_{args.lambda1}_{args.explainer}"
    )


    ##### Data loading and pre-processing #####

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
        trainval_dataset = dill.load(handle)

    with open(file_test, "rb") as handle:
        test_dataset = dill.load(handle)

    # Ligands in testing set are NOT in training set!

    train_dataset, val_dataset = train_test_split(
        trainval_dataset, random_state=args.seed, test_size=args.val_set_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        num_workers=args.num_workers,  # num_workers can go into args
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=args.num_workers,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    num_node_features, num_edge_features = get_num_features(train_dataset[0])

    ##### GNN training #####
    model = GNN(
        num_node_features,
        num_edge_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        conv_name=args.conv,
        pool=args.pool,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-4
    )
    min_error = None

    # Early stopping
    last_loss = 100
    patience = 4
    trigger_times = 0

    for epoch in range(1, args.epoch + 1):

        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        loss = train_epoch(
            train_loader,
            model,
            optimizer,
            loss_type=args.loss,
            lambda1=args.lambda1,
        )

        rmse_val, pcc_val = test_epoch(val_loader, model)
        scheduler.step(rmse_val)
        if min_error is None or rmse_val <= min_error:
            min_error = rmse_val

        t2 = time.time()
        rmse_test, pcc_test = test_epoch(test_loader, model)
        t3 = time.time()


        if epoch % args.verbose == 0:
            print(
                "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Test Loss: {:.5f}, Test PCC: {:.5f}".format(
                    epoch, t3 - t1, lr, loss, rmse_test, pcc_test
                )
            )
            continue

        print(
            "Epoch{:4d}[{:.3f}s]: LR: {:.5f}, Loss: {:.5f}, Validation Loss: {:.5f}, Val PCC: {:.5f}".format(
                epoch, t2 - t1, lr, loss, rmse_val, pcc_val
            )
        )

        # Early stopping
        if epoch > 50:
            if loss > last_loss:
                trigger_times += 1
                print("Trigger Times:", trigger_times)

                if trigger_times >= patience:
                    print(
                        f"Early stopping after {epoch} epochs!\nStart to test process."
                    )
                    break

            else:
                print("trigger times: 0")
                trigger_times = 0
        last_loss = loss

    print("Final test rmse: {:.4f}".format(rmse_test))
    print("Final test pcc: {:.4f}".format(pcc_test))

    # Save GNN scores
    os.makedirs(args.log_path, exist_ok=True)
    global_res_path = osp.join(
        args.log_path, f"model_scores_gnn_{train_params}_{args.target}.csv"
    )
    df = pd.DataFrame(
        [
            [
                args.target,
                args.seed,
                args.conv,
                args.pool,
                args.loss,
                args.lambda1,
                args.explainer,
                rmse_test,
                pcc_test,
            ]
        ],
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
    df.to_csv(global_res_path, index=False)

    ##### Feature Attribution #####
    model.eval()

    if args.explainer == "gradinput":
        explainer = GradInput(DEVICE, model)
    elif args.explainer == "ig":
        explainer = IntegratedGradient(DEVICE, model)
    elif args.explainer == "cam":
        explainer = CAM(DEVICE, model)
    elif args.explainer == "gradcam":
        explainer = GradCAM(DEVICE, model)
    elif args.explainer == "diff":
        explainer = Diff(DEVICE, model)

    def get_colors(pairs_list, explainer):
        colors = []
        for hetero_data in pairs_list:
            data_i, data_j = hetero_data["data_i"], hetero_data["data_j"]
            color_pred_i, color_pred_j = (
                explainer.explain_graph(data_i),
                explainer.explain_graph(data_j),
            )
            colors.append([color_pred_i, color_pred_j])
        return colors

    t0 = time.time()
    train_colors = get_colors(trainval_dataset, explainer)
    time_explainer = (time.time() - t0) / len(trainval_dataset)
    print("Average time to generate 1 explanation: ", time_explainer)
    test_colors = get_colors(test_dataset, explainer)

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

    accs_train, f1s_train = get_scores(trainval_dataset, train_colors, set="train")
    accs_test, f1s_test = get_scores(test_dataset, test_colors, set="test")

    global_dir_train = get_global_directions(
        trainval_dataset, train_colors, set="train"
    )
    global_dir_test = get_global_directions(test_dataset, test_colors, set="test")

    local_dir_train = get_local_directions(trainval_dataset, train_colors, set="train")
    local_dir_test = get_local_directions(test_dataset, test_colors, set="test")

    os.makedirs(args.result_path, exist_ok=True)
    global_res_path = osp.join(
        args.result_path, f"attr_scores_{train_params}_{args.target}.csv"
    )

    accs_train, f1s_train = (
        np.nanmean(accs_train, axis=0).tolist(),
        np.nanmean(f1s_train, axis=0).tolist(),
    )
    accs_test, f1s_test = (
        np.nanmean(accs_test, axis=0).tolist(),
        np.nanmean(f1s_test, axis=0).tolist(),
    )
    global_dir_train, global_dir_test = (
        np.nanmean(global_dir_train, axis=0).tolist(),
        np.nanmean(global_dir_test, axis=0).tolist(),
    )
    local_dir_train, local_dir_test = (
        np.nanmean(local_dir_train, axis=0).tolist(),
        np.nanmean(local_dir_test, axis=0).tolist(),
    )
    n_mcs_train, n_mcs_test = (
        np.sum(get_mcs(trainval_dataset), axis=0).tolist(),
        np.sum(get_mcs(test_dataset), axis=0).tolist(),
    )

    res_dict = {
        "target": [args.target] * 10,
        "seed": [args.seed] * 10,
        "conv": [args.conv] * 10,
        "pool": [args.pool] * 10,
        "loss": [args.loss] * 10,
        "lambda1": [args.lambda1] * 10,
        "explainer": [args.explainer] * 10,
        "time": [round(time_explainer, 4)] * 10,
        "acc_train": accs_train,
        "acc_test": accs_test,
        "f1_train": f1s_train,
        "f1_test": f1s_test,
        "global_dir_train": global_dir_train,
        "global_dir_test": global_dir_test,
        "local_dir_train": local_dir_train,
        "local_dir_test": local_dir_test,
        "mcs": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        "n_mcs_train": n_mcs_train,
        "n_mcs_test": n_mcs_test,
    }
    df = pd.DataFrame({key: pd.Series(value) for key, value in res_dict.items()})
    df.to_csv(global_res_path, index=False)


if __name__ == "__main__":

    parser = overall_parser()
    args = parser.parse_args()

    for loss in ["MSE", "MSE+AC", "MSE+UCN"]:
        args.loss = loss
        main(args)
