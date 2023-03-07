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
from molucn.feat_attribution.graphsvx import GraphSVX
from molucn.gnn.model import GNN
from molucn.utils.train_utils import DEVICE, test_epoch, train_epoch
from molucn.utils.utils import get_mcs, set_seed, get_colors


def test_gradinput_MSE():
    seed = 1337
    set_seed(seed)
    ##### Data loading and pre-processing #####
    # Check that data exists
    file_train = osp.join("data/1D3G-BRE/1D3G-BRE_seed_1337_train.pt")
    file_test = osp.join("data/1D3G-BRE/1D3G-BRE_seed_1337_test.pt")
    assert osp.exists(file_train)
    assert osp.exists(file_test)
    with open(file_train, "rb") as handle:
        trainval_dataset = dill.load(handle)
    with open(file_test, "rb") as handle:
        test_dataset = dill.load(handle)

    # Ligands in testing set are NOT in training set!
    train_dataset, val_dataset = train_test_split(
        trainval_dataset, random_state=seed, test_size=0.1
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
        num_workers=0,  # num_workers can go into args
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    num_node_features, num_edge_features = get_num_features(train_dataset[0])
    assert num_node_features == 50
    assert num_edge_features == 10
    ##### GNN training #####
    model = GNN(
        num_node_features,
        num_edge_features,
        hidden_dim=16,
        num_layers=3,
        conv_name="nn",
        pool="mean",
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-4
    )
    min_error = None

    # Early stopping
    last_loss = 100
    patience = 4
    trigger_times = 0

    for epoch in range(1, 200 + 1):

        t1 = time.time()
        lr = scheduler.optimizer.param_groups[0]["lr"]

        loss = train_epoch(
            train_loader,
            model,
            optimizer,
            loss_type="MSE",
            lambda1=1.0,
        )

        rmse_val, pcc_val = test_epoch(val_loader, model)
        scheduler.step(rmse_val)
        if min_error is None or rmse_val <= min_error:
            min_error = rmse_val

        t2 = time.time()
        rmse_test, pcc_test = test_epoch(test_loader, model)
        t3 = time.time()

        if epoch % 10 == 0:
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
    assert rmse_test < 0.4
    assert pcc_test > 0.985

    ##### Feature Attribution #####
    model.eval()
    explainer = GradInput(DEVICE, model)

    t0 = time.time()
    train_colors = get_colors(trainval_dataset, explainer)
    time_explainer = (time.time() - t0) / len(trainval_dataset)
    print("Average time to generate 1 explanation: ", time_explainer)
    test_colors = get_colors(test_dataset, explainer)

    accs_train, f1s_train = get_scores(trainval_dataset, train_colors, set="train")
    accs_test, f1s_test = get_scores(test_dataset, test_colors, set="test")

    global_dir_train = get_global_directions(
        trainval_dataset, train_colors, set="train"
    )
    global_dir_test = get_global_directions(test_dataset, test_colors, set="test")

    local_dir_train = get_local_directions(trainval_dataset, train_colors, set="train")
    local_dir_test = get_local_directions(test_dataset, test_colors, set="test")

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
    print("Train Accuracy: ", accs_train)
    print("Test Accuracy: ", accs_test)
    print("Train F1: ", f1s_train)
    print("Test F1: ", f1s_test)
    print("Train Global Directions: ", global_dir_train)
    print("Test Global Directions: ", global_dir_test)
    print("Train Local Directions: ", local_dir_train)
    print("Test Local Directions: ", local_dir_test)
    assert np.max(accs_train) > 0.5
    assert np.max(accs_test) > 0.6
    assert np.max(f1s_train) > 0.4
    assert np.max(f1s_test) > 0.5
    assert np.max(global_dir_train) > 0.77
    assert np.max(global_dir_test) > 0.85

