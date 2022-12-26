from molucn.evaluation.explain import get_colors
from molucn.evaluation.explain_direction_global import aggregated_color_direction
from molucn.gnn.train import train_gnn
import dill
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from molucn.utils.utils import (
    get_common_nodes,
    get_substituents,
) 
import os
import os.path as osp
import time

from molucn.feat_attribution.cam import CAM
from molucn.feat_attribution.diff import Diff
from molucn.feat_attribution.gradcam import GradCAM
from molucn.feat_attribution.gradinput import GradInput
from molucn.feat_attribution.ig import IntegratedGradient
from molucn.feat_attribution.graphsvx import GraphSVX
from molucn.utils.parser_utils import overall_parser
from molucn.utils.train_utils import DEVICE
from molucn.utils.utils import set_seed

from torch_geometric.data import Batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MSE_LOSS_FN = nn.MSELoss()



def get_num_cmn_sites(data_i, data_j):   
    idx_common_i, idx_common_j, map_i, map_j = get_common_nodes(
        data_i.mask.cpu(), data_j.mask.cpu()
    )
    subs_i, as_i = get_substituents(data_i, idx_common_i, map_i)
    subs_j, as_j = get_substituents(data_j, idx_common_j, map_j)
    cmn_sites = np.unique(np.concatenate([as_i, as_j]))
    return len(cmn_sites)

def get_scores_summary_by_pair(data_list, model, colors, args, type='test'):

    df = pd.DataFrame(
        columns=[
            "target",
            "type",
            "pair_id",
            "cmn_sites",
            "explainer",
            "rmse",
            "gdir",
        ]
    )
    
    for k, hetero_data in enumerate(data_list):
        data_i, data_j = hetero_data["data_i"].to(DEVICE), hetero_data["data_j"].to(DEVICE)
        out_i, out_j = model(data_i).squeeze().unsqueeze(0).detach(), model(data_j).squeeze().unsqueeze(0).detach()
        rmse_i = torch.sqrt(
            MSE_LOSS_FN(out_i, data_i.y)
        ).item()
        rmse_j = torch.sqrt(
            MSE_LOSS_FN(out_j, data_j.y)
        ).item()
        rmse = (rmse_i + rmse_j) / 2

        color_pred_i, color_pred_j = colors[k]
        mask_i, mask_j = data_i.mask, data_j.mask
        a_i, a_j = data_i.y, data_j.y
        diff = a_i - a_j

        acc = aggregated_color_direction(
            color_pred_i, color_pred_j, mask_i, mask_j, diff
        )

        mcs = np.array(hetero_data["mcs"]).astype(dtype=bool)

        n_cmn_sites = get_num_cmn_sites(data_i, data_j)

        row = {
            "target": args.target,
            "type": type,
            "pair_id": k,
            "mcs": mcs,
            "cmn_sites": n_cmn_sites,
            "explainer": args.explainer,
            "rmse": rmse,
            "gdir": acc,
            }
        df = df.append(row, ignore_index=True)

    return df

def main_by_pair(args):
    set_seed(args.seed)
    train_params = (
        f"{args.conv}_{args.loss}_{args.pool}_{args.lambda1}_{args.explainer}"
    )

    ##### Data loading and pre-processing #####
    print(args.data_path)

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

    model, rmse_test, pcc_test = train_gnn(args)
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
    elif args.explainer == "graphsvx":
        explainer = GraphSVX(DEVICE, model)

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

    ##### Summary #####
    test_summary = get_scores_summary_by_pair(test_dataset, model, test_colors, args, type='test')
    train_summary = get_scores_summary_by_pair(trainval_dataset, model, train_colors, args, type='train')

    os.makedirs(args.result_path, exist_ok=True)
    global_res_path = osp.join(
        args.result_path, f"pairs_scores_{train_params}_{args.target}.csv"
    )

    summary = pd.concat([train_summary, test_summary])
    summary.to_csv(global_res_path, index=False)
    return summary


if __name__ == "__main__":
    parser = overall_parser()
    args = parser.parse_args()
    args.loss = 'MSE+UCN'
    main_by_pair(args)



