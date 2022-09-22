import os
from typing import List, Tuple
import numpy as np
from xaikenza.utils.utils import (
    get_common_nodes,
    get_positions,
    get_substituents,
)

os.environ["WANDB_SILENT"] = "true"


EPS_ZERO = 1e-8


def get_imp_substituent(site, subs, pos_sub, color_pred):
    if pos_sub[site] == -1:
        return 0
    sub = subs[pos_sub[site]]
    color_pred_sub = np.array([color_pred[idx] for idx in sub]).flatten()
    return np.mean(color_pred_sub)


def attr_local_acc(
    color_pred_i,
    color_pred_j,
    subs_i,
    subs_j,
    pos_sub_i,
    pos_sub_j,
    a_i,
    a_j,
    cmn_sites,
):
    """
    Checks agreement between sign of sub attribution and activity cliff.
    """
    local_accs = []
    for site in cmn_sites:
        imp_i = get_imp_substituent(site, subs_i, pos_sub_i, color_pred_i)
        imp_j = get_imp_substituent(site, subs_j, pos_sub_j, color_pred_j)
        if imp_i == 0 and imp_j == 0:
            continue
        else:
            if (imp_i - imp_j) * (a_i - a_j) > 0:
                local_accs.append(1)
            else:
                local_accs.append(0)
    return local_accs


def get_local_directions(pairs_list: List[HeteroData], colors, set="train") -> Tuple:
    """_summary_

    Args:
        pairs_list (List[HeteroData]): _description_
        colors (_type_): _description_
        set (str, optional): _description_. Defaults to "train".

    Returns:
        Tuple: _description_
    """
    accs = []
    n_skip = 0
    for k in range(len(pairs_list)):
        hetero_data = pairs_list[k]
        mcs = np.array(hetero_data["mcs"]).astype(dtype=bool)
        data_i, data_j = hetero_data["data_i"], hetero_data["data_j"]
        color_pred_i, color_pred_j = colors[k]
        mask_i, mask_j = data_i.mask, data_j.mask
        if len(np.where(mask_i == 0)[0]) == 0 or len(np.where(mask_j == 0)[0]) == 0:
            # print('mask i, mask j:', mask_i, mask_j)
            # print('data i, data j:', data_i.smiles, data_j.smiles)
            print("No common nodes or weird coloring.")
            n_skip += 1
            continue
        a_i, a_j = data_i.y, data_j.y

        idx_common_i, idx_common_j, map_i, map_j = get_common_nodes(mask_i, mask_j)

        subs_i, as_i = get_substituents(data_i, idx_common_i, map_i)
        pos_sub_i = get_positions(subs_i, idx_common_i, map_i)

        subs_j, as_j = get_substituents(data_j, idx_common_j, map_j)
        pos_sub_j = get_positions(subs_j, idx_common_j, map_j)

        cmn_sites = np.unique(np.concatenate([as_i, as_j]))
        # print('cmn_sites', cmn_sites)
        local_accs = attr_local_acc(
            color_pred_i,
            color_pred_j,
            subs_i,
            subs_j,
            pos_sub_i,
            pos_sub_j,
            a_i,
            a_j,
            cmn_sites,
        )
        for local_acc in local_accs:
            res = np.where(mcs!=0,mcs,np.nan)*local_acc
            accs.append(res)
    # print("Number of skipped examples:", n_skip)
    return accs

