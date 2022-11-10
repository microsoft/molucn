from typing import List, Tuple

import numpy as np
from torch_geometric.data import HeteroData
from molucn.utils.utils import get_common_nodes, get_positions, get_substituents

EPS_ZERO = 1e-8


def get_imp_substituent(site, subs, pos_sub, color_pred):
    """Returns the importance of the substituent at the given site."""
    if pos_sub[site] == -1:
        return 0
    sub = subs[pos_sub[site]]
    color_pred_sub = np.array([color_pred[idx] for idx in sub]).flatten()
    return np.mean(color_pred_sub)


def attr_local_acc(
    color_pred_i: List[float],
    color_pred_j: List[float],
    subs_i: List[List[int]],
    subs_j: List[List[int]],
    pos_sub_i: List[int],
    pos_sub_j: List[int],
    a_i: float,
    a_j: float,
    cmn_sites: List[int],
) -> List[float]:

    """Checks agreement between sign of sub attribution and activity cliff.

    Args:
        color_pred_i (List[float]): the color assigned to the first molecule by the feature attribution method
        color_pred_j (List[float]): the color assigned to the second molecule by the feature attribution method
        subs_i (List[List[int]]): the list of substituents of the first molecule
        subs_j (List[List[int]]): the list of substituents of the second molecule
        pos_sub_i (List[int]): the position of the substituent of the first molecule
        pos_sub_j (List[int]): the position of the substituent of the second molecule
        a_i (float): the activity of the first molecule
        a_j (float): the activity of the second molecule
        cmn_sites (List[int]): the list of common sites between the two molecules

    Returns:
        float: the local direction scores for each substituent at each site in the common sites
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


def get_local_directions(
    pairs_list: List[HeteroData], colors: Tuple[List[float]], set="train"
) -> List[float]:
    """Computes the local direction scores for all the pairs in the dataset."""
    accs = []
    n_skip = 0
    for k in range(len(pairs_list)):
        hetero_data = pairs_list[k]
        mcs = np.array(hetero_data["mcs"]).astype(dtype=bool)
        data_i, data_j = hetero_data["data_i"], hetero_data["data_j"]
        color_pred_i, color_pred_j = colors[k]
        mask_i, mask_j = data_i.mask, data_j.mask
        if len(np.where(mask_i == 0)[0]) == 0 or len(np.where(mask_j == 0)[0]) == 0:
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
            res = np.where(mcs != 0, mcs, np.nan) * local_acc
            accs.append(res)
    return accs
