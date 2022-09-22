from typing import List, Tuple

import numpy as np
from sklearn.metrics import f1_score
from torch_geometric.data import HeteroData

f_score = lambda x, y: f1_score(x, y, zero_division=1)

EPS_ZERO = 1e-8


def aggregated_color_direction(
    color_pred_i: List[float],
    color_pred_j: List[float],
    color_true_i: List[float],
    color_true_j: List[float],
    diff: float,
) -> float:
    """Checks whether the average assigned colors to a substructural change matches the sign of the experimental difference.

    Args:
        color_pred_i (List[float]): the color assigned to the first molecule by the feature attribution method
        color_pred_j (List[float]): the color assigned to the second molecule by the feature attribution method
        color_true_i (List[float]): the true coloring of the first molecule
        color_true_j (List[float]): the true coloring of the second molecule
        diff (float):the difference in activity between the two molecules

    Returns:
        float: the global direction score for the pair of molecules
    """
    assert len(color_true_i) == len(color_pred_i)
    assert len(color_true_j) == len(color_pred_j)

    idx_noncommon_i = [idx for idx, val in enumerate(color_true_i) if val != 0.0]
    idx_noncommon_j = [idx for idx, val in enumerate(color_true_j) if val != 0.0]

    color_pred_i_noncommon = np.array(
        [color_pred_i[idx] for idx in idx_noncommon_i]
    ).flatten()
    color_pred_j_noncommon = np.array(
        [color_pred_j[idx] for idx in idx_noncommon_j]
    ).flatten()

    # If any of these arrays are empty (ie one molecule is a substructure of the other)
    # assign 0 to the mean

    if len(color_pred_i_noncommon) == 0:
        color_pred_i_noncommon = np.zeros(1)

    if len(color_pred_j_noncommon) == 0:
        color_pred_j_noncommon = np.zeros(1)

    i_higher_j = np.mean(color_pred_i_noncommon) > np.mean(color_pred_j_noncommon)

    if diff > 0 and i_higher_j:
        score = 1.0

    elif diff < 0 and not i_higher_j:
        score = 1.0
    else:
        score = 0.0
    return score


def get_global_directions(
    pairs_list: List[HeteroData], colors: Tuple[List[float]], set: str = "train"
) -> np.ndarray:
    """Computes the global direction scores for all the pairs in the dataset."""
    accs = []
    for k in range(len(pairs_list)):
        hetero_data = pairs_list[k]
        mcs = np.array(hetero_data["mcs"]).astype(dtype=bool)
        data_i, data_j = hetero_data["data_i"], hetero_data["data_j"]
        color_pred_i, color_pred_j = colors[k]
        mask_i, mask_j = data_i.mask, data_j.mask
        a_i, a_j = data_i.y, data_j.y
        diff = a_i - a_j
        acc = aggregated_color_direction(
            color_pred_i, color_pred_j, mask_i, mask_j, diff
        )
        mcs_accs = np.where(mcs != 0, mcs, np.nan) * acc
        accs.append(mcs_accs)
    return np.array(accs)
