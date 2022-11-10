from typing import Callable, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.data import HeteroData

f_score = lambda x, y: f1_score(x, y, zero_division=1)

EPS_ZERO = 1e-8


def color_agreement(color_true: List[int], color_pred: List[float], metric_f: Callable):
    """
    Checks agreement between true and predicted colors.
    """
    assert len(color_true) == len(color_pred)
    idx_noncommon = np.where(color_true != 0)[0]
    if len(idx_noncommon) == 0:
        return -1.0
    color_true_noncommon = np.array([color_true[idx] for idx in idx_noncommon])
    color_pred_noncommon = np.array(
        [color_pred[idx] for idx in idx_noncommon]
    ).flatten()
    color_pred_noncommon[color_pred_noncommon == 0] += np.random.uniform(
        low=-EPS_ZERO, high=EPS_ZERO, size=np.sum(color_pred_noncommon == 0)
    )  # fix: assign small random value to exactly zero-signed preds.
    color_pred_noncommon = np.sign(color_pred_noncommon)
    return metric_f(color_true_noncommon, color_pred_noncommon)


def get_scores(
    pairs_list: List[HeteroData], colors: List[Tuple[List[float]]], set="train"
) -> Tuple[Iterable[float], Iterable[float]]:
    """Computes the accuracy and F1 color agreement scores of the atom color predictions for a given set of pairs for each MCS value.

    Args:
        pairs_list (List[HeteroData]): list of pairs colored by the model
        colors (List[Tuple[List[float]]]): node importance scores for each pair predicted by the feature attribution method
        set (str, optional): Defaults to "train".

    Returns:
        Iterable[float]: numpy array of accuracy scores for each target at each MCS threshold. Size (N_targets, N_MCS_thresh)
        Iterable[float]: numpy array of F1 scores for each target at each MCS threshold. Size (N_targets, N_MCS_thresh)
    """
    accs, f1s = [], []
    for k in range(len(pairs_list)):
        hetero_data = pairs_list[k]
        mcs = np.array(hetero_data["mcs"]).astype(dtype=bool)
        mcs_nan = np.where(mcs != 0, mcs, np.nan)
        data_i, data_j = hetero_data["data_i"], hetero_data["data_j"]
        node_imp_i, node_imp_j = colors[k]
        mask_i, mask_j = data_i.mask, data_j.mask
        acc_i = color_agreement(
            mask_i,
            node_imp_i,
            metric_f=accuracy_score,
        )

        acc_j = color_agreement(
            mask_j,
            node_imp_j,
            metric_f=accuracy_score,
        )

        f1_i = color_agreement(
            mask_i,
            node_imp_i,
            metric_f=f_score,
        )

        f1_j = color_agreement(
            mask_j,
            node_imp_j,
            metric_f=f_score,
        )
        if acc_i != -1:
            accs_i, f1s_i = mcs_nan * acc_i, mcs_nan * f1_i
            accs.append(accs_i)
            f1s.append(f1s_i)
        if acc_j != -1:
            accs_j, f1s_j = mcs_nan * acc_j, mcs_nan * f1_j
            accs.append(accs_j)
            f1s.append(f1s_j)
    return np.array(accs), np.array(f1s)
