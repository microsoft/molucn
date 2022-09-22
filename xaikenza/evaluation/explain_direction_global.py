import numpy as np

from sklearn.metrics import f1_score

f_score = lambda x, y: f1_score(x, y, zero_division=1)

EPS_ZERO = 1e-8


def get_imp_uncommon(mask, color_pred):
    assert len(color_pred) == len(mask)
    idx_noncommon = np.where(mask != 0)[0]
    if len(idx_noncommon) == 0:
        return 0
    color_pred_noncommon = np.array(
        [color_pred[idx] for idx in idx_noncommon]
    ).flatten()
    return np.mean(color_pred_noncommon)


def aggregated_color_direction(
    color_pred_i, color_pred_j, color_true_i, color_true_j, diff
):
    """
    Checks whether the average assigned colors to a substructural
    change matches the sign of the experimental difference
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


def get_global_directions(pairs_list, colors, set="train"):
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
        mcs_accs = np.where(mcs!=0,mcs,np.nan)*acc
        accs.append(mcs_accs)
    return np.array(accs)
