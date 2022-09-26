import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from xaicode.utils.utils import (
    get_common_nodes,
    get_positions,
    get_substituent_info,
    get_substituents,
)


def loss_uncommon_node(
    data_i: Tensor, data_j: Tensor, model: nn.Module, reduction: str = "mean"
) -> Tensor:
    """
    Compute the loss that correlates decoration embeddings and activity cliff.
    """
    emb_i = model.get_uncommon_graph_rep(data_i)
    emb_j = model.get_uncommon_graph_rep(data_j)
    delta_emb = torch.squeeze(emb_i) - torch.squeeze(emb_j)
    delta_y = data_i.y - data_j.y
    if delta_emb.dim() == 0:
        delta_emb = delta_emb.unsqueeze(0)
    loss = F.mse_loss(delta_emb, delta_y, reduction=reduction)
    return loss


def loss_uncommon_node_local(
    data_i: Tensor, data_j: Tensor, model: nn.Module, reduction: str = "mean"
) -> Tensor:
    """
    Compute the loss that correlates decoration embeddings and activity cliff at individual site of the decorations.
    """
    idx_common_i, idx_common_j, map_i, map_j = get_common_nodes(
        data_i.mask.cpu(), data_j.mask.cpu()
    )

    subs_i, as_i = get_substituents(data_i, idx_common_i, map_i)
    pos_sub_i = get_positions(subs_i, idx_common_i, map_i)

    subs_j, as_j = get_substituents(data_j, idx_common_j, map_j)
    pos_sub_j = get_positions(subs_j, idx_common_j, map_j)

    cmn_sites = np.unique(np.concatenate([as_i, as_j]))
    loss = []
    pos_sub_filtered_i = {k: v for k, v in pos_sub_i.items() if k in cmn_sites}
    pos_sub_filtered_j = {k: v for k, v in pos_sub_j.items() if k in cmn_sites}
    for site in cmn_sites:

        if pos_sub_filtered_i[site] == -1:
            emb_i = torch.zeros(1).to(data_i.x.device)
        else:
            sub = subs_i[pos_sub_filtered_i[site]]
            emb_i = model.get_substituent_rep(sub, data_i)

        if pos_sub_filtered_j[site] == -1:
            emb_j = torch.zeros(1).to(data_j.x.device)
        else:
            sub = subs_j[pos_sub_filtered_j[site]]
            emb_j = model.get_substituent_rep(sub, data_j)

        batch_i, a_i = get_substituent_info(site, data_i, map_i)
        batch_j, a_j = get_substituent_info(site, data_j, map_j)
        loss.append(
            F.mse_loss(
                torch.squeeze(emb_i - emb_j), a_i - a_j, reduction=reduction
            ).item()
        )
    return np.mean(loss), len(cmn_sites)


def loss_activity_cliff(
    input_i: Tensor,
    input_j: Tensor,
    target_i: Tensor,
    target_j: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute the loss that preserves the activity cliff.
    """
    loss = F.mse_loss(input_i - input_j, target_i - target_j, reduction=reduction)
    return loss
