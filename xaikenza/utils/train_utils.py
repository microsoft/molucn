from functools import wraps

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from tqdm import tqdm
from xaikenza.gnn.loss import loss_uncommon_node, loss_uncommon_node_local

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MSE_LOSS_FN = nn.MSELoss()


# General function for training graph classification(regresion) task and
# node classification task under multiple graphs.
def train_epoch(
    train_loader,  # def train_epoch
    model: nn.Module,  # type hints
    optimizer,
    loss_type: str = 'MSE',
    lambda1: int = 1,
):
    model.train()
    loss_all = 0

    progress = tqdm(train_loader)
    total = 0
    for hetero_data in progress:
        data_i, data_j = Batch().from_data_list(
            hetero_data["data_i"]
        ), Batch().from_data_list(hetero_data["data_j"])
        optimizer.zero_grad()
        data_i, data_j = data_i.to(DEVICE), data_j.to(DEVICE)
        out_i, out_j = torch.squeeze(model(data_i)), torch.squeeze(model(data_j))
        if out_i.dim()==0:
            out_i = out_i.unsqueeze(0)
        if out_j.dim()==0:
            out_j = out_j.unsqueeze(0)
            
        mse_i, mse_j = MSE_LOSS_FN(
            out_i, data_i.y.to(DEVICE)
        ), MSE_LOSS_FN(out_j, data_j.y.to(DEVICE))
        loss = (mse_i + mse_j) / 2

        if loss_type == "MSE+UCNlocal":
            loss_batch, n_subs_in_batch = loss_uncommon_node_local(
                data_i, data_j, model
            )
            loss += lambda1 * loss_batch

            loss.backward()
            loss_all += loss.item() * n_subs_in_batch

            optimizer.step()
            progress.set_postfix({"loss": loss.item()})
            total += n_subs_in_batch
        else:
            if loss_type == "MSE+AC":
                loss += MSE_LOSS_FN(
                    out_i - out_j,
                    data_i.y.to(DEVICE) - data_j.y.to(DEVICE)
                )
            elif loss_type == "MSE+UCN":
                loss += lambda1 * loss_uncommon_node(data_i, data_j, model)

            loss.backward()
            loss_all += loss.item() * hetero_data.num_graphs

            optimizer.step()
            progress.set_postfix({"loss": loss.item()})
            total = len(train_loader.dataset)
    return loss_all / total


# Gtest / val do not need to compute metrics on a batch basis
def test_epoch(
    test_loader,
    model,
):

    model.eval()
    error = 0
    pcc = 0

    with torch.no_grad():
        k = 0
        for hetero_data in test_loader:
            data_i, data_j = Batch().from_data_list(hetero_data["data_i"]).to(
                DEVICE
            ), Batch().from_data_list(hetero_data["data_j"]).to(DEVICE)
            out_i, out_j = model(data_i), model(data_j)
            rmse_i = torch.sqrt(
                MSE_LOSS_FN(torch.squeeze(out_i).detach(), data_i.y)
            ).item()
            rmse_j = torch.sqrt(
                MSE_LOSS_FN(torch.squeeze(out_j).detach(), data_j.y)
            ).item()
            error += (rmse_i + rmse_j) / 2 * hetero_data.num_graphs

            pcc_i = np.corrcoef(
                torch.squeeze(out_i).cpu().numpy(), data_i.y.cpu().numpy()
            )[0, 1]
            pcc_j = np.corrcoef(
                torch.squeeze(out_j).cpu().numpy(), data_j.y.cpu().numpy()
            )[0, 1]
            if np.isnan(pcc_i) | np.isnan(pcc_j):
                try:
                    k += 1
                    raise ValueError(
                        "PCC is NaN. The batch has the same compound for all pairs. This batch will be ignored for the final results computation.\n \
                                    If you want to avoid removing batches, try shuffling the pair set with new seed."
                    )
                except ValueError as err:
                    print(err)
            else:
                pcc += (pcc_i + pcc_j) / 2 * hetero_data.num_graphs

        return error / len(test_loader.dataset), pcc / (len(test_loader.dataset) - k)


def overload(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        if len(args) + len(kargs) == 2:
            if len(args) == 2:  # for inputs like model(g)
                g = args[1]
            else:  # for inputs like model(graph=g)
                g = kargs["graph"]
            return func(args[0], g.x, g.edge_index, g.edge_attr, g.batch)

        elif len(args) + len(kargs) == 5:
            if len(args) == 5:  # for inputs like model(x, ..., batch)
                return func(*args)
            else:  # for inputs like model(x=x, ..., batch=batch)
                return func(args[0], **kargs)

        elif len(args) + len(kargs) == 6:
            if len(args) == 6:  # for inputs like model(x, ..., batch, pos)
                return func(*args[:-1])
            else:  # for inputs like model(x=x, ..., batch=batch, pos=pos)
                return func(
                    args[0],
                    kargs["x"],
                    kargs["edge_index"],
                    kargs["edge_attr"],
                    kargs["batch"],
                )
        else:
            raise TypeError

    return wrapper
