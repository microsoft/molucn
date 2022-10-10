# Code adapted from tensorflow to pytorch from https://github.com/google-research/graph-attribution/tree/main/graph_attribution
from copy import deepcopy

import networkx as nx
import torch
from torch_geometric.data import Data
from molucn.feat_attribution.explainer_base import Explainer


def gen_steps(graph, n_steps: int, version=2):
    """
    Generates straight path between the node features of `graph`
    using a Monte Carlo approx. of `n_steps`.
    """
    graphs = []

    feat = graph.x
    if version == 3:
        e_feat = graph.edge_attr

    for step in range(1, n_steps + 1):
        factor = step / n_steps
        g = deepcopy(graph)
        g.x = factor * feat
        if version == 3:
            g.edge_attr = factor * e_feat
        graphs.append(g)
    return graphs


class IntegratedGradient(Explainer):
    def __init__(self, device, model):
        super(IntegratedGradient, self).__init__(device, model)
        self.device = device

    def explain_graph(
        self,
        graph: Data,
        model: torch.nn.Module = None,
        n_steps: int = 50,
        version: int = 2,
        feature_scale: bool = True,
    ) -> torch.Tensor:

        """Computes path integral of the node features of `graph` for a
        specific `task` number, using a Monte Carlo approx. of `n_steps`.

        Parameters
        ----------
        graph : DGL graph
        g_feat : torch.Tensor
        model : MPNN
        task : int, optional
        n_steps : int, optional
        version : int, optional

        Returns
        -------
        atom_importances : torch.Tensor
        bond_importances : torch.Tensor
        values_global : torch.Tensor
        """

        if model == None:
            model = self.model

        tmp_graph = graph.clone().to(self.device)

        graphs = gen_steps(tmp_graph, n_steps=n_steps, version=version)
        values_atom = []
        values_bond = []
        for g in graphs:
            g = g.to(self.device)
            g.x.requires_grad_()
            g.edge_attr.requires_grad_()
            pred = model(g)
            pred.backward()
            atom_grads = g.x.grad.unsqueeze(2)
            bond_grads = g.edge_attr.grad.unsqueeze(2)

            values_atom.append(atom_grads)
            values_bond.append(bond_grads)

        node_weights = torch.cat(values_atom, dim=2).mean(dim=2).cpu()
        edge_weights = torch.cat(values_bond, dim=2).mean(dim=2).cpu()

        if feature_scale:
            node_weights *= graph.x
            edge_weights *= graph.edge_attr

        node_weights = node_weights.sum(dim=1).numpy()
        edge_weights = edge_weights.sum(dim=1).numpy()

        for idx in range(graph.num_edges):
            e_imp = edge_weights[idx]
            node_weights[graph.edge_index[0, idx]] += e_imp / 2
            node_weights[graph.edge_index[1, idx]] += e_imp / 2

        return node_weights
