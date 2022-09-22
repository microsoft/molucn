# Code adapted from tensorflow to pytorch from https://github.com/josejimenezluna/xaibench_tf

from copy import deepcopy

import torch

from feat_attribution.explainer_base import Explainer
from torch_geometric.data import Data


class Diff(Explainer):
    def __init__(self, device, model):
        super(Diff, self).__init__(device, model)
        self.device = device

    def explain_graph(self, graph, model=None):

        if model == None:
            model = self.model

        tmp_graph = graph.clone().to(self.device)
        masked_gs = gen_masked_atom_feats(tmp_graph)

        mod_preds = torch.Tensor([model(masked_g) for masked_g in masked_gs])
        og_pred = torch.Tensor([model(tmp_graph)] * len(mod_preds))

        node_weights = torch.squeeze(og_pred - mod_preds)
        return node_weights.cpu().detach().numpy()


def gen_masked_atom_feats(og_g):
    """
    Given a graph, returns a list of graphs where individual atoms
    are masked.
    """
    masked_gs = []
    for node_idx in range(og_g.num_nodes):
        g = deepcopy(og_g)
        g.x[node_idx] *= 0.0
        masked_gs.append(g)
    return masked_gs
