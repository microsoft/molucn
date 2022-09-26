# Code adapted from tensorflow to pytorch from https://github.com/google-research/graph-attribution/tree/main/graph_attribution

import torch
from torch_geometric.data import Data

from xaicode.feat_attribution.explainer_base import Explainer


class CAM(Explainer):
    """CAM: Decompose output as a linear sum of nodes and edges.
    CAM (Class Activation Maps) assumes the model has a global average pooling
    layer (GAP-layer) right before prediction. This means the prediction can be
    written as weighted sum of the pooled elements plus an final activation.
    In the case of graphs, a GAP layer should take nodes and edges activations
    and will sum them to create a graph embedding layer. The CAM model follows
    the equation:
      CAM(x) = (node_activations + edge_activations)*w
    Based on "Learning Deep Features for Discriminative Localization"
    (https://arxiv.org/abs/1512.04150).
    """

    def __init__(self, device: torch.device, model: torch.nn.Module):
        super(CAM, self).__init__(device, model)
        self.device = device

    def explain_graph(self, graph: Data, model: torch.nn.Module = None) -> torch.Tensor:

        if model == None:
            model = self.model

        tmp_graph = graph.clone().to(self.device)

        node_act, edge_act = model.get_gap_activations(tmp_graph)
        weights = model.get_prediction_weights()
        node_weights = torch.einsum("ij,j", node_act, weights)
        edge_weights = torch.einsum("ij,j", edge_act, weights)

        for idx in range(graph.num_edges):
            e_imp = edge_weights[idx]
            node_weights[graph.edge_index[0, idx]] += e_imp / 2
            node_weights[graph.edge_index[1, idx]] += e_imp / 2

        return node_weights.cpu().detach().numpy()
