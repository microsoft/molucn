# Code adapted from tensorflow to pytorch from https://github.com/google-research/graph-attribution/tree/main/graph_attribution

import torch
from torch_geometric.data import Data
from molucn.feat_attribution.explainer_base import Explainer


class GradCAM(Explainer):
    """GradCAM: intermediate activations and gradients as input importance.
    GradCAM is the gradient version of CAM using ideas from Gradient times Input,
    removing the necessity of a GAP layer.
    For each convolution layer, in the case of graphs a GNN block, the
    activations can be retrieved and interpreted as a transformed version of the
    input. In a GNN intermediate activations are graphs with updated information.
    The gradient of a target y w.r.t these activations can be seen as measure of
    importance. The equation for gradCAM are:
      GradCAM(x) = reduce_i mean(w_i^T G_i(x), axis=-1)
    G_i(x) is the intermediate layer activations.
    reduce_i is an reduction operation over intermediate layers (e.g. mean, sum).
    Based on "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization" (https://arxiv.org/abs/1610.02391).
    """

    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        last_layer_only: bool = False,
    ):
        """GradCAM constructor.
        Args:
          last_layer_only: If to use only the last layer activations, if not will
            use all last activations.
          reduce_fn: Reduction operation for layers, should have the same call
            signature as tf.reduce_mean (e.g. tf.reduce_sum).
        """
        super(GradCAM, self).__init__(device, model)
        self.device = device
        self.last_layer_only = last_layer_only

    def explain_graph(self, graph: Data, model: torch.nn.Module = None) -> torch.Tensor:

        if model == None:
            model = self.model

        tmp_graph = graph.clone().to(self.device)

        acts, grads, _ = model.get_intermediate_activations_gradients(tmp_graph)
        node_w, edge_w = [], []
        layer_indices = [-1] if self.last_layer_only else list(range(len(acts)))
        for index in layer_indices:
            node_act, edge_act = acts[index]
            node_grad, edge_grad = grads[index]
            node_w.append(torch.einsum("ij,ij->i", node_act, node_grad))
            edge_w.append(torch.einsum("ij,ij->i", edge_act, edge_grad))

        node_weights = torch.stack(node_w, dim=0).sum(dim=0)
        edge_weights = torch.stack(edge_w, dim=0).sum(dim=0)
        for idx in range(graph.num_edges):
            e_imp = edge_weights[idx]
            node_weights[graph.edge_index[0, idx]] += e_imp / 2
            node_weights[graph.edge_index[1, idx]] += e_imp / 2

        return node_weights.cpu().detach().numpy()
