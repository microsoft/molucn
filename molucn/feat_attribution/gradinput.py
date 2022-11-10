# Code adapted from tensorflow to pytorch from https://github.com/google-research/graph-attribution/tree/main/graph_attribution
import torch
from torch.autograd import Variable
from torch_geometric.data import Data
from molucn.feat_attribution.explainer_base import Explainer


class GradInput(Explainer):
    def __init__(self, device: torch.device, model: torch.nn.Module):
        super(GradInput, self).__init__(device, model)
        self.device = device

    def explain_graph(self, graph: Data, model: torch.nn.Module = None) -> torch.Tensor:

        if model == None:
            model = self.model

        tmp_graph = graph.clone().to(self.device)
        tmp_graph.edge_attr = Variable(tmp_graph.edge_attr, requires_grad=True)
        tmp_graph.x = Variable(tmp_graph.x, requires_grad=True)
        pred = model(tmp_graph)
        pred.backward()

        node_weights = torch.einsum("ij,ij->i", tmp_graph.x, tmp_graph.x.grad)
        edge_weights = torch.einsum(
            "ij,ij->i", tmp_graph.edge_attr, tmp_graph.edge_attr.grad
        )

        for idx in range(graph.num_edges):
            e_imp = edge_weights[idx]
            node_weights[graph.edge_index[0, idx]] += e_imp / 2
            node_weights[graph.edge_index[1, idx]] += e_imp / 2
        return node_weights.cpu().detach().numpy()
