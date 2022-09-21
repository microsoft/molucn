import math

import numpy as np
import torch

EPS = 1e-6


class Explainer(object):
    def __init__(self, device, model):
        self.device = device
        self.model = model
        self.model.eval()
        self.model_name = self.model.__class__.__name__
        self.name = self.__class__.__name__

    def explain_graph(self, graph, **kwargs):
        """
        Main part for different graph attribution methods
        :param graph: target graph instance to be explained
        :param kwargs:
        :return: edge_imp, i.e., attributions for edges, which are derived from the attribution methods.
        """
        raise NotImplementedError

    @staticmethod
    def get_rank(lst, r=1):

        topk_idx = list(np.argsort(-lst))
        top_pred = np.zeros_like(lst)
        n = len(lst)
        k = int(r * n)
        for i in range(k):
            top_pred[topk_idx[i]] = n - i
        return top_pred

    @staticmethod
    def norm_imp(imp):
        # imp[imp < 0] = 0
        imp += 1e-16
        return imp / imp.sum()

    def _relabel(self, g, edge_index):

        sub_nodes = torch.unique(edge_index)
        x = g.x[sub_nodes]
        batch = g.batch[sub_nodes]
        row, col = edge_index
        pos = None
        try:
            pos = g.pos[sub_nodes]
        except:
            pass

        # remapping the nodes in the explanatory subgraph to new ids.
        node_idx = row.new_full((g.num_nodes,), -1)
        node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
        edge_index = node_idx[edge_index]
        return x, edge_index, batch, pos

    def _reparameterize(self, log_alpha, beta=0.1, training=True):

        if training:
            random_noise = torch.rand(log_alpha.size()).to(self.device)
            gate_inputs = torch.log2(random_noise) - torch.log2(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta + EPS
            gate_inputs = gate_inputs.sigmoid()
        else:
            gate_inputs = log_alpha.sigmoid()

        return gate_inputs

    def pack_explanatory_subgraph(
        self, top_ratio=0.2, graph=None, imp=None, relabel=True
    ):

        if graph is None:
            graph, imp = self.last_result
        assert len(imp) == graph.num_edges, "length mismatch"

        top_idx = torch.LongTensor([])
        graph_map = graph.batch[graph.edge_index[0, :]]
        exp_subgraph = graph.clone()
        exp_subgraph.y = graph.y
        for i in range(graph.num_graphs):
            edge_indicator = torch.where(graph_map == i)[0].detach().cpu()
            Gi_n_edge = len(edge_indicator)
            topk = min(max(math.ceil(top_ratio * Gi_n_edge), 1), Gi_n_edge)
            Gi_pos_edge_idx = np.argsort(-imp[edge_indicator])[:topk]
            top_idx = torch.cat([top_idx, edge_indicator[Gi_pos_edge_idx]])
        # retrieval properties of the explanatory subgraph
        # .... the edge_attr.
        exp_subgraph.edge_attr = graph.edge_attr[top_idx]
        # .... the edge_index.
        exp_subgraph.edge_index = graph.edge_index[:, top_idx]
        # .... the nodes.
        # exp_subgraph.x = graph.x
        if relabel:
            (
                exp_subgraph.x,
                exp_subgraph.edge_index,
                exp_subgraph.batch,
                exp_subgraph.pos,
            ) = self._relabel(exp_subgraph, exp_subgraph.edge_index)

        return exp_subgraph

    def evaluate_recall(self, topk=10):

        graph, imp = self.last_result
        E = graph.num_edges
        if isinstance(graph.ground_truth_mask, list):
            graph.ground_truth_mask = graph.ground_truth_mask[0]
        index = np.argsort(-imp)[:topk]
        values = graph.ground_truth_mask[index]
        return float(values.sum()) / float(graph.ground_truth_mask.sum())

    def evaluate_acc(self, top_ratio_list, graph=None, imp=None):

        if graph is None:
            assert self.last_result is not None
            graph, imp = self.last_result
        acc = np.array([[]])
        prob = np.array([[]])
        y = graph.y
        for idx, top_ratio in enumerate(top_ratio_list):

            if top_ratio == 1.0:
                self.model(graph)
            else:
                exp_subgraph = self.pack_explanatory_subgraph(
                    top_ratio, graph=graph, imp=imp
                )
                self.model(exp_subgraph)
            res_acc = (
                (y == self.model.readout.argmax(dim=1))
                .detach()
                .cpu()
                .float()
                .view(-1, 1)
                .numpy()
            )
            res_prob = (
                self.model.readout[0, y].detach().cpu().float().view(-1, 1).numpy()
            )
            acc = np.concatenate([acc, res_acc], axis=1)
            prob = np.concatenate([prob, res_prob], axis=1)
        return acc, prob
