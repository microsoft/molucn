from typing import List

import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Linear as Lin
from torch.nn import ModuleList, ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import (
    BatchNorm,
    GATConv,
    GENConv,
    GINEConv,
    NNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from xaicode.gnn.aggregation import AttentionalAggregation
from xaicode.utils.train_utils import overload


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_classes: int = 1,
        conv_name: str = "nn",
        pool: str = "mean",
    ):
        super().__init__()
        (
            self.num_node_features,
            self.num_edge_features,
            self.num_classes,
            self.num_layers,
            self.hidden_dim,
        ) = (
            num_node_features,
            num_edge_features,
            num_classes,
            num_layers,
            hidden_dim,
        )

        self.node_emb = Lin(self.num_node_features, self.hidden_dim)
        self.edge_emb = Lin(self.num_edge_features, self.hidden_dim)
        self.relu_nn = ModuleList([ReLU() for i in range(self.num_layers)])

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(self.num_layers):
            if conv_name == "nn":
                conv = NNConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    nn=Seq(Lin(self.hidden_dim, self.hidden_dim * self.hidden_dim)),
                )
            elif conv_name == "gine":
                conv = GINEConv(
                    nn=Seq(
                        Lin(self.hidden_dim, 2 * self.hidden_dim),
                        self.relu_nn[i],
                        Lin(2 * self.hidden_dim, self.hidden_dim),
                    )
                )
            elif conv_name == "gat":
                conv = GATConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    edge_dim=self.hidden_dim,
                    concat=False,
                )
            elif conv_name == "gen":
                conv = GENConv(self.hidden_dim, self.hidden_dim)
            else:
                raise ValueError(f"Unknown convolutional layer {conv_name}")
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.hidden_dim))
            self.relus.append(ReLU())

        self.lin1 = Lin(self.hidden_dim, self.hidden_dim // 2)
        self.relu = ReLU()
        self.lin2 = Lin(self.hidden_dim // 2, self.num_classes)

        self.pool = pool
        if self.pool == "att":
            # self.pool_fn = SAGpooling(self.hidden_dim, min_score=0)
            self.pool_fn = AttentionalAggregation(
                gate_nn=Seq(Lin(self.hidden_dim, 1)),
                nn=Seq(Lin(self.hidden_dim, self.hidden_dim)),
            )
        elif self.pool == "mean":
            self.pool_fn = global_mean_pool
        elif self.pool == "max":
            self.pool_fn = global_max_pool
        elif self.pool == "add":
            self.pool_fn = global_add_pool
        elif self.pool == "mean+att":
            self.pool_fn = global_mean_pool
            self.pool_fn_ucn = AttentionalAggregation(
                gate_nn=Seq(Lin(self.hidden_dim, 1)),
                nn=Seq(Lin(self.hidden_dim, self.hidden_dim)),
            )
        else:
            raise ValueError(f"Unknown pool {self.pool}")

    @overload
    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = self.pool_fn(node_x, batch)
        return self.get_pred(graph_x)

    @overload
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        """Returns the node embeddings just before the pooling layer."""
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for conv, batch_norm, ReLU in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_attr)
            x = ReLU(batch_norm(x))
        node_x = x
        return node_x

    @overload
    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        """Returns the graph embedding after pooling."""
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = self.pool_fn(node_x, batch)
        return graph_x

    @overload
    def get_gap_activations(self, x, edge_index, edge_attr, batch):
        """Returns the node-wise and edge-wise contributions to graph embedding (before the pooling layer)."""
        node_act = self.get_node_reps(x, edge_index, edge_attr, batch)
        edge_act = self.edge_emb(edge_attr)
        return node_act, edge_act

    def get_prediction_weights(self):
        """Gets prediction weights of the before last layer."""
        w = self.lin1.weight.data[0]
        return w

    @overload
    def get_intermediate_activations_gradients(self, x, edge_index, edge_attr, batch):
        """Gets intermediate layer activations and gradients."""
        acts = []
        grads = []

        x = Variable(x, requires_grad=True)
        edge_attr = Variable(edge_attr, requires_grad=True)
        acts.append((x, edge_attr))

        x = self.node_emb(x)
        x.retain_grad()
        edge_attr = self.edge_emb(edge_attr)
        edge_attr.retain_grad()
        acts.append((x, edge_attr))

        for conv, batch_norm, ReLU in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_attr)
            x = ReLU(batch_norm(x))
            x.retain_grad()
            edge_attr.retain_grad()
            acts.append((x, edge_attr))
        node_x = x
        graph_x = self.pool_fn(node_x, batch)
        y = self.get_pred(graph_x)
        y.backward()
        grads = [(act[0].grad, act[1].grad) for act in acts]
        return acts, grads, y

    def get_substituent_rep(self, sub: List[int], data: torch.Tensor) -> torch.Tensor:
        """Gets the hidden representation of one substituent in a molecule as the model prediction on this subgraph.
        Args:
            sub (List[int]): list of the indicies of the substituent atoms
            data (torch.Tensor): data object containing the graph

        Returns:
            torch.Tensor: hidden representation of the substituent
        """
        node_x = self.get_node_reps(data.x, data.edge_index, data.edge_attr, data.batch)
        bool_mask = torch.zeros(
            data.mask.size(), dtype=torch.bool, device=data.x.device
        )
        bool_mask[sub] = 1
        masked_batch = data.batch[bool_mask]
        bs = len(torch.unique(data.batch))
        masked_bs = len(torch.unique(masked_batch))
        assert masked_bs == 1
        unique_batch = torch.zeros(
            data.mask.size(), dtype=torch.long, device=data.x.device
        )
        if masked_batch.numel() == 0:
            print("No elements in batch")
            return torch.zeros(bs, self.num_classes).to(data.x.device)
        if self.pool == "att":
            uncommon_graph_x = self.pool_fn.masked_forward(
                node_x, bool_mask, unique_batch
            )
        if self.pool == "mean+att":
            uncommon_graph_x = self.pool_fn_ucn.masked_forward(
                node_x, bool_mask, unique_batch
            )
        else:
            uncommon_graph_x = self.pool_fn(node_x[bool_mask], batch=None)
        uncommon_pred = self.relu(self.lin1(uncommon_graph_x))
        uncommon_pred = self.lin2(uncommon_pred)
        return uncommon_pred

    def get_uncommon_graph_rep(self, data: torch.Tensor) -> torch.Tensor:
        """Gets the hidden representation of the uncommon part of a molecule as the model prediction on this subgraph.
        Args:
            data (torch.Tensor): data object containing the graph

        Returns:
            torch.Tensor: hidden representation of the uncommon part of the molecule
        """
        node_x = self.get_node_reps(data.x, data.edge_index, data.edge_attr, data.batch)
        mask = data.mask.cpu().numpy()
        bool_mask = torch.BoolTensor(np.where(mask == 0, 0, 1))
        masked_batch = data.batch[bool_mask]
        bs = len(torch.unique(data.batch))
        masked_bs = len(torch.unique(data.batch[bool_mask]))

        if masked_batch.numel() == 0:
            return torch.zeros(bs, self.num_classes).to(data.x.device)

        if self.pool == "att":
            uncommon_graph_x = self.pool_fn.masked_forward(
                node_x, bool_mask, data.batch
            )
        if self.pool == "mean+att":
            uncommon_graph_x = self.pool_fn_ucn.masked_forward(
                node_x, bool_mask, data.batch
            )
        else:
            uncommon_graph_x = self.pool_fn(node_x[bool_mask], data.batch[bool_mask])

        uncommon_pred = self.relu(self.lin1(uncommon_graph_x))
        uncommon_pred = self.lin2(uncommon_pred)

        if masked_bs < bs:
            non_zeros_mask_idx = np.intersect1d(
                torch.unique(data.batch).cpu().numpy(),
                torch.unique(data.batch[bool_mask]).cpu().numpy(),
            )
            new_emb = torch.zeros(bs, self.num_classes).to(data.x.device)
            new_emb[non_zeros_mask_idx] = uncommon_pred[non_zeros_mask_idx]
            uncommon_pred = new_emb
        return uncommon_pred

    def get_pred(self, graph_x):
        """Returns the prediction of the model on a graph embedding after the graph convolutional layers."""
        pred = self.relu(self.lin1(graph_x))
        pred = self.lin2(pred)
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)
