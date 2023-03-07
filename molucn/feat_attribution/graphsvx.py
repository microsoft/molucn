""" explainers.py

    Define the different explainers: GraphSVX and baselines
"""
# Import packages
import random
import time
from copy import deepcopy
from itertools import combinations
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data
from molucn.feat_attribution.explainer_base import Explainer
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from sklearn.metrics import r2_score
import scipy.special


class GraphSVX(Explainer):
    def __init__(self, device: torch.device, model: torch.nn.Module):
        super(GraphSVX, self).__init__(device, model)
        self.device = device
        self.neighbours = None  #  nodes considered
        self.F = None  # number of features considered
        self.M = None  # number of features and nodes considered
        self.base_values = []

        self.model.eval()

    def explain_graph(self, graph: Data, model: torch.nn.Module = None) -> torch.Tensor:

        if model == None:
            model = self.model

        tmp_graph = graph.clone().to(self.device)
        self.graph = tmp_graph

        phi = self.gen_shap_weights()
        return phi

    ################################
    # Core function - explain
    ################################
    def gen_shap_weights(
        self,
        num_samples=10,
        info=False,
        multiclass=False,
        fullempty=None,
        S=3,
        args_feat="Expectation",
        args_coal="Smarter",
        args_g="WLS",
        regu=None,
    ):
        """Explains prediction for a graph classification task - GraphSVX method

        Args:
            node_indexes (list, optional): indexes of the nodes of interest. Defaults to [0].
            hops (int, optional): number k of k-hop neighbours to consider in the subgraph
                                                    around node_index. Defaults to 2.
            num_samples (int, optional): number of samples we want to form GraphSVX's new dataset.
                                                    Defaults to 10.
            info (bool, optional): Print information about explainer's inner workings.
                                                    And include vizualisation. Defaults to True.
            multiclass (bool, optional): extension - consider predicted class only or all classes
            fullempty (bool, optional): enforce high weight for full and empty coalitions
            S (int, optional): maximum size of coalitions that are favoured in mask generation phase
            args_feat (str, optional): way to switch off and discard node features (0 or expectation)
            args_coal (str, optional): how we sample coalitions z
            args_g (str, optional): method used to train model g on (z, f(z'))
            regu (int, optional): extension - apply regularisation to balance importance granted
                                                    to nodes vs features
            vizu (bool, optional): creates vizualisation or not

        Returns:
                [tensors]: shapley values for features/neighbours that influence node v's pred
                        and base value
        """

        # Time
        start = time.time()

        y_pred = self.model(self.graph)

        D = self.graph.x.shape[0]
        self.neighbours = list(range(D))

        # Total number of features + neighbours considered for node v
        self.F = 0
        self.M = self.F + D

        # Def range of endcases considered
        args_K = S

        # --- MASK GENERATOR ---
        z_, weights = self.mask_generation(
            num_samples, args_coal, args_K, D, info, regu
        )

        # Discard full and empty coalition if specified
        if fullempty:
            weights[(weights == 1000).nonzero()] = 0

        # --- GRAPH GENERATOR ---
        # Create dataset (z, f(GEN(z'))), stored as (z_, fz)
        # Retrieve z' from z and x_v, then compute f(z')
        fz = self.graph_classification(num_samples, D, z_, args_feat)

        # --- EXPLANATION GENERATOR ---
        # Train Surrogate Weighted Linear Regression - learns shapley values
        phi, base_value = eval("self." + args_g)(z_, weights, fz, multiclass, info)

        return phi

    ################################
    # Mask Generator
    ################################

    def mask_generation(self, num_samples, args_coal, args_K, D, info, regu):
        """Applies selected mask generator strategy
        Args:
            num_samples (int): number of samples for GraphSVX
            args_coal (str): mask generator strategy
            args_K (int): size param for indirect effect
            D (int): number of nodes considered after selection
            info (bool): print information or not
            regu (int): balances importance granted to nodes and features
        Returns:
            [tensor] (num_samples, M): dataset of samples/coalitions z'
            [tensor] (num_samples): vector of kernel weights corresponding to samples
        """
        if args_coal == "SmarterSeparate" or args_coal == "NewSmarterSeparate":
            weights = torch.zeros(num_samples, dtype=torch.float64)
            if self.F == 0 or D == 0:
                num = int(num_samples * self.F / self.M)
            elif regu != None:
                num = int(num_samples * regu)
                # num = int( num_samples * ( self.F/self.M + ((regu - 0.5)/0.5)  * (self.F/self.M) ) )
            else:
                num = int(0.5 * num_samples / 2 + 0.5 * num_samples * self.F / self.M)
            # Features only
            z_bis = eval("self." + args_coal)(num, args_K, 1)
            z_bis = z_bis[torch.randperm(z_bis.size()[0])]
            s = (z_bis != 0).sum(dim=1)
            weights[:num] = self.shapley_kernel(s, self.F)
            z_ = torch.zeros(num_samples, self.M)
            z_[:num, : self.F] = z_bis
            # Node only
            z_bis = eval("self." + args_coal)(num_samples - num, args_K, 0)
            z_bis = z_bis[torch.randperm(z_bis.size()[0])]
            s = (z_bis != 0).sum(dim=1)
            weights[num:] = self.shapley_kernel(s, D)
            z_[num:, :] = torch.ones(num_samples - num, self.M)
            z_[num:, self.F :] = z_bis

        else:
            # If we choose to sample all possible coalitions
            if args_coal == "All":
                num_samples = min(10000, 2**self.M)

            # Coalitions: sample num_samples binary vectors of dimension M
            z_ = eval("self." + args_coal)(num_samples, args_K, regu)

            # Shuffle them
            z_ = z_[torch.randperm(z_.size()[0])]

            # Compute |z| for each sample z: number of non-zero entries
            s = (z_ != 0).sum(dim=1)

            # GraphSVX Kernel: define weights associated with each sample
            weights = self.shapley_kernel(s, self.M)

        return z_, weights

    ################################
    # Graph Generator
    ################################

    def graph_classification(self, num_samples, D, z_, args_feat):
        """Construct z' from z and compute prediction f(z') for each sample z
            In fact, we build the dataset (z, f(z')), required to train the weighted linear model.
            Graph Regression task
        Args:
            Variables are defined exactly as defined in explainer function
            Note that adjacency matrices are dense (square) matrices (unlike node classification)
        Returns:
            (tensor): f(z') - probability of belonging to each target classes, for all samples z'
            Dimension (N * C) where N is num_samples and C num_classses.
            Here C = 1, because graph regression task.
        """
        # Store discarded nodes (z_j=0) for each sample z
        excluded_nei = {}
        for i in range(num_samples):
            # Excluded nodes' indexes
            nodes_id = []
            for j in range(D):
                if z_[i, self.F + j] == 0:
                    nodes_id.append(self.neighbours[j])
            excluded_nei[i] = nodes_id
            # Dico with key = num_sample id, value = excluded neighbour index

        # Init
        fz = torch.zeros(num_samples)
        adj = deepcopy(to_scipy_sparse_matrix(self.graph.edge_index).toarray())
        if args_feat == "Null":
            av_feat_values = torch.zeros(self.graph.x.shape[1])
        else:
            av_feat_values = self.graph.x.mean(dim=0)

        # Create new matrix A and X - for each sample ≈ reform z' from z
        for (key, ex_nei) in tqdm(excluded_nei.items()):

            # Change adj matrix
            A = deepcopy(adj)
            A[ex_nei, :] = 0
            A[:, ex_nei] = 0

            # Change edge attr
            removed_ei = np.array([], dtype="int64")
            for e in ex_nei:
                removed_ei = np.concatenate(
                    (
                        removed_ei,
                        np.where(
                            self.graph.edge_index.cpu().detach().numpy() == int(e)
                        )[1],
                    )
                )
            removed_ei = np.sort(np.unique(removed_ei))
            edge_attr = torch.Tensor(
                np.delete(
                    self.graph.edge_attr.cpu().detach().numpy(), removed_ei, axis=0
                )
            )

            # Also change features of excluded nodes (optional)
            X = deepcopy(self.graph.x)
            for nei in ex_nei:
                X[nei] = av_feat_values

            # Apply model on (X,A) as input.
            edge_index, edge_weights = from_scipy_sparse_matrix(csr_matrix(A))
            sample_graph = Data(
                x=X,
                edge_index=edge_index.to(self.device),
                edge_attr=edge_attr.to(self.device),
                batch=torch.zeros(X.size(0), dtype=torch.int64),
            )
            fz[key] = self.model(sample_graph.to(self.device)).item()

        return fz

    ################################
    # Feature selector
    ################################

    def Smarter(self, num_samples, args_K, *unused):
        """Smart Mask generator
        Nodes and features are considered together but separately

        Args:
            num_samples ([int]): total number of coalitions z_
            args_K: max size of coalitions favoured in sampling

        Returns:
            [tensor]: z_ in {0,1}^F x {0,1}^D (num_samples x self.M)
        """
        # Define empty and full coalitions
        z_ = torch.ones(num_samples, self.M)
        z_[1::2] = torch.zeros(num_samples // 2, self.M)
        i = 2
        k = 1
        # Loop until all samples are created
        while i < num_samples:
            # Look at each feat/nei individually if have enough sample
            # Coalitions of the form (All nodes/feat, All-1 feat/nodes) & (No nodes/feat, 1 feat/nodes)
            if i + 2 * self.M < num_samples and k == 1:
                z_[i : i + self.M, :] = torch.ones(self.M, self.M)
                z_[i : i + self.M, :].fill_diagonal_(0)
                z_[i + self.M : i + 2 * self.M, :] = torch.zeros(self.M, self.M)
                z_[i + self.M : i + 2 * self.M, :].fill_diagonal_(1)
                i += 2 * self.M
                k += 1

            else:
                # Split in two number of remaining samples
                # Half for specific coalitions with low k and rest random samples
                samp = i + 9 * (num_samples - i) // 10
                while i < samp and k <= args_K:
                    # Sample coalitions of k1 neighbours or k1 features without repet and order.
                    L = list(combinations(range(self.F), k)) + list(
                        combinations(range(self.F, self.M), k)
                    )
                    random.shuffle(L)
                    L = L[: samp + 1]

                    for j in range(len(L)):
                        # Coalitions (All nei, All-k feat) or (All feat, All-k nei)
                        z_[i, L[j]] = torch.zeros(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples - i, self.M).random_(2)
                            return z_
                        # Coalitions (No nei, k feat) or (No feat, k nei)
                        z_[i, L[j]] = torch.ones(k)
                        i += 1
                        # If limit reached, sample random coalitions
                        if i == samp:
                            z_[i:, :] = torch.empty(num_samples - i, self.M).random_(2)
                            return z_
                    k += 1

                # Sample random coalitions
                z_[i:, :] = torch.empty(num_samples - i, self.M).random_(2)
                return z_
        return z_

    ################################
    # Explanation Generator
    ################################

    def WLS(self, z_, weights, fz, multiclass, info):
        """Weighted Least Squares Method
            Estimates shapley values via explanation model

        Args:
            z_ (tensor): binary vector representing the new instance
            weights ([type]): shapley kernel weights for z
            fz ([type]): prediction f(z') where z' is a new instance - formed from z and x

        Returns:
            [tensor]: estimated coefficients of our weighted linear regression - on (z, f(z'))
            Dimension (M * num_classes)
        """
        # Add constant term
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)

        # WLS to estimate parameters
        try:
            tmp = np.linalg.inv(np.dot(np.dot(z_.T, np.diag(weights)), z_))
        except np.linalg.LinAlgError:  # matrix not invertible
            if info:
                print("WLS: Matrix not invertible")
            tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
            tmp = np.linalg.inv(
                tmp + np.diag(10 ** (-5) * np.random.randn(tmp.shape[1]))
            )

        phi = np.dot(tmp, np.dot(np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))

        # Test accuracy
        y_pred = z_.detach().numpy() @ phi
        if info:
            print("r2: ", r2_score(fz, y_pred))
            # print('weighted r2: ', r2_score(fz, y_pred, weights))

        return phi[:-1], phi[-1]

    ################################
    # GraphSVX kernel
    ################################

    def shapley_kernel(self, s, M):
        """Computes a weight for each newly created sample
        Args:
            s (tensor): contains dimension of z for all instances
                (number of features + neighbours included)
            M (tensor): total number of features/nodes in dataset
        Returns:
                [tensor]: shapley kernel value for each sample
        """
        shapley_kernel = []

        for i in range(s.shape[0]):
            a = s[i].item()
            if a == 0 or a == M:
                # Enforce high weight on full/empty coalitions
                shapley_kernel.append(1000)
            elif scipy.special.binom(M, a) == float("+inf"):
                # Treat specific case - impossible computation
                shapley_kernel.append(1 / (M**2))
            else:
                shapley_kernel.append(
                    (M - 1) / (scipy.special.binom(M, a) * a * (M - a))
                )

        shapley_kernel = np.array(shapley_kernel)
        shapley_kernel = np.where(shapley_kernel < 1.0e-40, 1.0e-40, shapley_kernel)
        return torch.tensor(shapley_kernel)
