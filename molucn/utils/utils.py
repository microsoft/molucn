import fnmatch
import os
import random
from typing import List

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx


def get_mcs(pairs_list: List[HeteroData]) -> np.ndarray:
    MCS = []
    for hetero_data in pairs_list:
        MCS.append(np.array(hetero_data["mcs"]).astype(dtype=bool))
    return np.array(MCS)


def make_dir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)
        print(f"{path} created")
    else:
        print(f"{path} already exists")


def avg(*args):
    """Average of a list of values, ignoring -1.0"""
    return tuple(map(lambda x: np.mean(x[x >= 0.0]), args))


def std(*args):
    """Standard deviation of a list of values, ignoring -1.0"""
    return tuple(map(lambda x: np.std(x[x >= 0.0]), args))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_list_targets(n_targets):
    """Read the .txt file containing the list of targets with n_targets protein targets"""
    list_targets = []
    list_file = "list_targets_{}.txt".format(n_targets)
    if os.path.exists(list_file):
        with open(list_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                list_targets.append(line.strip())
    return list_targets


def get_list_targets(dir="data/"):
    """Get the list of targets with more than 50 pairs of ligands in their training set"""
    list_targets = []
    for target in os.listdir(dir):
        if len(target) == 8:
            count = len(fnmatch.filter(os.listdir(os.path.join(dir, target)), "*.*"))
            if count > 2:
                list_targets.append(target)
    list_file = "list_targets_{}.txt".format(len(list_targets))
    if os.path.exists(list_file) == False:
        with open(list_file, "w") as f:
            for item in list_targets:
                f.write("%s\n" % item)
    return list_targets


def get_common_nodes(mask_i, mask_j):
    """Get the common nodes between two graphs and their mapping to the same indexing."""

    idx_common_i = np.where(np.array(mask_i) == 0)[0]
    idx_common_j = np.where(np.array(mask_j) == 0)[0]
    assert len(idx_common_i) == len(idx_common_j)

    n_cmn = len(idx_common_i)

    map_i = dict({idx_common_i[i]: i for i in range(n_cmn)})
    map_j = dict({idx_common_j[i]: i for i in range(n_cmn)})
    return idx_common_i, idx_common_j, map_i, map_j


def get_substituents(data, idx_common, map):
    """Get the substituents and the active sites of the scaffold."""
    G = to_networkx(data)
    H = G.subgraph(idx_common)
    G.remove_edges_from(list(H.edges()))
    G.remove_nodes_from(list(nx.isolates(G)))
    active_sites = np.array(
        [map[i] for i in np.intersect1d(np.array(G.nodes()), idx_common)]
    ).flatten()
    substituents = [
        sorted(sub) for sub in sorted(nx.connected_components(G.to_undirected()))
    ]
    return substituents, active_sites


def get_positions(subs, idx_common, map):
    """Map active sites on the scaffold to the index of the substituent attached to it."""
    pos_sub = {i: -1 for i in range(len(idx_common))}
    for k, sub in enumerate(subs):
        active_sites = np.intersect1d(sub, idx_common)
        if len(active_sites) > 1:
            # print('Warning: multiple active sites for the same substituent.')
            for pos in active_sites:
                pos_sub[map[pos]] = k
        elif len(active_sites) == 1:
            pos_sub[map[active_sites[0]]] = k
    return pos_sub


def get_substituent_info(site, data, map):
    """Get the batch and activity of the substituent attached to the active site."""
    inverse_map = dict([(val, key) for key, val in map.items()])
    attach_node = inverse_map[site]
    batch = data.batch[attach_node]
    activity = data.y[batch]
    return batch, activity