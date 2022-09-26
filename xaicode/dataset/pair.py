import ast
import os
import os.path as osp
from collections.abc import Sequence
from typing import List, Tuple, Union

import dill
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch_geometric.data import HeteroData
from xaicode.dataset.featurization import (
    create_pytorch_geometric_data_list_from_smiles_and_labels,
)
from xaicode.utils.parser_utils import overall_parser

IndexType = Union[slice, Tensor, np.ndarray, Sequence]


def get_num_features(hetero_data: HeteroData) -> Tuple[int]:
    """Returns the number of node features and edge features in a HeteroData object.

    Args:
        hetero_data (HeteroData): 'HeteroData' object containing the data.

    Returns:
        int: Number of node features.
        int: Number of edge features.
    """
    return hetero_data["data_i"].x.shape[1], hetero_data["data_i"].edge_attr.shape[1]


def convert_to_dicts(list_objects: List[object]) -> List[dict]:
    """Converts a list of objects to a list of dictionaries.

    Args:
        list_objects (List[object]): List of objects.

    Returns:
        List[dict]: List of dictionaries.
    """
    list_dicts = []
    for obj in list_objects:
        list_dicts.append(obj.__dict__)
    return np.array(list_dicts)


def create_heterodata(input: dict) -> HeteroData:
    """Converts a pair of molecules with smiles, masks, activities, and mcs to a HeteroData object.

    Args:
        input (dict): Dictionary containing the data about a pair of compounds (smiles, masks, activities, mcs).

    Returns:
        HeteroData: HeteroData object containing the data about the pair of compounds.
    """
    hetero_data = HeteroData()

    data_i = create_pytorch_geometric_data_list_from_smiles_and_labels(
        input["smiles_i"], input["a_i"]
    )
    data_i.mask = torch.Tensor(input["mask_i"])
    data_i.smiles = input["smiles_i"]
    hetero_data["data_i"] = data_i

    data_j = create_pytorch_geometric_data_list_from_smiles_and_labels(
        input["smiles_j"], input["a_j"]
    )
    data_j.mask = torch.Tensor(input["mask_j"])
    data_j.smiles = input["smiles_j"]
    hetero_data["data_j"] = data_j
    hetero_data["mcs"] = input["mcs"]
    return hetero_data


def rebalance_pairs(
    train_pairs: List[HeteroData], test_pairs: List[HeteroData], test_set_size=0.2
) -> Tuple[List[HeteroData], List[HeteroData]]:
    """Rebalances the pairs in the training and test sets to a 0.8/0.2 ratio.

    Args:
        train_pairs (List[HeteroData]): training pairs
        test_pairs (List[HeteroData]): testing pairs
        test_set_size (float, optional): ratio of pairs in the test set. Defaults to 0.2.
    """
    n = len(train_pairs) / (1 - test_set_size)
    if len(test_pairs) > test_set_size * n:
        test_pairs = test_pairs[: int(test_set_size * n)]
    else:
        n = len(test_pairs) / test_set_size
        train_pairs = train_pairs[: int(n * (1 - test_set_size))]
    return train_pairs, test_pairs


def train_test_split_pairs(
    pairs_list: List[HeteroData], ligands_list: List[str], test_set_size: float, seed=42
) -> Tuple[List[HeteroData], List[HeteroData]]:
    """Split the ligands into training and test sets, construct training and test sets of pairs and rebalance them.

    Args:
        pairs_list (List[HeteroData]): list of pairs of ligands
        ligands_list (List[str]): list of the individual ligands present in the pairs
        test_set_size (float): Ratio of pairs in the test set
        seed (int, optional): Defaults to 42.

    Returns:
        List[HeteroData]: rebalanced training pairs
        List[HeteroData]: rebalanced test pairs
    """
    train_ligands, test_ligands = train_test_split(
        ligands_list, random_state=seed, test_size=test_set_size
    )
    train_pairs, test_pairs = [], []
    for pair in pairs_list:
        if (pair["data_i"].smiles in train_ligands) and (
            pair["data_i"].smiles in train_ligands
        ):
            train_pairs.append(pair)
        elif (pair["data_i"].smiles in test_ligands) and (
            pair["data_i"].smiles in test_ligands
        ):
            test_pairs.append(pair)
    return rebalance_pairs(train_pairs, test_pairs, test_set_size)


def create_ligands_list(pairs_list: List[HeteroData]) -> List[str]:
    """Creates a list of ligands present in the pairs.

    Args:
        pairs_list (List[HeteroData]): list of pairs of ligands
    Returns:
        List[str]: list of ligand smiles
    """
    ligands_list = []
    for pair in pairs_list:
        ligands_list.append(pair["data_i"].smiles)
        ligands_list.append(pair["data_j"].smiles)
    return np.unique(ligands_list)


def get_list_targets(data_ori_path="data/selected_processed_data"):
    """Returns the list of targets present in the dataset.

    Args:
        data_ori_path (str, optional): directory with the processed data of protein targets. Defaults to "data/selected_processed_data".

    Returns:
        List[str]: list of targets present in the processed data folder
    """
    LIST_TARGETS = []
    for folder in os.listdir(data_ori_path):
        LIST_TARGETS.append(folder[:8])
    return LIST_TARGETS


if __name__ == "__main__":

    data_ori_path = "data/selected_processed_data"

    args = overall_parser().parse_args()
    os.makedirs(args.data_path, exist_ok=True)

    for folder in os.listdir(data_ori_path):
        print(folder)
        target = folder[:8]

        dir_target = osp.join(args.data_path, target)
        os.makedirs(dir_target, exist_ok=True)

        pairs_list = []
        f = open(
            osp.join(data_ori_path, folder),
        )
        Lines = f.readlines()
        for line in Lines:
            input = ast.literal_eval(line)
            pairs_list.append(create_heterodata(input))

        with open(osp.join(dir_target, f"{target}_heterodata_list.pt"), "wb") as fp:
            dill.dump(pairs_list, fp)

        ligands_list = create_ligands_list(pairs_list)
        train_pairs, test_pairs = train_test_split_pairs(
            pairs_list, ligands_list, test_set_size=args.test_set_size, seed=args.seed
        )

        # Ligands in testing set are NOT in training set!
        if len(train_pairs) > 50:
            with open(
                osp.join(dir_target, f"{target}_seed_{args.seed}_train.pt"), "wb"
            ) as handle:
                dill.dump(train_pairs, handle)

            with open(
                osp.join(dir_target, f"{target}_seed_{args.seed}_test.pt"), "wb"
            ) as handle:
                dill.dump(test_pairs, handle)

            n_compounds = len(np.unique(ligands_list))
            n_pairs_init = len(pairs_list)
            n_pairs = len(train_pairs) + len(test_pairs)
            n_pairs_train = len(train_pairs)
            n_pairs_test = len(test_pairs)

            df_stats = pd.DataFrame(
                {
                    "n_compounds": [n_compounds],
                    "n_pairs_init": [n_pairs_init],
                    "n_pairs": [n_pairs],
                    "n_pairs_train": [n_pairs_train],
                    "n_pairs_test": [n_pairs_test],
                }
            )
            df_stats.to_csv(
                osp.join(dir_target, f"{target}_seed_{args.seed}_stats.csv"),
                index=False,
            )

            with open(
                osp.join(dir_target, f"{target}_seed_{args.seed}_info.txt"), "w"
            ) as handle:
                handle.write(f"Initial number of compounds: {n_compounds}\n")
                handle.write(f"Initial number of pairs: {n_pairs_init}\n")
                handle.write(f"Number of pairs after train/test split: {n_pairs}\n")
                handle.write(f"Size of training set: {n_pairs_train}\n")
                handle.write(f"Size of testing set: {n_pairs_test}\n")
