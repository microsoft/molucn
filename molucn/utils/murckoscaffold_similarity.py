import os
import dill
import numpy as np
import pandas as pd
from molucn.utils.utils import read_list_targets


from collections import defaultdict
from typing import Dict, List, Set, Tuple, Union
from torch_geometric.data import HeteroData

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
import numpy as np


def make_mol(s: str, keep_h: bool, add_h: bool):
    """
    Builds an RDKit molecule from a SMILES string.

    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles. This does not add hydrogens, it only keeps them if they are specified.
    :return: RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
        )
    else:
        mol = Chem.MolFromSmiles(s)
    if add_h:
        mol = Chem.AddHs(mol)
    return mol


def generate_scaffold(
    mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
    include_chirality: bool = False,
) -> str:
    """
    Computes the Bemis-Murcko scaffold for a SMILES string.

    :param mol: A SMILES or an RDKit molecule.
    :param include_chirality: Whether to include chirality in the computed scaffold..
    :return: The Bemis-Murcko scaffold for the molecule.
    """
    if isinstance(mol, str):
        mol = make_mol(mol, keep_h=False, add_h=False)
    if isinstance(mol, tuple):
        mol = mol[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality
    )

    return scaffold


def scaffold_to_smiles(
    mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
    use_indices: bool = False,
) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes the scaffold for each SMILES and returns a mapping from scaffolds to sets of smiles (or indices).

    :param mols: A list of SMILES or RDKit molecules.
    :param use_indices: Whether to map to the SMILES's index in :code:`mols` rather than
                        mapping to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all SMILES (or indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        scaffold = generate_scaffold(mol)
        if use_indices:
            scaffolds[scaffold].add(i)
        else:
            scaffolds[scaffold].add(mol)

    return scaffolds


def get_unique_list_mol(pairs_list: List[HeteroData]):
    list_mol = []
    for k in range(len(pairs_list)):
        hetero_data = pairs_list[k]
        data_i, data_j = hetero_data["data_i"], hetero_data["data_j"]
        smile_i, smile_j = data_i.smiles, data_j.smiles
        list_mol.append(smile_i)
        list_mol.append(smile_j)
    return np.unique(list_mol)


def get_murckoscaffol_sim(n_targets: int, par_dir: str):
    """
    Returns the average tanimoto similarity of all targets in list_targets as a dictionary
    """
    targets = read_list_targets(n_targets, par_dir)
    mol_sim = dict()
    for i in range(len(targets)):
        target = targets[i]
        target_path = os.path.join(par_dir, target, f"{target}_heterodata_list.pt")
        with open(target_path, "rb") as handle:
            pairs_list = dill.load(handle)
        list_compounds = get_unique_list_mol(pairs_list)
        scaffolds = scaffold_to_smiles(list_compounds)
        print(
            f"Target: {target} - Number of scaffolds: {len(scaffolds)} - Number of compounds: {len(list_compounds)}"
        )
        mol_sim[target] = len(scaffolds) / len(list_compounds)
    return mol_sim


if __name__ == "__main__":
    par_dir = "/cluster/work/zhang/kamara/molucn/data/"
    save_dir = "/cluster/home/kamara/molucn/data/"
    n_targets = 350
    mol_sim = get_murckoscaffol_sim(n_targets, par_dir)
    print(mol_sim)
    df_mol_sim = pd.DataFrame(
        {"target": mol_sim.keys(), "murcko_sim": mol_sim.values()}
    )
    df_mol_sim.to_csv(os.path.join(save_dir, f"murcko_sim_norm_{n_targets}.csv"))
