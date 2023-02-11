from typing import List
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data

PERMITTED_LIST_OF_ATOMS = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Br",
    "Na",
    "Ca",
    "I",
    "B",
    "H",
    "*",
]


def one_hot_encoding(x: str, permitted_list: List[str]) -> List[int]:
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """

    if x not in permitted_list:
        x = permitted_list[-1]

    binary_encoding = [
        int(boolean_value)
        for boolean_value in list(map(lambda s: x == s, permitted_list))
    ]

    return binary_encoding


def get_atom_features(
    atom: Atom, use_chirality: bool = True, hydrogens_implicit: bool = True
) -> List[float]:
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms

    if hydrogens_implicit == False:
        PERMITTED_LIST_OF_ATOMS = ["H"] + PERMITTED_LIST_OF_ATOMS

    # compute atom features

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), PERMITTED_LIST_OF_ATOMS)

    n_heavy_neighbors_enc = one_hot_encoding(
        int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"]
    )

    formal_charge_enc = one_hot_encoding(
        int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
    )

    hybridisation_type_enc = one_hot_encoding(
        str(atom.GetHybridization()),
        ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"],
    )

    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]

    vdw_radius_scaled = [
        float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)
    ]

    covalent_radius_scaled = [
        float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)
    ]

    atom_feature_vector = (
        atom_type_enc
        + n_heavy_neighbors_enc
        + formal_charge_enc
        + hybridisation_type_enc
        + is_in_a_ring_enc
        + is_aromatic_enc
        + atomic_mass_scaled
        + vdw_radius_scaled
        + covalent_radius_scaled
    )

    if use_chirality:
        chirality_type_enc = one_hot_encoding(
            str(atom.GetChiralTag()),
            [
                "CHI_UNSPECIFIED",
                "CHI_TETRAHEDRAL_CW",
                "CHI_TETRAHEDRAL_CCW",
                "CHI_OTHER",
            ],
        )
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(
            int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"]
        )
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond: Bond, use_stereochemistry: bool = True) -> np.ndarray:
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]

    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)

    bond_is_conj_enc = [int(bond.GetIsConjugated())]

    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(
            str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"]
        )
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def create_pytorch_geometric_data_list_from_smiles_and_labels(
    smiles: str, y_val: float
) -> Data:
    """Converts a list of SMILES strings and a list of labels into a list of Pytorch Geometric Data objects.

    Args:
        smiles (str): SMILES of mol
        y_val (float): activity value

    Returns:
        Data: molecule as Pytorch Geometric Data object
    """
    # convert SMILES to PyG Data object
    mol = Chem.MolFromSmiles(smiles)

    # get feature dimensions
    n_nodes = mol.GetNumAtoms()
    n_edges = 2 * mol.GetNumBonds()
    unrelated_smiles = "O=O"
    unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
    n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
    n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

    # construct node feature matrix of shape (n_nodes, n_node_features)
    node_feat = np.zeros((n_nodes, n_node_features))

    for atom in mol.GetAtoms():
        node_feat[atom.GetIdx(), :] = get_atom_features(atom)

    node_feat = torch.tensor(node_feat, dtype=torch.float)

    # construct edge index array of shape (2, n_edges)
    (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
    torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
    torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([torch_rows, torch_cols], dim=0)

    # construct edge feature array EF of shape (n_edges, n_edge_features)
    edge_feat = np.zeros((n_edges, n_edge_features))

    for (k, (i, j)) in enumerate(zip(rows, cols)):

        edge_feat[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

    edge_feat = torch.tensor(edge_feat, dtype=torch.float)

    # construct label tensor
    y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)

    # construct Pytorch Geometric data object and append to data list
    data = Data(x=node_feat, edge_index=edge_index, edge_attr=edge_feat, y=y_tensor)
    return data
