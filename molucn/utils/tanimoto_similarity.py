import os
import dill
import json
from typing import List
from torch_geometric.data import HeteroData
from molucn.utils.utils import read_list_targets
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import numpy as np
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
import seaborn as sns

from rdkit.DataManip.Metric import GetTanimotoDistMat
from rdkit.DataManip.Metric import GetTanimotoSimMat

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit.DataStructs import TanimotoSimilarity


from rdkit import rdBase
from rdkit.Chem import RDConfig

print(rdBase.rdkitVersion)


def tanimoto_sim(mol_i, mol_j, radius=2):
    """
    Returns tanimoto similarity for a pair of mols
    """
    fp_i, fp_j = (
        GetMorganFingerprint(mol_i, radius),
        GetMorganFingerprint(mol_j, radius),
    )
    return TanimotoSimilarity(fp_i, fp_j)


def get_target_avgTanimotosimilarity(pairs_list: List[HeteroData]):
    Tansim = []
    for k in range(len(pairs_list)):
        hetero_data = pairs_list[k]
        data_i, data_j = hetero_data["data_i"], hetero_data["data_j"]
        smile_i, smile_j = data_i.smiles, data_j.smiles
        Tansim.append(
            tanimoto_sim(
                Chem.MolFromSmiles(smile_i), Chem.MolFromSmiles(smile_j), radius=2
            )
        )
    return np.mean(Tansim)


def get_tanimoto_sim(n_targets: int, par_dir: str):
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
        mol_sim[target] = get_target_avgTanimotosimilarity(pairs_list)
    return mol_sim


if __name__ == "__main__":
    par_dir = "/cluster/work/zhang/kamara/molucn/data/"
    save_dir = "/cluster/home/kamara/molucn/data/"
    n_targets = 350
    mol_sim = get_tanimoto_sim(n_targets, par_dir)
    print(mol_sim)
    df_mol_sim = pd.DataFrame({"target": mol_sim.keys(), "tan_sim": mol_sim.values()})
    df_mol_sim.to_csv(os.path.join(save_dir, f"tanimoto_sim_{n_targets}.csv"))
