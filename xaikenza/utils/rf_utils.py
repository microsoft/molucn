from copy import deepcopy

import numpy as np
from rdkit.Chem import AllChem, DataStructs, MolFromSmiles
from tqdm import tqdm

FP_SIZE = 1024
BOND_RADIUS = 2


def gen_dummy_atoms(mol, dummy_atom_no=47):
    """
    Given a specific rdkit mol, returns a list of mols where each individual atom
    has been replaced by a dummy atom type.
    """
    mod_mols = []

    for idx_atom in range(mol.GetNumAtoms()):
        mol_cpy = deepcopy(mol)
        mol_cpy.GetAtomWithIdx(idx_atom).SetAtomicNum(dummy_atom_no)
        mod_mols.append(mol_cpy)
    return mod_mols


def featurize_ecfp4(mol, fp_size=FP_SIZE, bond_radius=BOND_RADIUS):
    """
    Gets an ECFP4 fingerprint for a specific rdkit mol. 
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=fp_size)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr



def pred_pairs_diff(pair_df, model, mol_read_f=MolFromSmiles):
    preds_diff = []

    for row in tqdm(pair_df.itertuples(), total=len(pair_df)):
        data_i, data_j = row['data_i'], row['data_j']
        sm_i, sm_j = data_i.smiles, data_j.smiles
        mol_i, mol_j = mol_read_f(sm_i), mol_read_f(sm_j)
        fp_i, fp_j = featurize_ecfp4(mol_i), featurize_ecfp4(mol_j)
        pred_i, pred_j = (
            model.predict(fp_i[np.newaxis, :]).squeeze(),
            model.predict(fp_j[np.newaxis, :]).squeeze(),
        )
        pred = pred_i - pred_j
        preds_diff.append(pred)
    return preds_diff



def color_pairs_diff(pair_df, model, diff_fun):
    """
    Uses Sheridan's (2019) method to color all pairs of molecules
    available in `pair_df`.
    """
    colors = []

    for row in tqdm(pair_df.itertuples(), total=len(pair_df)):
        data_i, data_j = row['data_i'], row['data_j']
        color_i, color_j = (
            diff_fun(data_i.smiles, model.predict),
            diff_fun(data_j.smiles, model.predict),
        )
        colors.append((color_i, color_j))
    return colors



def diff_mask(
    mol_string,
    pred_fun,
    fp_size=1024,
    bond_radius=2,
    dummy_atom_no=47,
    mol_read_f=MolFromSmiles,
):
    """
    Given a mol specified by a string (SMILES, inchi), uses Sheridan's method (2019)
    alongside an sklearn model to compute atom attribution.
    """
    mol = mol_read_f(mol_string)
    og_fp = featurize_ecfp4(mol, fp_size, bond_radius)

    og_pred = pred_fun(og_fp[np.newaxis, :]).squeeze()

    mod_mols = gen_dummy_atoms(mol, dummy_atom_no)

    mod_fps = [featurize_ecfp4(mol, fp_size, bond_radius) for mol in mod_mols]
    mod_fps = np.vstack(mod_fps)
    mod_preds = pred_fun(mod_fps).squeeze()
    return og_pred - mod_preds


def gen_masked_atom_feats(og_g):
    """ 
    Given a graph, returns a list of graphs where individual atoms
    are masked.
    """
    masked_gs = []
    for node_idx in range(og_g[0]["nodes"].shape[0]):
        g = deepcopy(og_g)
        g[0]["nodes"][node_idx] *= 0.0
        masked_gs.append(g[0])
    return masked_gs

