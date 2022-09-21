import json
import os
import shutil

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromInchi, MolFromSmiles, MolToInchi, MolToSmiles
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.MolStandardize.rdMolStandardize import Cleanup

MW_THR = 800
MIN_P_DIFF = 1.0
SYMBOL_EXCLUDE = set(["=", ">", "<"])


def convert_to_mask(dict):
    """Converts a dictionary of atom indices to a mask."""
    arr = np.array(list(dict.values()))
    return arr


def ensure_readability(strings, read_f):
    """Ensures that all strings can be read by a given function."""
    valid_idx = []
    for idx, string in enumerate(strings):
        mol = read_f(string)
        if mol is not None:
            valid_idx.append(idx)
    return valid_idx


def translate(strings_, fromfun, tofun):
    """Translat a list of strings from one format to another."""
    trans = []
    idx_success = []
    for idx, s in enumerate(strings_):
        try:
            mol = fromfun(s)
            if mol is not None:
                trans.append(tofun(mol))
                idx_success.append(idx)
        except:
            continue
    return trans, idx_success


def process_tsv(
    tsv_file,
    ligcol="Ligand SMILES",
    affcol="IC50 (nM)",
    use_log=True,
    min_p_diff=MIN_P_DIFF,
    mw_thr=MW_THR,
):

    """Extracts the SMILES and activities from a TSV file."""

    df = pd.read_csv(tsv_file, sep="\t")
    df = df.loc[df[affcol].notna()]

    if df[affcol].dtype == np.dtype("O"):
        df = df[~df[affcol].str.contains("|".join(SYMBOL_EXCLUDE))]

    ok_read_idx = ensure_readability(df[ligcol].to_list(), MolFromSmiles)

    df = df.iloc[ok_read_idx]
    values = df[affcol].values.astype(np.float32)

    inchis = []
    idx_suc = []
    for idx, sm in enumerate(df[ligcol]):
        mol = Cleanup(MolFromSmiles(sm))
        if mol is not None:
            inchis.append(MolToInchi(mol))
            idx_suc.append(idx)

    values = values[idx_suc]

    df_clean = pd.DataFrame({"inchis": inchis, "values": values})
    df_clean = df_clean.groupby(["inchis"], as_index=False)["values"].mean()

    smiles, idx_trans = translate(df_clean["inchis"], MolFromInchi, MolToSmiles)
    values = df_clean["values"].iloc[idx_trans].values
    smiles = np.array(smiles)

    mws = np.array([MolWt(MolFromSmiles(lig)) for lig in smiles])
    idx_below_mw = np.argwhere(mws <= mw_thr).flatten()

    smiles = smiles[idx_below_mw]
    values = values[idx_below_mw]

    if use_log:
        values = -np.log10(1e-9 * values)

    df = pd.DataFrame(columns=["smiles", "pchemvalues"])
    df["smiles"] = smiles
    df["pchemvalues"] = values

    return df


def gen_data(tsv_file, pairs_file, colors_file):
    """Generates the data for the dataset. 
    Extract smiles and activities from the TSV file, 
    the pairs from the pairs file 
    and the masks from the colors file."""
    df_pairs = pd.read_csv(pairs_file)
    colors = pd.read_pickle(colors_file)
    df_activities = process_tsv(tsv_file)

    pairs_list = []
    for k in range(len(colors)):
        if colors[k][0] is not None:
            smiles_i = df_pairs.loc[k, "smiles_i"]
            smiles_j = df_pairs.loc[k, "smiles_j"]

            rowi = df_activities[df_activities["smiles"] == smiles_i]
            a_i = rowi["pchemvalues"].item()

            rowj = df_activities[df_activities["smiles"] == smiles_j]
            a_j = rowj["pchemvalues"].item()

            mask_i = convert_to_mask(colors[k][0][0]).tolist()
            mask_j = convert_to_mask(colors[k][0][1]).tolist()

            mcs = np.ones(10)
            for p in range(10):
                if colors[k][p] is None:
                    mcs[p] = 0

            if (
                len(np.where(np.array(mask_i) == 0)[0]) > 0
                and len(np.where(np.array(mask_j) == 0)[0]) > 0
            ):
                new_row = {
                    "smiles_i": smiles_i,
                    "smiles_j": smiles_j,
                    "a_i": a_i,
                    "a_j": a_j,
                    "mask_i": mask_i,
                    "mask_j": mask_j,
                    "mcs": mcs.tolist(),
                }
                pairs_list.append(new_row)
    return pairs_list


if __name__ == "__main__":
    dir_benchmark = "xaibench/benchmark"
    dir_data = "xaibench/data/validation_sets"

    os.makedirs("data", exist_ok=True)
    os.makedirs("xaibench/processed_data", exist_ok=True)
    os.makedirs("data/selected_processed_data", exist_ok=True)

    for folder in os.listdir(dir_benchmark):
        tsv_file = os.path.join(dir_data, f"{folder}/{folder}.tsv")
        pairs_file = os.path.join(dir_benchmark, f"{folder}/pairs.csv")
        colors_file = os.path.join(dir_benchmark, f"{folder}/colors.pt")
        if (
            os.path.exists(tsv_file)
            and os.path.exists(pairs_file)
            and os.path.exists(colors_file)
        ):
            print("Processing folder {}".format(folder))
            pairs_list = gen_data(tsv_file, pairs_file, colors_file)
            n_pairs = len(pairs_list)
            file_name = f"{folder}_processed_data_{n_pairs}.json"
            dir_data_save = os.path.join("xaibench/processed_data", file_name)
            with open(dir_data_save, "w") as fp:
                for pair in pairs_list:
                    fp.write(json.dumps(pair))
                    fp.write("\n")

            shutil.copy(
                dir_data_save, os.path.join("data/selected_processed_data", file_name)
            )
