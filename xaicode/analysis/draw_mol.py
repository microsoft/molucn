#%%
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
size = (120, 120)  

m = Chem.MolFromSmiles('c1ccccc1')
fig = Draw.MolToMPL(m, size=size)
# %%
import os
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 600


from rdkit.Chem.rdchem import Mol
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
par_dir = "/home/t-kenzaamara/molucn"

def draw_mol2pdf(mol: Mol, filename: str= "draw.pdf", path: Union[str, os.PathLike]= "./"):
    cwd = Path.cwd()
    os.chdir(path)
    temp_svg_file = filename.split(".")[0]+".svg"
    Draw.MolToFile(mol, imageType="svg", filename=temp_svg_file)
    drawing = svg2rlg(temp_svg_file)
    renderPDF.drawToFile(drawing, filename)
    os.remove(temp_svg_file)
    os.chdir(cwd)
    
# %%
m1 = Chem.MolFromSmiles('O=C(O)c1c(O)c(-c2ccc(O)cc2)nc2ccc(F)cc12')
m2 = Chem.MolFromSmiles('Cc1cccc2nc(-c3ccc(Cl)cc3)c(O)c(C(=O)O)c12')

draw_mol2pdf(m1, filename='figures/draw_m1.pdf', path = par_dir)
# %%
canvas = Draw.rdMolDraw2D.MolDraw2DCairo(*(600, 600))
canvas.drawOptions().setAtomPalette({-1:(0,0,0)})
Draw.rdMolDraw2D.PrepareAndDrawMolecule(canvas, m1)
canvas.WriteDrawingText(os.path.join(par_dir, 'figures/draw_m1.png'))
# %%
canvas = Draw.rdMolDraw2D.MolDraw2DCairo(*(600, 600))
canvas.drawOptions().setAtomPalette({-1:(0,0,0)})
Draw.rdMolDraw2D.PrepareAndDrawMolecule(canvas, m2)
canvas.WriteDrawingText(os.path.join(par_dir, 'figures/draw_m2.png'))
# %%
template = Chem.MolFromSmiles('O=C(O)c1c(O)c(-c2ccccc2)nc2ccccc12')
AllChem.Compute2DCoords(template)
smile1 = 'O=C(O)c1c(O)c(-c2ccc(O)cc2)nc2ccc(F)cc12'
smile2 = 'Cc1cccc2nc(-c3ccc(Cl)cc3)c(O)c(C(=O)O)c12'
m1 = Chem.MolFromSmiles(smile1)
m2 = Chem.MolFromSmiles(smile2)
M1 = AllChem.GenerateDepictionMatching2DStructure(m1, template)
M2 = AllChem.GenerateDepictionMatching2DStructure(m2, template)

Draw.MolToFile(m1, os.path.join(par_dir, 'figures/draw_m1.png'))
Draw.MolToFile(m2, os.path.join(par_dir, 'figures/draw_m2.png'))
# %%
import os
import os.path as osp
import dill
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
#matplotlib.rc('font', family='sans-serif') 
matplotlib.rc('text', usetex='True')
from collections import Counter
sns.set_theme(style="whitegrid")
matplotlib.rcParams["figure.facecolor"] = "white"
par_dir = '/home/t-kenzaamara/molucn'

#%%
train_data_path = os.path.join(par_dir, 'data/1D3G-BRE/1D3G-BRE_seed_1337_train.pt')
with open(train_data_path, "rb") as fp:
    train_pairs = dill.load(fp)
train_pairs

#%%
train_data_path = os.path.join(par_dir, 'colors/gradinput/1D3G-BRE/1D3G-BRE_seed_1337_nn_MSE+UCN_mean_1.0_gradinput_train.pt')
with open(train_data_path, "rb") as fp:
    train_colors = dill.load(fp)
train_colors
# %%
pair = train_pairs[3]
c_i = pair.data_i
c_j = pair.data_j
c_i.x.size()
#%%
colors = train_colors[3]
colors_i = colors[0]
colors_j = colors[1]
len(colors_i)
# %%
mi = Chem.MolFromSmiles(c_i.smiles)
mj = Chem.MolFromSmiles(c_j.smiles)
mi

# %%


import os
import matplotlib as mpl
import matplotlib.cm as cm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib import pyplot as plt
import matplotlib.colors as colors


MAXIMUM_NUMBER_OF_RINGS = 9
NON_ATOM_CHARACTERS = set(
    [
        str(index)
        for index in range(1, MAXIMUM_NUMBER_OF_RINGS)
    ] + ['(', ')', '#', '=']
)
CMAP = cm.coolwarm
COLOR_NORMALIZERS = {
    'linear': mpl.colors.Normalize,
    'logarithmic': mpl.colors.LogNorm
}


def _get_index_and_colors(values, objects, predicate, color_mapper):
    """
    Get index and RGB colors from a color map using a rule.
    
    The predicate acts on a tuple of (value, object).
    """
    indices = []
    colors = {}
    for index, value in enumerate(
        map(
            lambda t: t[0],
            filter(
                lambda t: predicate(t),
                zip(values, objects)
            )
        )
    ):
        indices.append(index)
        colors[index] = color_mapper.to_rgba(value)
    return indices, colors

def get_color_mapper(values):
    normalize = colors.TwoSlopeNorm(vmin=min(values), vcenter=0, vmax=max(values))
    color_mapper = cm.ScalarMappable(
        norm=normalize, cmap=CMAP
    )
    return color_mapper

def smiles_attention_to_svg(
    values, atoms_and_bonds, molecule, color_mapper, file_name,
    atom_radii=0.5, svg_width=400, svg_height=400,
    color_normalization='linear'
):
    """
    Generate an svg of the molecule highlighiting atoms and bonds.
    
    Args:
        - values: values used to color atoms and bonds.
        - atoms_and_bonds: atoms and bonds obtained tokenizing a SMILES
            representation (len(atoms_and_bonds) == len(values)).
        - molecule: RDKit molecule generated using Chem.MolFromSmiles.
        - atom_radii: size of the atoms.
        - svg_width: width of the svg figure.
        - svg_height: height of the svg figure.
        - color_normalization: color normalizer ('linear' or 'logarithmic').
    Returns:
        the svg as a string.
    """
    # define a color map
    #normalize = COLOR_NORMALIZERS.get(
        #color_normalization, mpl.colors.LogNorm
    #)(
        #vmin=min(values),
        #vmax=max(values)
    #)
    # get atom colors
    _, highlight_atom_colors = _get_index_and_colors(
        values, atoms_and_bonds,
        lambda t: t[1] not in NON_ATOM_CHARACTERS,
        color_mapper
    )
    highlight_atoms = np.array(atoms_and_bonds).tolist()
    # add coordinates
    Chem.rdDepictor.Compute2DCoords(molecule)
    # draw the molecule
    drawer = rdMolDraw2D.MolDraw2DCairo(svg_width, svg_height)
    drawer.DrawMolecule(
        molecule,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_atom_colors,
        highlightAtomRadii={
            index: atom_radii
            for index in highlight_atoms
        }
    )
    drawer.FinishDrawing()
    # return the drawn molecule
    p = drawer.GetDrawingText()
    drawer.WriteDrawingText(os.path.join(par_dir, 'figures', file_name))

    import IPython.display
    IPython.display.Image(p)
    return #drawer.GetDrawingText().replace('\n', ' ')

#idx = range(len(c_i.x))
#smiles_attention_to_svg(colors_i, idx, mi, file_name = 'draw_mi.png')
#smiles_attention_to_svg(colors_j, range(len(c_j.x)), mj, file_name = 'draw_mj.png')
# %%
import torch
np.where(c_i.mask == 1)[0]
#%%
bool_mask = torch.BoolTensor(np.where(c_i.mask == 0, 0, 1))
colors_i[bool_mask]

#%%
bool_mask = torch.BoolTensor(np.where(c_j.mask == 0, 0, 1))
colors_j[bool_mask]
# %%
color_mapper = get_color_mapper([*colors_i, *colors_j])
smiles_attention_to_svg(colors_i[torch.BoolTensor(np.where(c_i.mask == 0, 0, 1))], np.where(c_i.mask != 0)[0], mi, color_mapper, file_name = 'draw_mi.png')
smiles_attention_to_svg(colors_j[torch.BoolTensor(np.where(c_j.mask == 0, 0, 1))], np.where(c_j.mask != 0)[0], mj, color_mapper, file_name = 'draw_mj.png')
# %%
c_j.mask
# %%
for atom in mi.GetAtoms():
    print('atom idx: ', atom.GetIdx())
    print(c_i.x[atom.GetIdx(), :])
# %%
color_mapper = get_color_mapper([*colors_i, *colors_j])
# %%
print(c_i.y, c_j.y)
print(colors_i[torch.BoolTensor(np.where(c_i.mask == 0, 0, 1))], colors_j[torch.BoolTensor(np.where(c_j.mask == 0, 0, 1))])
# %%
color_mapper
# %%
