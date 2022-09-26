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
