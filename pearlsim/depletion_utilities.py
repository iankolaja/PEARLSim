import numpy as np
import pandas as pd
#import openmc
openmc = None

def nuclide_labels_serpent_to_openMC(dataframe):
    '''
    Convert the isotope headers of a dataframe from ZAId integer format into symbol-mass string format.
    '''
    z_map = {'Ac': 89, 'Ag': 47, 'Al': 13, 'Am': 95, 'Ar': 18, 'As': 33, 'At': 85, 'Au': 79, 'B': 5, 'Ba': 56, 'Be': 4,
             'Bh': 107, 'Bi': 83, 'Bk': 97, 'Br': 35, 'C': 6, 'Ca': 20, 'Cd': 48, 'Ce': 58,
             'Cf': 98, 'Cl': 17, 'Cm': 96, 'Co': 27, 'Cr': 24, 'Cs': 55, 'Cu': 29, 'Ds': 110, 'Db': 105, 'Dy': 66, 'Er': 68,
             'Es': 99, 'Eu': 63, 'F': 9, 'Fe': 26, 'Fm': 100, 'Fr': 87, 'Ga': 31, 'Gd':
                 64, 'Ge': 32, 'H': 1, 'He': 2, 'Hf': 72, 'Hg': 80, 'Ho': 67, 'Hs': 108, 'I': 53, 'In': 49, 'Ir': 77,
             'K': 19, 'Kr': 36, 'La': 57, 'Li': 3, 'Lr': 103, 'Lu': 71, 'Md': 101, 'Mg': 12, 'Mn':
                 25, 'Mo': 42, 'Mt': 109, 'N': 7, 'Na': 11, 'Nb': 41, 'Nd': 60, 'Ne': 10, 'Ni': 28, 'No': 102, 'Np': 93,
             'O': 8, 'Os': 76, 'P': 15, 'Pa': 91, 'Pb': 82, 'Pd': 46, 'Pm': 61, 'Po': 84, 'Pr':
                 59, 'Pt': 78, 'Pu': 94, 'Ra': 88, 'Rb': 37, 'Re': 75, 'Rf': 104, 'Rg': 111, 'Rh': 45, 'Rn': 86, 'Ru': 44,
             'S': 16, 'Sb': 51, 'Sc': 21, 'Se': 34, 'Sg': 106, 'Si': 14, 'Sm': 62, 'Sn': 50,
             'Sr': 38, 'Ta': 73, 'Tb': 65, 'Tc': 43, 'Te': 52, 'Th': 90, 'Ti': 22, 'Tl': 81, 'Tm': 69, 'U': 92, 'V': 23,
             'W': 74, 'Xe': 54, 'Y': 39, 'Yb': 70, 'Zn': 30, 'Zr': 40}
    z_map_inv = {v: k for k, v in z_map.items()}
    rename_map = {}

    for raw_column in dataframe.columns:
        column = str(raw_column)
        mt = ""
        if "-" in column:
            column, mt = column.split("-")
            mt = f"-{mt}"
        column = column.replace("<lib>","0")
        if not column.isnumeric():
            continue
        iso_num = int(column[-1])
        a = int(column[-4:-1])
        z = int(column[:-4])
        renamed = f"{z_map_inv[z]}{a}"
        if a > 300:
            a -= 100
            iso_num
        if iso_num > 0:
            renamed += f"_m{iso_num}"
        renamed += mt
        rename_map[raw_column] = renamed
    renamed_dataframe = dataframe.rename(columns=rename_map)

    return renamed_dataframe, rename_map

def create_MicroXS_batch(xs_df, chain, nuclide_labels, replace_keys={}, drop_columns = []):
    reaction_index_key = {"102":0, "16":1, "107":2, "103":3, "17":4, "37":5, "18":6}
    data = np.zeros([len(nuclide_labels), 7, len(xs_df)])
    for raw_column in xs_df.columns:
        nuclide, mt = raw_column.split("-")
        try:
            nuclide_index = nuclide_labels.index(nuclide)
            reaction_index = reaction_index_key[mt]
            data[nuclide_index, reaction_index, :] = xs_df[raw_column]
        except:
            print(f"Skipping {nuclide}...")
    microXS_list = []
    for index in range(len(xs_df)):    
        microXS = openmc.deplete.MicroXS(data[:,:,index], nuclide_labels, chain.reactions)
        microXS_list += [microXS]
    return microXS_list
