import time
import numpy as np
import os
import re
import struct
import pandas as pd
from pearlsim.material import Material
from scipy.stats import norm
from pearlsim.serpent_spectrum_tools import *
from pearlsim.results_processing import *

PEBBLE_FUEL_VOLUME = 0.36263376

def create_spectra_with_serpent(conc_df, decay_template_path, gamma_template_path, decay_point_path,
                                energy_grid, decay_days, decay_day_rel_unc, num_cores, iteration,
                                debug = 1, delete_inputs = True):
    # Create a set of materials and dummy-geometry spheres to insert into 
    # Serpent template
    start_time = time.time()
    decay_point_file = f"decay_points_{iteration}.csv"
    
    num_mats = len(conc_df)
    
    mat_counter = 0
    
    with open(decay_point_path, 'r') as i:
        with open(decay_point_file, 'w') as o:
            for line in i:
                line = line.split()
                r = np.sqrt(float(line[0])**2+float(line[1])**2)
                if r < 101:
                    continue
                else:
                    o.write(f"{line[0]} {line[1]} {line[2]} {line[3]} decay{mat_counter}u\n")
                mat_counter += 1
                if mat_counter > num_mats-1:
                    break
    
    decay_s = ""
    for i in range(num_mats):
        material_s = Material(f"decay{i}_R1Z1G1", conc_df.iloc[i].to_dict()).write_input(1, {}, 1, 
                                                                                  volume=PEBBLE_FUEL_VOLUME, never_burn=False)
        material_s += f"\nsurf decay{i}s sph 0 0 0 1.0 %\n"
        material_s += f"cell decay{i}c decay{i}u decay{i}_R1Z1G1 -decay{i}s%\n\n"
        decay_s += material_s
    
    with open(decay_template_path, 'r') as f:
        decay1_input_s = f.read()

    decay1_input_s = decay1_input_s.replace("<point_file>",str(decay_point_file))
    
    num_days = round(norm(decay_days, decay_days*decay_days_rel_unc).rvs(),6)
    
    decay1_input_s = decay1_input_s.replace("<days>",str(num_days))
    decay1_input_s += "\n%%% Decay Input Definitions %%%\n\n"
    decay1_input_s += decay_s
    
    decay1_file_name = f"decay_{iteration}_step1.serpent"
    with open(decay1_file_name, 'w') as f:
        f.write(decay1_input_s)
    os.system(f"sss2_2_0 {decay1_file_name} -omp {num_cores}")
    post_decay_concentrations = extract_from_bumat(decay1_file_name+".bumat1")

    # Skip the xenon dummy material
    conc_df = pd.DataFrame(post_decay_concentrations[1:]).fillna(0)

    
    decay_s = ""
    surfaces_s = ""
    for i in range(num_mats):
        material_s = Material(f"decay{i}_R1Z1G1", conc_df.iloc[i].to_dict()).write_input(1, {}, 1, 
                                                                                  volume=PEBBLE_FUEL_VOLUME, never_burn=True)
        material_s += f"\nsurf decay{i}s sph 0 0 0 1.0 %\n"
        material_s += f"cell decay{i}c decay{i}u decay{i}_R1Z1G1 -decay{i}s%\n\n"
        decay_s += material_s

    with open(gamma_template_path, 'r') as f:
        decay2_input_s = f.read()
    
    decay2_input_s += f"ene detector_grid 1 {np.array2string(energy_grid,threshold=20000,precision=10)[1:-1]}\n"
    decay2_input_s += f"set dspec detector_grid detector_grid"
    decay2_input_s = decay2_input_s.replace("<point_file>",str(decay_point_file))
    decay2_input_s += "\n%%% Decay Input Definitions %%%\n\n"
    decay2_input_s += decay_s
    
    decay2_file_name = f"decay_{iteration}_step2.serpent"
    with open(decay2_file_name, 'w') as f:
        f.write(decay2_input_s)
    os.system(f"sss2_2_0 {decay2_file_name} -omp {num_cores}")
    #if delete_inputs:
    #    os.system(f"rm {decay1_file_name}")
    #    os.system(f"rm {decay2_file_name}")
    return f"decay_{iteration}_step2.serpent_gsrc.m"




def run_one_pebble_decay(conc_dict, pebble_id, decay_template_path, gamma_template_path,
                                energy_grid, decay_days, decay_days_unc, num_cores,
                                triso_file, particles, debug = 1, delete_inputs = True):
    # Create a set of materials and dummy-geometry spheres to insert into 
    # Serpent template
    start_time = time.time()
    
    num_day_steps = len(decay_days)+1

    # STEP 1: Decay of pebble nuclides
    mat_s = ""

    mat_s += Material("decay0_R1Z1G1", conc_dict).write_input(1, {}, 1, 
                        volume=PEBBLE_FUEL_VOLUME, never_burn=False)

    
    with open(decay_template_path, 'r') as f:
        decay1_input_s = f.read()

    decay1_input_s = decay1_input_s.replace("<triso_file>",str(triso_file))
    
    day_array = []
    
    for t in range(len(decay_days)):
        mean = decay_days[t]
        std = decay_days_unc[t]
        day_array += [str(round(norm(mean, std).rvs(),6))]
    
    decay1_input_s = decay1_input_s.replace("<days>"," ".join(day_array))
    decay1_input_s += "\n%%% Decay Input Definitions %%%\n\n"
    decay1_input_s += mat_s
    
    decay1_file_name = f"decay_{pebble_id}_step1.serpent"
    with open(decay1_file_name, 'w') as f:
        f.write(decay1_input_s)
    os.system(f"sss2_2_0 {decay1_file_name} -omp {num_cores}")
    
    post_decay_concentrations = {}
    decay_days = [0] + decay_days
    for i in range(num_day_steps):
        
        post_decay_concentrations[decay_days[i]] = extract_from_bumat(f"{decay1_file_name}.bumat{i}")[0]


    # STEP 2: Gamma Transport To Surface
    
    for t in range(num_day_steps):
        day = decay_days[t]
        
        mat_s = Material(f"decay0_R1Z1G1", post_decay_concentrations[day]).write_input(1, {}, 1, 
                            volume=PEBBLE_FUEL_VOLUME, never_burn=True)
    
        with open(gamma_template_path, 'r') as f:
            decay2_input_s = f.read()
        
        decay2_input_s += f"ene detector_grid 1 {np.array2string(energy_grid,threshold=20000,precision=10)[1:-1]}\n"
        decay2_input_s += f"set dspec detector_grid detector_grid"
        decay2_input_s = decay2_input_s.replace("<particles>",str(particles))
        decay2_input_s = decay2_input_s.replace("<triso_file>",str(triso_file))

        decay2_input_s += "\n%%% Decay Input Definitions %%%\n\n"
        decay2_input_s += mat_s
        
        decay2_file_name = f"decay_{pebble_id}_step2_{round(day,3)}d.serpent"
        with open(decay2_file_name, 'w') as f:
            f.write(decay2_input_s)
        os.system(f"sss2_2_0 {decay2_file_name} -omp {num_cores}")

    return f"decay_{iteration}_step2.serpent_gsrc.m"


def read_gspec(file_name, energy_spectrum):
    results = {}
    num_energies = len(energy_spectrum)-1
    with open(file_name, 'r') as f:
        reading = False
        for line in f:
            if "];" in line:
                reading = False
            if reading:
                line = line.split()
                value_array[i] = float(line[0])
                i += 1
            if "gspec =" in line:
                reading = True
                key = line.split("_")[1]
                i = 0
                value_array = [0]*num_energies
                results[key] = value_array
    return results
    
ZAI_list = [922350, 922380, 942390, 942410, 541350, 551370, 10010, 10030, 20040, 30070, 40100, 50100, 50110, 60120, 60140, 80160, 80170, 290670, 300660, 300670, 300680, 300700, 300720, 310690, 310710, 310720, 320720, 320730, 320740, 320760, 320770, 320780, 330750, 330760, 330770, 330780, 330790, 330810, 340760, 340770, 340780, 340790, 340791, 340800, 340810, 340811, 340820, 340830, 340840, 340850, 340860, 350790, 350810, 350820, 350830, 350840, 350841, 350850, 350860, 350870, 350880, 360800, 360810, 360820, 360830, 360831, 360840, 360850, 360851, 360860, 360870, 360880, 360890, 360900, 360910, 370830, 370840, 370850, 370860, 370870, 370880, 370890, 370900, 370901, 370910, 370920, 370930, 380860, 380870, 380880, 380890, 380900, 380910, 380920, 380930, 380940, 380950, 390880, 390890, 390891, 390900, 390910, 390911, 390920, 390930, 390940, 390950, 390960, 390961, 390970, 400900, 400910, 400920, 400930, 400940, 400950, 400960, 400970, 400980, 400990, 401000, 401010, 401020, 410930, 410931, 410940, 410950, 410951, 410960, 410970, 410980, 410981, 410990, 410991, 411000, 411010, 420940, 420950, 420960, 420970, 420980, 420990, 421000, 421010, 421020, 421030, 421040, 421050, 421060, 430980, 430990, 430991, 431000, 431010, 431020, 431030, 431040, 431050, 431060, 431070, 440990, 441000, 441010, 441020, 441030, 441040, 441050, 441060, 441070, 441080, 441090, 451020, 451021, 451030, 451031, 451040, 451041, 451050, 451051, 451060, 451061, 451070, 451080, 451090, 461040, 461050, 461060, 461070, 461080, 461090, 461100, 461110, 461120, 471090, 471091, 471101, 471110, 471111, 471120, 471130, 471150, 481100, 481110, 481120, 481130, 481131, 481140, 481150, 481151, 481160, 481170, 481171, 481180, 491130, 491150, 491151, 491170, 491171, 491191, 501150, 501160, 501170, 501171, 501180, 501190, 501191, 501200, 501210, 501211, 501220, 501230, 501231, 501240, 501250, 501251, 501260, 501270, 501271, 501280, 501290, 501291, 501300, 501301, 501310, 501311, 501320, 511210, 511220, 511230, 511240, 511250, 511260, 511261, 511270, 511280, 511281, 511290, 511300, 511301, 511310, 511320, 511321, 511330, 521220, 521230, 521231, 521240, 521250, 521251, 521260, 521270, 521271, 521280, 521290, 521291, 521300, 521310, 521311, 521320, 521330, 521331, 521340, 521350, 521360, 531260, 531270, 531280, 531290, 531300, 531301, 531310, 531320, 531321, 531330, 531340, 531341, 531350, 531360, 531361, 531370, 531380, 541280, 541290, 541300, 541310, 541311, 541320, 541330, 541331, 541340, 541351, 541360, 541370, 541380, 541390, 541400, 551320, 551330, 551340, 551341, 551350, 551351, 551360, 551380, 551381, 551390, 551400, 551410, 561320, 561340, 561350, 561360, 561370, 561371, 561380, 561390, 561400, 561410, 561420, 561430, 561440, 561450, 571370, 571380, 571390, 571400, 571410, 571420, 571430, 571440, 571450, 571460, 571461, 581380, 581390, 581400, 581410, 581420, 581430, 581440, 581450, 581460, 581470, 581480, 591410, 591420, 591421, 591430, 591440, 591441, 591450, 591460, 591470, 591480, 591481, 591490, 591510, 601420, 601430, 601440, 601450, 601460, 601470, 601480, 601490, 601500, 601510, 601520, 601530, 611460, 611470, 611480, 611481, 611490, 611510, 611520, 611530, 611540, 621470, 621480, 621490, 621500, 621510, 621520, 621530, 621540, 621550, 621560, 621570, 621580, 631510, 631520, 631530, 631540, 631541, 631550, 631560, 631570, 631580, 631590, 641520, 641540, 641550, 641560, 641570, 641580, 641590, 641600, 651580, 651590, 651600, 651610, 661600, 661610, 661620, 661630, 661640, 661660, 671650, 671660, 671661, 681660, 681670, 681680, 681690, 681700, 691690, 691710, 701720, 902310, 902320, 902340, 912310, 912340, 922320, 922340, 922360, 922370, 922390, 932370, 932380, 932390, 942380, 942400, 942420, 952410]

def rename_ZAI_columns(dataframe):
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

    for column in dataframe.columns:
        raw_label = column
        column = str(column)
        if not column.isnumeric():
            continue
        iso_num = int(column[-1])
        a = int(column[-4:-1])
        z = int(column[:-4])
        renamed = f"{z_map_inv[z]}-{a}"
        if iso_num > 0:
            renamed += "m"
        rename_map[raw_label] = renamed
    renamed_dataframe = dataframe.rename(columns=rename_map)

    return renamed_dataframe

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


# Function to read a restart file and extract material data. Can put any ZAI in ZAI_fields, or one/multiple of the following in nonZAI_fiels: n, name, bu_global, bu_days, nnuc, adens, mdens, burnup 
def read_restart(list_paths, step=0, mat_parent=None, nonZAI_fields=['burnup'], ZAI_fields=[], df=True, min_ZAI_adens=None, parent_only=False, printing=True):
    '''
    Function by Yves to extract materials from a restart file.
    '''
    if isinstance(list_paths, str):
        list_paths = [list_paths]
    materials = dict()
    ZAI_indices = []
    for path_to_file in list_paths:
        if printing:
            print(path_to_file)
        current_step = -1
        burnups = dict()
        # Read restart file
        with open(path_to_file, mode='rb') as file:  # b is important -> binary
            while True:
                s = file.read(8)
                if not s:
                    break
                material = {}
                material['n'] = struct.unpack("q", s)[0]  # length of material name
                material['name'] = struct.unpack("{}s".format(material['n']), file.read(material['n']))[0].decode('UTF-8') # material name
                material['bu_global'] = struct.unpack("d", file.read(8))[0] # BU of snapshot
                material['bu_days'] = struct.unpack("d", file.read(8))[0] # time of snapshot
                material['nnuc'] = struct.unpack("q", file.read(8))[0] # Number of nuclides in material
                material['adens'] = struct.unpack("d", file.read(8))[0] # Atomic density of material
                material['mdens'] = struct.unpack("d", file.read(8))[0] # Mass density of material
                material['burnup'] = struct.unpack("d", file.read(8))[0] # Burnup of material
                if len(burnups) == 0 or material['bu_global'] != burnups[list(burnups.keys())[-1]]:
                    current_step += 1
                    burnups[current_step] = material['bu_global']

                # Check if material name matches
                if mat_parent and ((not parent_only and material['name'][:min(len(mat_parent), len(material['name']))+1] != f'{mat_parent}z') or current_step != step):
                    # Seek to the next block by calculating the number of bytes to skip
                    bytes_to_skip = 16 * material['nnuc']  # Size of the data block (16 bytes for each nuclide)
                    file.seek(bytes_to_skip, 1)  # Move the file pointer forward by the calculated number of bytes
                    continue

                materials[material['name']] = {field: material[field] for field in nonZAI_fields}
                if len(ZAI_fields)==0:
                    # Seek to the next block by calculating the number of bytes to skip
                    file.seek(16 * material['nnuc'], 1)  # Move the file pointer forward by the calculated number of bytes
                    continue

                # Just once
                if len(ZAI_indices) == 0:
                    adens_list = []
                    ZAI_list = []
                    for i in range(material['nnuc']):
                        ZAI, adens = struct.unpack("qd", file.read(16))
                        ZAI_list.append(ZAI)
                        adens_list.append(adens)
                    if isinstance(ZAI_fields, str) and ZAI_fields=='all':
                        ZAI_indices = range(len(ZAI_list))
                        ZAI_fields = ZAI_list
                    else:
                        ZAI_indices = [ZAI_list.index(int(ZAI)) for ZAI in ZAI_fields]
                    ZAI_indices, ZAI_fields = zip(*sorted(zip(ZAI_indices, ZAI_fields)))
                    ZAI_empty = {key: True for key in ZAI_fields}
                    for i, index in enumerate(ZAI_indices):
                        if not min_ZAI_adens or adens_list[index] > min_ZAI_adens: 
                            materials[material['name']][int(ZAI_fields[i])] = adens_list[index]
                            ZAI_empty[int(ZAI_fields[i])] = False
                        
                # The rest of the cases
                else:
                    last_index = 0
                    for i in range(len(ZAI_fields)):
                        index = ZAI_indices[i]
                        file.seek(16*(index-last_index), 1)
                        ZAI, adens = struct.unpack("qd", file.read(16))
                        if not min_ZAI_adens or adens > min_ZAI_adens:
                            materials[material['name']][int(ZAI)] = adens
                            ZAI_empty[int(ZAI)] = False
                        last_index = int(index)+1
                    file.seek(16*(material['nnuc']-last_index), 1)
        
    if df:
        materials = pd.DataFrame.from_dict(materials, orient='index')
        materials = materials.fillna(0).loc[natural_sort(materials.index)]
    else:
        materials = {key: materials[key] for key in natural_sort(materials)}
    return materials