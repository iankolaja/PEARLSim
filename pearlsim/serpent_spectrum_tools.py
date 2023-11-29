import time
import numpy as np
import os
import pandas as pd
from pearlsim.material import Material

PEBBLE_FUEL_VOLUME = 0.36263376

def create_spectra_with_serpent(conc_df, decay_template_path, gamma_template_path, decay_point_path,
                                energy_grid, decay_days, num_cores, iteration,
                                debug = 1, delete_inputs = True):
    # Create a set of materials and dummy-geometry spheres to insert into 
    # Serpent template
    start_time = time.time()
    decay_point_file = f"decay_points_{iteration}.csv"
    
    num_mats = len(conc_df)
    
    mat_counter = 1
    
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
        material_s = Material(f"decay{i}", conc_df.iloc[i].to_dict()).write_input(1, {}, 1, 
                                                                                  volume=PEBBLE_FUEL_VOLUME, never_burn=False)
        material_s += f"\nsurf decay{i}s sph 0 0 0 1.0 %\n"
        material_s += f"cell decay{i}c decay{i}u decay{i} -decay{i}s%\n\n"
        decay_s += material_s
    
    with open(decay_template_path, 'r') as f:
        decay1_input_s = f.read()

    decay1_input_s = decay1_input_s.replace("<point_file>",str(decay_point_file))
    decay1_input_s += "\n%%% Decay Input Definitions %%%\n\n"
    decay1_input_s += decay_s
    
    decay1_file_name = f"decay_{iteration}_step1.serpent"
    with open(decay1_file_name, 'w') as f:
        f.write(decay1_input_s)
    os.system(f"sss2_2_0 {decay1_file_name} -omp {num_cores}")
    post_decay_concentrations = extract_from_bumat(decay1_file_name+".bumat1")
    conc_df = pd.DataFrame(post_decay_concentrations).fillna(0)

    
    decay_s = ""
    surfaces_s = ""
    for i in range(len(conc_df)):
        material_s = Material(f"decay{i}", conc_df.iloc[i].to_dict()).write_input(1, {}, 1, 
                                                                                  volume=PEBBLE_FUEL_VOLUME, never_burn=True)
        material_s += f"\nsurf decay{i}s sph 0 0 0 1.0 %\n"
        material_s += f"cell decay{i}c decay{i}u decay{i} -decay{i}s%\n\n"
        decay_s += material_s
    with open(gamma_template_path, 'r') as f:
        decay2_input_s = f.read()
    
    decay2_input_s += f"ene detector_grid 1 {np.array2string(energy_grid,threshold=10000,precision=8)[1:-1]}\n"
    decay2_input_s += f"set dspec detector_grid detector_grid"
    decay2_input_s = decay2_input_s.replace("<point_file>",str(decay_point_file))
    decay2_input_s += "\n%%% Decay Input Definitions %%%\n\n"
    decay2_input_s += decay_s
    
    decay2_file_name = f"decay_{iteration}_step2.serpent"
    with open(decay2_file_name, 'w') as f:
        f.write(decay2_input_s)
    os.system(f"sss2_2_0 {decay2_file_name} -omp {num_cores}")
    
    gspec_result_file = f"decay_{iteration}_step2.serpent_gsrc.m"
    gamma_spectrum = read_gspec(gspec_result_file, detector_energy_grid)
    spectrum_results = read_gspec(gspec_result_file, energy_grid)
    if delete_inputs:
        os.system(f"rm {decay1_input_s}*")
        os.system(f"rm {decay2_input_s}*")
    return spectrum_results

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
    