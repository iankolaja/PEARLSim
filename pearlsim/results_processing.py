import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from .ml_utilities import extract_from_bumat, flatten_mesh, read_det_file
from pearlsim.group_definitions import *
from pearlsim.material import Material
from pearlsim.serpent_spectrum_tools import *
import os
import time
import gzip

def read_core_flux(file_name, variable_name = "14group_flux", normalize_and_label=False,
                  radius_grid = RADIUS_GRID_4, height_grid = HEIGHT_GRID_10, energy_grid = ENERGY_GRID_14):
    reading=False
    core_flux = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("DET","")
            if variable_name in line:
                reading = True
                data_array = []
                unc_array = []
            elif "]" in line and reading:
                core_flux = data_array
                avg_uncertainty = np.mean(np.array(unc_array))
                break
            elif reading:
                data = float(line.split()[10])
                unc = float(line.split()[11])
                data_array += [data]
                unc_array += [unc]
    if normalize_and_label:
        core_flux_headers = []
        num_radius_grid = len(radius_grid)-1
        num_height_grid = len(height_grid)-1
        num_energy_grid = len(energy_grid)-1
        bin_e = 0
        bin_r = -1
        bin_z = 0
        for i in range(len(core_flux)):
            if bin_r == num_radius_grid-1:
                bin_r = 0
                if bin_z == num_height_grid-1:
                    bin_z = 0
                    bin_e += 1
                else:
                    bin_z += 1
            else:
                bin_r += 1
            # Calculate volume of a washer, noting that height bins are in descending
            # order while radius bins are increasing
            volume = np.pi*(radius_grid[bin_r+1]**2-radius_grid[bin_r]**2)*(height_grid[bin_z+1]-height_grid[bin_z])
            energy_width = energy_grid[bin_e+1]-energy_grid[bin_e]
            core_flux[i] = core_flux[i]/volume/(energy_width*1e6)
            core_flux_headers += [f"binR{bin_r+1}Z{bin_z+1}E{bin_e+1}"]
    else:
        core_flux_headers = ["bin" + str(n) for n in range(1, 1 + len(core_flux))]
    core_flux = pd.DataFrame([core_flux], columns=core_flux_headers)
    return core_flux, avg_uncertainty



def read_res_file(file_name, parameter, burnup_step=2, sub_index=0):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        matches = 0
        matching_lines = []
    # Sample : "ANA_KEFF    (idx, [1:   6]) = [  1.29640E+00 0.00153  1.28804E+00 0.00150  8.70269E-03 0.02988 ];"
        for i in range(len(lines)):
            if parameter in lines[i]:
                matches += 1
                if burnup_step == "all":
                    matching_lines += [lines[i]]
                if matches == burnup_step:
                    result = lines[i]
                    break
    # Sample : " [  1.29640E+00 0.00153  1.28804E+00 0.00150  8.70269E-03 0.02988 ];"
    if type(burnup_step) == int:    
        result = result.split("=")[1]
        value = float(result.split()[1+sub_index*2])
        unc = float(result.split()[2+sub_index*2])
        return value, unc
    else:
        values = []
        unc_values = []
        for result in matching_lines:
            result = result.split("=")[1]
            values += [float(result.split()[1+sub_index*2])]
            unc_values += [float(result.split()[2+sub_index*2])]
        return values, unc_values


def extract_lstm_input_features(directory, step, run_name="gFHR_equilibrium", merge=True):
    averaged_features = {} 

    # Extract operating controls
    operating_parameter_file = f"{directory}/{run_name}_operating_{step}.json"
    with open(operating_parameter_file, 'r') as fin:
        input_table = pd.Series(json.loads(fin.read()))

    # Extract discharge and discard pebble data
    discharge_material_file = f"{directory}/{run_name}_discharge_inventory{step}.json"
    with open(discharge_material_file, 'r') as fin:
         discharge_material_data = json.loads(fin.read())

    discard_material_file = f"{directory}/{run_name}_discard_inventory{step}.json"
    with open(discard_material_file, 'r') as fin:
         discard_material_data = json.loads(fin.read())
    
    # Extract zone_averaged features
    radial_materials = {}
    for key in discharge_material_data.keys():
        if "graph" in key:
            continue
        id = key.split("_")[2]
        radial_zone = id.split("R")[1].split("Z")[0]
        fima_data = (discharge_material_data[key]["count"], 
                           discharge_material_data[key]["FIMA"], 
                           discharge_material_data[key]["FIMA_last_pass"])
        if radial_zone in radial_materials.keys():
            radial_materials[radial_zone] += [fima_data]
        else:
            radial_materials[radial_zone] = [fima_data]

    # Track number of pebbles discarded
    num_discarded = 0
    for key in discard_material_data.keys():
        num_discarded += discard_material_data[key]["count"]
    discard_series = pd.Series({"num_discarded":num_discarded})
    
    fima_averages = {}
    total_weighted_fima_sum = 0.0
    total_discharge_pebbles = 0
    fima_bin_counts = np.zeros(len(BURNUP_BINS_9)-1)
    fima_bin_boundaries = BURNUP_BINS_9
    for key in radial_materials.keys():
        #weighted_fima_sum = 0.0
        weighted_last_fima_sum = 0.0
        radial_total_pebbles = 0
        for fima_data in radial_materials[key]:
            #weighted_fima_sum += fima_data[0]*fima_data[1]
            total_weighted_fima_sum += fima_data[0]*fima_data[1]
            weighted_last_fima_sum += fima_data[0]*fima_data[2]
            radial_total_pebbles += fima_data[0]
            total_discharge_pebbles += fima_data[0]
            fima_bin_index = np.searchsorted(fima_bin_boundaries,fima_data[1])-1
            fima_bin_counts[fima_bin_index] += fima_data[0]
        #fima_averages[f"R{key}_avg_FIMA"] = weighted_fima_sum/radial_total_pebbles
        fima_averages[f"R{key}_avg_FIMA_last_pass"] = weighted_last_fima_sum/radial_total_pebbles
    
    for i in range(len(fima_bin_counts)):
        fima_averages[f"burnup_bin{i+1}_count"] = fima_bin_counts[i] 
    
    fima_averages["avg_FIMA"] = total_weighted_fima_sum / total_discharge_pebbles
    fima_average_series = pd.Series(fima_averages)
    
    
    averaged_features = pd.concat([fima_average_series, 
                                  discard_series])

    if merge:
        return pd.concat([input_table, averaged_features])
    else:
        return input_table, averaged_features


def collect_results_from_step(directory, step, run_name="gFHR_equilibrium", keff_index = -1, meshes = [], 
                              include_simulation_data=True):
    # Inputs to sequence model
    input_table = {}
    # Targets to predict with sequence model
    output_table = {}
    output_meshes = {}

    # Representative averages of features produced in hyperfidelity model
    averaged_features = {} 
    
    # Table of data needed for hyperfidelity and gamma spectrum simulations
    simulation_data = {}
    


    res_file = f"{directory}/{run_name}_{step}.serpent_res.m"
    keff_values, keff_unc_values = read_res_file(res_file, "ANA_KEFF", burnup_step="all")
    output_table["final_analog_keff"] = keff_values[keff_index]
    output_table["final_analog_keff_unc"] = keff_unc_values[keff_index]

    det_file = f"{directory}/{run_name}_{step}.serpent_det{len(keff_values)-1}.m"
    if not os.path.exists(det_file):
        det_file = f"{directory}/{run_name}_{step}.serpent_det{len(keff_values)-2}.m"
    out_meshes = read_det_file(det_file, 
                           meshes_to_read=["subzone_3group_flux",
                                           "subzone_power"], 
                           read_pebbles=False, 
                           mesh_grids_rze = [(RADIUS_GRID_8, HEIGHT_GRID_20, ENERGY_GRID_3),
                                             (RADIUS_GRID_8, HEIGHT_GRID_20, [0.0])])
    # Extract the meshes used for PEARLSim hyperfidelity
    if include_simulation_data:
        det_file = f"{directory}/{run_name}_{step}.serpent_det0.m"
        coarse_mesh = read_det_file(det_file, 
                               meshes_to_read=["coarse_7group_flux"], 
                               read_pebbles=False, 
                               mesh_grids_rze = [(RADIUS_GRID_3, HEIGHT_GRID_8, ENERGY_GRID_7)])
        simulation_data["coarse_7group_flux"] = coarse_mesh["coarse_7group_flux"]
    output_table.update(flatten_mesh(out_meshes["subzone_3group_flux"]["data"]))
    output_meshes["subzone_3group_flux"] = out_meshes["subzone_3group_flux"]
    output_table.update(flatten_mesh(out_meshes["subzone_power"]["data"], parameter_name="power"))
    output_meshes["subzone_power"] = out_meshes["subzone_power"]

    input_table, averaged_features = extract_lstm_input_features(directory, 
                                                                 step, 
                                                                 run_name=run_name,
                                                                 merge=False)
    
    return input_table, averaged_features, output_table, output_meshes, simulation_data

    
        
        