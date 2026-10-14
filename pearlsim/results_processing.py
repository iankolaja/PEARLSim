import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from .ml_utilities import *
from pearlsim.group_definitions import *
from pearlsim.material import Material
from pearlsim.serpent_spectrum_tools import *
import os
import time
import gzip
from matplotlib.colors import TwoSlopeNorm


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

    
def plot_mesh_comparison_from_series(mesh_predicted_series, mesh_actual_series, plot_label, fig, ax, num_row, plot_energy_group=None, points_per_cm=4):
    radius_grid = RADIUS_GRID_8
    height_grid = HEIGHT_GRID_20

    mesh_predicted = np.zeros( (len(radius_grid)-1, len(height_grid)-1 ) )
    mesh_actual =  np.zeros( (len(radius_grid)-1, len(height_grid)-1 ) )
    
    for header in mesh_predicted_series.index:
        radial_index = int(header.split("R")[1].split("Z")[0])-1
        axial_index = int(header.split("Z")[1].split("E")[0])-1
        if plot_energy_group is None:
            mesh_predicted[radial_index, axial_index] = mesh_predicted_series[header]
            mesh_actual[radial_index, axial_index] = mesh_actual_series[header]
        else:
            energy_index = int(header.split("E")[1])
            if energy_index == plot_energy_group:
                mesh_predicted[radial_index, axial_index] = mesh_predicted_series[header]
                mesh_actual[radial_index, axial_index] = mesh_actual_series[header]
        
    
    min_r = np.min(radius_grid)
    max_r = np.max(radius_grid)
    r_bins = len(radius_grid)-1
    min_z = np.min(height_grid)
    max_z = np.max(height_grid)
    z_bins = len(height_grid)-1

    
    num_r_points = int((max_r-min_r)*points_per_cm)
    num_z_points = int((max_z-min_z)*points_per_cm)
    r_array = np.linspace(0+1e-6,120-1e-6, num_r_points)
    z_array = np.linspace(min_z+1e-6,max_z-1e-6,num_z_points)
    r_bin_bounds = np.linspace(min_r,max_r,r_bins+1)
    z_bin_bounds = np.linspace(min_z,max_z,z_bins+1)
    
    mesh_predicted_points = np.zeros([num_r_points,num_z_points])
    mesh_actual_points = np.zeros([num_r_points,num_z_points])
   
    
    for r_ind in range(len(r_array)):
        for z_ind in range(len(z_array)):
            r = r_array[r_ind]
            z = z_array[z_ind]
            r_bin = np.searchsorted(r_bin_bounds, r)-1
            z_bin = np.searchsorted(z_bin_bounds, z)-1
            mesh_predicted_points[r_ind,z_ind] = mesh_predicted[r_bin,z_bin]
            mesh_actual_points[r_ind,z_ind] = mesh_actual[r_bin,z_bin]
    mesh_diff = (mesh_predicted_points - mesh_actual_points)/mesh_actual_points*100
    
    
    im1 = ax[num_row, 0].imshow(np.rot90(mesh_predicted_points,1), extent=(0,120,60,370), aspect='auto')
    plt.colorbar(im1, ax=ax[num_row, 0])
    #ax[num_row, 0].text(115, 350, plot_label + " (Predicted)", ha="right", va="center", color="white")
    ax[num_row, 0].set_ylabel("Components\n"+plot_label)
    ax[num_row, 0].set_xticks([])
    ax[num_row, 0].set_yticks([])
    
    #im2=ax[num_row, 1].imshow(np.rot90(mesh_actual_points,1), extent=(0,120,60,370), aspect='auto')
    #plt.colorbar(im2, ax=ax[num_row, 1])
    #ax[num_row, 1].set_xticks([])
    #ax[num_row, 1].set_yticks([])
    ##plt.text(115, 350, plot_label + " (Actual)", ha="right", va="center", color="white")

    min_val = min([-2,mesh_diff.min()])
    max_val = max([2,mesh_diff.max()])
        
    norm = TwoSlopeNorm(vmin=min_val-0.51, vcenter=0, vmax=max_val+0.51 )

    im3 = ax[num_row, 1].imshow(np.rot90(mesh_diff,1), extent=(0,120,60,370), aspect='auto',norm=norm,  cmap="RdBu")
    plt.colorbar(im3,ax=ax[num_row, 1], ticks=[round(min_val), round(min_val/2), 0, round(max_val/2), round(max_val)])
    ax[num_row, 1].set_xticks([])
    ax[num_row, 1].set_yticks([])
    #ax[num_row, 1].text(115, 350, plot_label + " (% Error)", ha="right", va="center", color="white")
    return 


def plot_mesh_from_series(mesh_predicted_series, mesh_actual_series, plot_label, file_name, plot_energy_group=None, points_per_cm=4):
    radius_grid = RADIUS_GRID_8
    height_grid = HEIGHT_GRID_20

    mesh_predicted = np.zeros( (len(radius_grid)-1, len(height_grid)-1 ) )
    mesh_actual =  np.zeros( (len(radius_grid)-1, len(height_grid)-1 ) )
    
    for header in mesh_predicted_series.index:
        radial_index = int(header.split("R")[1].split("Z")[0])-1
        axial_index = int(header.split("Z")[1].split("E")[0])-1
        if plot_energy_group is None:
            mesh_predicted[radial_index, axial_index] = mesh_predicted_series[header]
            mesh_actual[radial_index, axial_index] = mesh_actual_series[header]
        else:
            energy_index = int(header.split("E")[1])
            if energy_index == plot_energy_group:
                mesh_predicted[radial_index, axial_index] = mesh_predicted_series[header]
                mesh_actual[radial_index, axial_index] = mesh_actual_series[header]
        
    
    min_r = np.min(radius_grid)
    max_r = np.max(radius_grid)
    r_bins = len(radius_grid)-1
    min_z = np.min(height_grid)
    max_z = np.max(height_grid)
    z_bins = len(height_grid)-1

    
    num_r_points = int((max_r-min_r)*points_per_cm)
    num_z_points = int((max_z-min_z)*points_per_cm)
    r_array = np.linspace(0+1e-6,120-1e-6, num_r_points)
    z_array = np.linspace(min_z+1e-6,max_z-1e-6,num_z_points)
    r_bin_bounds = np.linspace(min_r,max_r,r_bins+1)
    z_bin_bounds = np.linspace(min_z,max_z,z_bins+1)
    
    mesh_predicted_points = np.zeros([num_r_points,num_z_points])
    mesh_actual_points = np.zeros([num_r_points,num_z_points])
   
    
    for r_ind in range(len(r_array)):
        for z_ind in range(len(z_array)):
            r = r_array[r_ind]
            z = z_array[z_ind]
            r_bin = np.searchsorted(r_bin_bounds, r)-1
            z_bin = np.searchsorted(z_bin_bounds, z)-1
            mesh_predicted_points[r_ind,z_ind] = mesh_predicted[r_bin,z_bin]
            mesh_actual_points[r_ind,z_ind] = mesh_actual[r_bin,z_bin]
    mesh_diff = (mesh_predicted_points - mesh_actual_points)/mesh_actual_points*100
    
    plt.figure()
    #plt.imshow(np.rot90(mesh_points,1), extent=(0,120,60,370), aspect='auto')
    plt.imshow(np.rot90(mesh_predicted_points,1), extent=(0,120,60,370), aspect='auto')
    plt.colorbar()
    #plt.title(f"Power Distribution by Subzone")
    plt.xlabel("Radial Position (cm)")
    plt.ylabel("Height (cm)")
    plt.text(115, 350, plot_label + " (Predicted)", ha="right", va="center", color="white")
    plt.savefig(file_name+"_pred.png")
    plt.show()

    plt.figure()
    #plt.imshow(np.rot90(mesh_points,1), extent=(0,120,60,370), aspect='auto')
    plt.imshow(np.rot90(mesh_actual_points,1), extent=(0,120,60,370), aspect='auto')
    plt.colorbar()
    #plt.title(f"Power Distribution by Subzone")
    plt.xlabel("Radial Position (cm)")
    plt.ylabel("Height (cm)")
    plt.text(115, 350, plot_label + " (Actual)", ha="right", va="center", color="white")
    plt.savefig(file_name+"_actual.png")
    plt.show()

    plt.figure()
    #plt.imshow(np.rot90(mesh_points,1), extent=(0,120,60,370), aspect='auto')
    norm = TwoSlopeNorm(vmin=min([-0.1,mesh_diff.min()]), vcenter=0, vmax=max( [0.1,mesh_diff.max()] ) )

    plt.imshow(np.rot90(mesh_diff,1), extent=(0,120,60,370), aspect='auto',norm=norm,  cmap="RdBu")
    plt.colorbar()
    #plt.title(f"Power Distribution by Subzone")
    plt.xlabel("Radial Position (cm)")
    plt.ylabel("Height (cm)")
    plt.text(115, 350, plot_label + " (% Error)", ha="right", va="center", color="white")
    plt.savefig(file_name+"_diff.png")
    plt.show()
    return 
        