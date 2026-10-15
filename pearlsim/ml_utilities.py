import numpy as np
import pandas as pd
from pearlsim.group_definitions import *

# These energy and spatial bins are defined in the Serpent input



def read_det_file(file_name, meshes_to_read=["coarse_7group_flux"],
                  mesh_grids_rze = [(RADIUS_GRID_3, HEIGHT_GRID_8, ENERGY_GRID_7)]):
    """Read cylindrical r-z mesh detectors from a Serpent _det.m file as CELL AVERAGES.

    Serpent writes each mesh cell's result integrated over the cell (a power mesh
    with ``dr -8 void`` under ``set power`` is in W per cell). This reader divides
    every value by the cell's geometric volume pi (r2^2 - r1^2) dz, so it returns
    power in W/cm3 and flux in n/cm2/s. ``unc`` is Serpent's relative uncertainty,
    unchanged. pearlsim.records.power_summary multiplies the volumes back for W.
    """
    reading=False
    skip_names = ["E", "PHI", "Z", "R"]

    meshes = {}
    for ind in range(len(meshes_to_read)):
        mesh_name = meshes_to_read[ind]
        meshes[mesh_name] = {}
        radius_grid = mesh_grids_rze[ind][0]
        meshes[mesh_name]["radius_grid"] = radius_grid
        height_grid = mesh_grids_rze[ind][1]
        meshes[mesh_name]["height_grid"] = height_grid
        energy_grid = mesh_grids_rze[ind][2]
        meshes[mesh_name]["energy_grid"] = energy_grid

        # Power has no energy grid
        if "power" in mesh_name:
            meshes[mesh_name]["data"] = np.zeros( (len(radius_grid)-1,
                                                   len(height_grid)-1,
                                                   1 ) )
            meshes[mesh_name]["unc"] = np.zeros( (len(radius_grid)-1,
                                                  len(height_grid)-1,
                                                   1 ) )
        else:
            meshes[mesh_name]["data"] = np.zeros( (len(radius_grid)-1,
                                                   len(height_grid)-1,
                                                   len(energy_grid)-1 ) )
            meshes[mesh_name]["unc"] = np.zeros( (len(radius_grid)-1,
                                                  len(height_grid)-1,
                                                  len(energy_grid)-1 ) )
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("DET","")

            # Skip the mesh definitions since they should be known
            if "=" in line:
                if any([x in line for x in skip_names]):
                    check = line.split(" ")[0]
                    if check[-1] in ["E", "R", "Z", "I"]:
                        continue
                    else:
                        pass

                # if mesh
                if any(name in line for name in meshes.keys()):
                    set_name = line.split()[0]
                    reading = True
                    mesh_data = meshes[set_name]["data"]
                    mesh_data_unc = meshes[set_name]["unc"]
                    radius_grid = meshes[set_name]["radius_grid"]
                    height_grid = meshes[set_name]["height_grid"]
                    energy_grid = meshes[set_name]["energy_grid"]

            elif "];" in line and reading:
                reading = False

            elif reading:
                line = line.split()
                bin_e = int(line[1])-1
                bin_r = int(line[9])-1
                bin_z = int(line[7])-1
                data = float(line[10])
                unc = float(line[11])
                # integrated -> cell average (see docstring)
                volume = np.pi*(radius_grid[bin_r+1]**2-radius_grid[bin_r]**2)*(height_grid[bin_z+1]-height_grid[bin_z])
                if len(energy_grid) > 1:
                    energy_width = energy_grid[bin_e+1]-energy_grid[bin_e]
                    meshes[set_name]["data"][bin_r, bin_z, bin_e] = data/volume#/(energy_width*1e6)
                    meshes[set_name]["unc"][bin_r, bin_z, bin_e] = unc
                else:
                    meshes[set_name]["data"][bin_r, bin_z] = data/volume
                    meshes[set_name]["unc"][bin_r, bin_z] = unc
    return meshes


def flatten_mesh(mesh_data, parameter_name="flux"):
    num_r, num_z, num_e = np.shape(mesh_data)
    flattened_data = {}
    for r in range(num_r):
        for z in range(num_z):
            for e in range(num_e):
                if "power" in parameter_name:
                    key = f"{parameter_name}R{r+1}Z{z+1}"
                else:                
                    key = f"{parameter_name}R{r+1}Z{z+1}E{e+1}"
                flattened_data[key] = mesh_data[r, z, e]
    return flattened_data
                

def extract_from_bumat(file_path, return_list = True):
    if return_list:
        concentrations = []
    else:
        concentrations = {}
    with open(file_path, 'r') as f:
        reading = False
        first_mat = True
        for line in f:
            line = line.split()
            if len(line) == 0:
                continue
            if line[0] == "mat":
                if not first_mat:
                    if return_list:
                        concentrations += [current_conc]
                    else:
                        concentrations[current_mat_name] = current_conc
                current_mat_name = line[1].split("pp")[0]
                current_conc = {}
                reading = True
                first_mat = False
            elif reading:
                id = line[0].split(".")
                if len(id) > 1:
                    nuclide = id[0] + "<lib>"
                else:
                    nuclide = id[0]
                amount = float(line[1].replace("\n", ""))
                current_conc[nuclide] = amount
    if return_list:
        concentrations += [current_conc]
    else:
        concentrations[current_mat_name] = current_conc
    return concentrations

def standardize(raw_data, mean=None, std=None, axis=0):
    if mean is None:
        mean = np.mean(raw_data, axis = axis)
    if std is None:
        std = np.std(raw_data, axis = axis)
        std[ std==0 ] = 1
    result = (raw_data - mean) / std
    return result, mean, std

def unstandardize(standardized_data, mean, std):
    raw_data = (standardized_data*std.values)+mean.values
    return raw_data



def add_computed_features(input_df,num_pebbles=250131,):
    fuel_insertion_fraction = 1-input_df["graphite_insertion_fraction"]
    for i in range(1, 11):
        fuel_insertion_fraction[-i] = fuel_insertion_fraction[0]
    reactor_fuel_fraction = fuel_insertion_fraction.sort_index().rolling(10).sum()[10:]/10
    input_df["average_power_per_pebble"] = input_df["power"]/(num_pebbles*reactor_fuel_fraction)
    return input_df

def add_feature_noise(input_df, noise_dict):
    length = len(input_df)
    for key in noise_dict.keys():
        std_dev = noise_dict[key]/np.sqrt(2/np.pi)
        noise_array = np.random.normal(1, std_dev, length)
        input_df[key] = input_df[key]*noise_array
    return input_df

def exclude_features(input_df, features_to_drop = []):
    input_df.drop(columns=features_to_drop, errors="ignore")
    return input_df 

def add_computed_targets(output_df):
    output_df["reactivity"] = (output_df["final_analog_keff"]-1)/output_df["final_analog_keff"]
    return output_df

