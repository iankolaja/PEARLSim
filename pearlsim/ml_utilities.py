import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pearlsim.material import get_cross_section_string
from sklearn.neighbors import KNeighborsRegressor
from scipy.interpolate import griddata
from pearlsim.group_definitions import *

# These energy and spatial bins are defined in the Serpent input

PEBBLE_SURFACE_AREA = 4*np.pi*2.0**2
PEBBLE_KERNEL_VOLUME = 0.02125**3*np.pi*4/3*9022


def read_det_file(file_name, meshes_to_read=["coarse_7group_flux"], read_pebbles=False, 
                 mesh_grids_rze = [(RADIUS_GRID_3, HEIGHT_GRID_8, ENERGY_GRID_7)], 
                  pebble_energy_grid = ENERGY_GRID_56):
    reading=False
    skip_names = ["E", "PHI", "Z", "R"]
    id_array = []
    if read_pebbles:
        x_array = []
        y_array = []
        z_array = []
        pebble_target_datasets = {"peb_flux" : [], "peb_current" : [], 
        "peb_flux_unc" : [], "peb_current_unc" : []}
        zone_datasets = {}
        energy_centers = (pebble_energy_grid[1:] + pebble_energy_grid[:-1])/2
        energy_center_labels = []
        for e in energy_centers:
            energy_center_labels += [np.format_float_scientific(e,4)]

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

                # for group averaged flux
                # ex: DETavgflux_fuel20_R1Z1G1 = [
                if "avgflux" in line and read_pebbles:
                    set_name = line.split("_")[1] + "_" + line.split("_")[2] #zone_id
                    set_name = set_name.split(" ")[0]
                    reading = True
                    data_array = []
                    unc_array = []
                
                # if pebble
                elif "peb" in line and read_pebbles:
                    header = line.split("_")
                    data_array = []
                    unc_array = []
                    reading = True
                    if "pebflux" in header:
                        set_name = "peb_flux"
                    elif "pebcurrent" in header:
                        set_name = "peb_current"
                        id_array += [int(header[1])]
                        x_array += [float(header[2])]
                        y_array += [float(header[3])]
                        z_array += [float(header[4])]
                    else:
                        set_name = "peb"
            
            elif "];" in line and reading:
                reading = False
                
                if "peb" in set_name:
                    pebble_target_datasets[set_name] += [data_array]
                    pebble_target_datasets[set_name+"_unc"] += [unc_array]

                if "G" in set_name:
                    zone_datasets[set_name] = {}
                    zone_datasets[set_name]["flux"] = pd.Series(data_array, index=energy_center_labels)
                    zone_datasets[set_name]["flux_unc"] = pd.Series(unc_array, index=energy_center_labels)
                    
                    
            elif reading:
                line = line.split()
                if "peb" in set_name or "G" in set_name:
                    data = float(line[10])
                    unc = float(line[11])
                    data_array += [data]
                    unc_array += [unc]
                else:
                    bin_e = int(line[1])-1
                    bin_r = int(line[9])-1
                    bin_z = int(line[7])-1
                    data = float(line[10])
                    unc = float(line[11])
                    volume = np.pi*(radius_grid[bin_r+1]**2-radius_grid[bin_r]**2)*(height_grid[bin_z+1]-height_grid[bin_z])
                    if len(energy_grid) > 1:
                        energy_width = energy_grid[bin_e+1]-energy_grid[bin_e]
                        meshes[set_name]["data"][bin_r, bin_z, bin_e] = data/volume/(energy_width*1e6)
                        meshes[set_name]["unc"][bin_r, bin_z, bin_e] = unc   
                    else:
                        meshes[set_name]["data"][bin_r, bin_z] = data/volume
                        meshes[set_name]["unc"][bin_r, bin_z] = unc         
    if read_pebbles:
        for key in pebble_target_datasets.keys():
            pebble_target_datasets[key] = np.array(pebble_target_datasets[key])
            pebble_target_datasets[key] = pd.DataFrame(pebble_target_datasets[key], columns=energy_center_labels)
        pebble_features = pd.DataFrame({"x": x_array, "y": y_array, "z": z_array, "pebble_id":id_array })
        return meshes, pebble_features, pebble_target_datasets, zone_datasets
    else: 
        return meshes




def extract_xs_from_mdx(file_name):
    single_material_list = []
    with open(file_name, 'r') as f:
        is_reading = False
        all_material_dict = {}
        mat_name = "none"
        for line in f.readlines():
            if "FLUX" in line:
                flux = float(line.split()[3])
                flux_unc = float(line.split()[4])
            if "];" in line:
                is_reading = False
                if mat_name != "none":
                    xs_df = pd.DataFrame(single_material_list)
                    all_material_dict[mat_name] = {"flux": flux,
                                                   "flux_unc": flux_unc,
                                                   "cross_sections": xs_df}
                    mat_name = "none"
                    single_material_list = []
            if is_reading:
                line = line.split()

                isotope = int(line[0])
                mt = int(line[1])
                xs = float(line[5])
                xs_unc = float(line[6])
                flag = int(line[2])
                
                single_material_list += [{"isotope":isotope,
                                        "mt":mt,
                                        "flag":flag,
                                        "reaction_id":f"{isotope}-{mt}",
                                        "xs":xs,
                                        "xs_unc":xs_unc}]
                
            if "XS" in line:
                is_reading = True
                mat_name = line.split("_")[1][1:] + "_" + line.split("_")[2].split(" ")[0]
        
        return all_material_dict

def match_flux_to_xs_and_flatten(zone_features, peb_targets, xs_dict, unc_threshold=0.1):
    input_df_list = []
    input_reaction_dict = {}
    total_flux_list = []
    unique_flux_list = []
    xs_df_list = []
    xs_reaction_dict = {}
    sample_counter = 0
    for mat_name in xs_dict.keys():
        if "sample" in mat_name:
            assert str(sample_counter+1) in mat_name
            flux_series = peb_targets.iloc[sample_counter] 
            sample_counter += 1
        else:
            flux_series = zone_features[mat_name]["flux"]
        xs_data = xs_dict[mat_name]["cross_sections"].loc[:,["xs","xs_unc"]]
        unique_flux_list += [flux_series]
        total_flux_list += [xs_dict[mat_name]["flux"]]
        reaction_data = xs_dict[mat_name]["cross_sections"][["reaction_id"]].copy()
        for index, value in flux_series.items():
            reaction_data[index] = value
        input_df_list += [reaction_data]
        xs_df_list += [xs_data]
    input_data = pd.concat(input_df_list).reset_index(drop=True)
    xs_data = pd.concat(xs_df_list).reset_index(drop=True)
    for reaction_id in input_data['reaction_id'].unique():

        # get the data points that are for this reaction, and are nonzero
        reaction_indices = input_data.index[(input_data['reaction_id'] == reaction_id) & 
                                            (xs_data['xs'] != 0)]
    
        # If the uncertainty tends to be higher than the threshold, then use
        # the mean to set this threshold. This effectively ensures that only
        # group data will be used since it'll have below average uncertaioty.
        mean_unc = xs_data.iloc[reaction_indices]["xs_unc"].mean()
        if mean_unc < unc_threshold:
            threshold = unc_threshold
        else:
            threshold = mean_unc*1.1

        # Refilter only data points with acceptable uncertainty
        reaction_indices = input_data.index[(input_data['reaction_id'] == reaction_id) & 
                                            (xs_data['xs'] != 0) &
                                            (xs_data['xs_unc'] < threshold)]
        
        
        input_reaction_dict[reaction_id] = input_data.iloc[reaction_indices].drop(columns=["reaction_id"])
        xs_reaction_dict[reaction_id] = xs_data.loc[reaction_indices]['xs']
    input_reaction_dict["total_flux"] = pd.DataFrame(unique_flux_list).reset_index(drop=True)
    xs_reaction_dict["total_flux"] = pd.Series(total_flux_list, name="total_flux")
    display(input_reaction_dict["total_flux"])
    display(xs_reaction_dict["total_flux"])
    return input_reaction_dict, xs_reaction_dict
    
def expand_pebble_features_by_energy(pebble_features, energy_centers=ENERGY_CENTERS_56):
    feature_array = []
    for index, row in pebble_features.iterrows():
        for energy in energy_centers:
            point = row.copy()
            point["energy"] = energy
            feature_array += [point]
    flattened_features = pd.DataFrame(feature_array).reset_index(drop=True)
    return flattened_features

def flatten_pebble_data(pebble_features, pebble_target):
    feature_array = []
    target_array = []
    for index, row in pebble_target.iterrows():
        for energy, value in row.items():
            slice_features = pebble_features.iloc[index].copy()
            slice_features["energy"] = float(energy)
            feature_array += [slice_features]
            target_array += [value]
    flattened_features = pd.DataFrame(feature_array).reset_index(drop=True)
    return flattened_features, np.array(target_array)

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

def generate_pebble_burnup_model(template_path, surface_current, power, concentrations, temp, time,
                                energy_grid = ENERGY_GRID_14):
    lib_str = get_cross_section_string(temp)
    with open(template_path, "r") as f:
        input_s = f.read()
    concentration_s = ""
    for key in concentrations.keys():
        concentration_s  += f"  {key}    {concentrations[key]}\n".replace("<lib>", lib_str)
    input_s = input_s.replace("<concentrations>", concentration_s)
    input_s = input_s.replace("<temperature>", str(temp))
    input_s = input_s.replace("<time>", str(time))
    input_s = input_s.replace("<power>", str(power))

    weights = [0]+list((surface_current/1e9).astype(int))
    source_str = f"sb {len(weights)} 1\n"
    for i in range(len(surface_current)+1):
        source_str += f"  {energy_grid[i]} {weights[i]}\n"
    input_s = input_s.replace("<current>", source_str)
    return input_s


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



def interpolate_core_flux_3d(data_chunk):
    input_df = data_chunk["features"][["radius","height","energy"]]
    core_flux_map = data_chunk["flux_mesh"]
    num_neighbors = data_chunk["flux_num_neighbors"]
    interp_energy_grid = data_chunk["interp_energy_grid"]
    points_per_cm = data_chunk["points_per_cm"]
    radius_grid = data_chunk["flux_radius_grid"]
    height_grid = data_chunk["flux_height_grid"]
    energy_grid = data_chunk["energy_grid"]
    plot_energy = data_chunk["plot_energy"]
    min_r = np.min(radius_grid)
    max_r = np.max(radius_grid)
    r_bins = len(radius_grid)-1
    min_z = np.min(height_grid)
    max_z = np.max(height_grid)
    z_bins = len(height_grid)-1
    min_e = np.min(energy_grid)
    max_e = np.max(energy_grid)
    e_bins = len(energy_grid)-1
    
    num_r_points = int((max_r-min_r)*points_per_cm)
    num_z_points = int((max_z-min_z)*points_per_cm)
    
    r_array = np.linspace(min_r+1e-6,max_r-1e-6, num_r_points)
    z_array = np.linspace(min_z+1e-6,max_z-1e-6,num_z_points)
    e_array = interp_energy_grid
    num_e_points = len(e_array)

    R, Z, E = np.meshgrid(r_array,z_array, e_array, indexing="ij")
    Rp, Zp = np.meshgrid(r_array,z_array, indexing="ij")

    knn_model = KNeighborsRegressor(n_neighbors=num_neighbors)

    point_fluxes = np.zeros(num_r_points*num_z_points*num_e_points)
    plot_fluxes = np.zeros(num_r_points*num_z_points)
    p = 0
    plot_group = np.searchsorted(energy_grid, plot_energy)-1
    for e_ind in range(len(e_array)):
        for z_ind in range(len(z_array)):
            for r_ind in range(len(r_array)):
                r = r_array[r_ind]
                z = z_array[z_ind]
                e = e_array[e_ind]
                r_bin = np.searchsorted(radius_grid, r)-1
                z_bin = np.searchsorted(height_grid, z)-1
                e_bin = np.searchsorted(energy_grid, e)-1
                point_fluxes[p] = core_flux_map[r_bin,z_bin,e_bin]
                p += 1
    training_data = pd.DataFrame({"radius":R.T.flatten(), "height":Z.T.flatten(), "energy":E.T.flatten()})
    target_data = pd.DataFrame({"flux":point_fluxes})

    
    std_training_data, train_mean, train_std = standardize(training_data)
    std_input_df, _, _ = standardize(input_df, mean=train_mean, std=train_std)
    knn_model.fit(std_training_data, target_data)
    predicted = knn_model.predict(std_input_df)
    
    if type(plot_energy) == float and data_chunk["process_num"] == 0:

        point_grid = point_fluxes.reshape(num_e_points,num_z_points,num_r_points)
        plt.figure()
        plt.imshow(np.flipud(point_grid[plot_group,:,:]),  extent=(min_r,max_r,min_z,max_z), aspect='auto')
        plt.colorbar()
        plt.title(f"Flux Distribution by Zone")
        plt.xlabel("Radial Position (cm)")
        plt.ylabel("Height (cm)")
        plt.show()

        p = 0
        for z_ind in range(len(z_array)):
            for r_ind in range(len(r_array)):
                r = r_array[r_ind]
                z = z_array[z_ind]
                r_bin = np.searchsorted(radius_grid, r)-1
                z_bin = np.searchsorted(height_grid, z)-1
                plot_fluxes[p] = core_flux_map[r_bin,z_bin,plot_group]
                p+=1
        group_energy = (energy_grid[plot_group]+energy_grid[plot_group+1])/2
        plot_data = pd.DataFrame({"radius":Rp.T.flatten(), "height":Zp.T.flatten()})
        plot_data["energy"] = group_energy
        std_plot_data,_,_ = standardize(plot_data, mean=train_mean, std=train_std)
        predicted_grid = knn_model.predict(std_plot_data).reshape(num_z_points,num_r_points)
        plt.figure()
        plt.imshow(np.flipud(predicted_grid[:,:]),  extent=(min_r,max_r,min_z,max_z), aspect='auto')
        plt.colorbar()
        plt.xlabel("Radial Position (cm)")
        plt.ylabel("Height (cm)")
        plt.show()

        predict_plot_df = input_df.drop_duplicates(subset=['radius', 'height']).copy()
        predict_plot_df["energy"] = group_energy
        std_predict_plot_df,_,_ = standardize(predict_plot_df, mean=train_mean, std=train_std)
        plot_predicted = knn_model.predict(std_predict_plot_df)
        interpolated_values = griddata((predict_plot_df['radius'],predict_plot_df['height']), 
                                       plot_predicted, 
                                       (Rp, Zp), 
                                       rescale=True)
        plt.figure()
        plt.imshow(np.rot90(interpolated_values,1),  extent=(min_r,max_r,min_z,max_z), aspect='auto')
        plt.colorbar() 
        plt.title(f"KNN Interpolated Flux at points")
        plt.xlabel("Radial Position (cm)")
        plt.ylabel("Height (cm)")
        plt.show()

    return predicted

def interpolate_core_power(data_chunk):
    position_df = data_chunk["features"][["radius","height"]]
    mesh = data_chunk["power_mesh"]
    num_neighbors = data_chunk["power_num_neighbors"]
    points_per_cm = data_chunk["points_per_cm"]
    radius_grid = data_chunk["power_radius_grid"]
    height_grid = data_chunk["power_height_grid"]
    do_plot = data_chunk["do_plot"]
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
    R, Z = np.meshgrid(r_array,z_array)
    knn_model = KNeighborsRegressor(n_neighbors=num_neighbors)
    predicted_df = pd.DataFrame()
    power_matrix = mesh[:,:,0]

    point_powers = np.zeros([num_r_points,num_z_points])

    for r_ind in range(len(r_array)):
        for z_ind in range(len(z_array)):
            r = r_array[r_ind]
            z = z_array[z_ind]
            r_bin = np.searchsorted(r_bin_bounds, r)-1
            z_bin = np.searchsorted(z_bin_bounds, z)-1
            point_powers[r_ind,z_ind] = power_matrix[r_bin,z_bin]
    training_data = pd.DataFrame({"radius":R.T.flatten(), "height":Z.T.flatten()})
    target_data = pd.DataFrame({"flux":point_powers.flatten()})
    knn_model.fit(training_data, target_data)
    predicted = knn_model.predict(position_df)
    if do_plot and data_chunk["process_num"] == 0:
        plt.figure()
        plt.imshow(np.rot90(point_powers,1), extent=(0,120,60,370), aspect='auto')
        plt.colorbar()
        plt.title(f"Power Distribution by Subzone")
        plt.xlabel("Radial Position (cm)")
        plt.ylabel("Height (cm)")
        plt.show()
        
        predicted_grid = knn_model.predict(training_data).reshape(num_r_points,num_z_points)
        plt.figure()
        plt.imshow(np.rot90(predicted_grid,1),  extent=(min_r,max_r,min_z,max_z), aspect='auto')
        plt.colorbar()
        plt.title(f"KNN Resampled Power on Finer Grid")
        plt.xlabel("Radial Position (cm)")
        plt.ylabel("Height (cm)")
        plt.show()
        
        interpolated_values = griddata((position_df['radius'],position_df['height']), predicted, (R, Z), rescale=True)
        plt.figure()
        plt.imshow(interpolated_values,origin='lower',  extent=(min_r,max_r,min_z,max_z), aspect='auto')
        plt.colorbar() 
        plt.title(f"KNN Interpolated Power at Points")
        plt.xlabel("Radial Position (cm)")
        plt.ylabel("Height (cm)")
        plt.show()

    return pd.Series(predicted[:,0], name="interpolated_power")

def extract_ml_samples_from_training_step(step, directory, current_threshold, flux_threshold, xs_threshold, example_plot = True,
                                         pebble_kernel_volume=PEBBLE_KERNEL_VOLUME, pebble_surface_area=PEBBLE_SURFACE_AREA):
    current_energy_widths = np.diff(ENERGY_GRID_56)
    current_aux_features = ["local_fima", "local_graphite_frac"]
    det_name = f"gFHR_equilibrium_training_{step}.serpent_det0.m"
    aux_name = f"current_auxiliary_features{step}.csv"
    meshes, pebble_features, pebble_targets, zone_features = read_det_file(directory+det_name, 
                                                                           read_pebbles=True,
                                                                        meshes_to_read=["coarse_7group_flux",
                                                                                        "core_56group_flux",
                                                                                        "subzone_power"], 
                                                                        mesh_grids_rze = [(RADIUS_GRID_3, HEIGHT_GRID_8, ENERGY_GRID_7),
                                                                                          ([0.0, 120.0], [60.0, 369.47], ENERGY_GRID_56),
                                                                                          (RADIUS_GRID_8, HEIGHT_GRID_20, [0,20]) ] )

    if example_plot:
        plt.figure()
        plt.xlabel("Neutron Energy")
        plt.ylabel("Pebble Kernel Flux")
        plt.yscale("log")
        plt.xscale("log")
        for p in range(10):
            y = pebble_targets["peb_flux"].iloc[p]
            plt.step(ENERGY_GRID_56[1:], y/current_energy_widths/pebble_kernel_volume)
        plt.show()

        plt.figure()
        plt.xlabel("Neutron Energy")
        plt.ylabel("Pebble Surface Inward Current")
        plt.yscale("log")
        plt.xscale("log")
        for p in range(10):
            y = pebble_targets["peb_current"].iloc[p]
            plt.step(ENERGY_GRID_56[1:], y/current_energy_widths/pebble_surface_area)
        plt.show()
    
    aux_features = pd.read_csv(directory+aux_name, index_col=0)
    current_features = aux_features[current_aux_features].join(pebble_features)
    current_features['radius'] = round(np.sqrt(current_features['x']**2+current_features['y']**2),4)
    current_features['height'] = round(current_features['z'],4)
    current_features = current_features.drop(columns=['x','y','z','pebble_id'])
    flattened_current_features, flattened_current = flatten_pebble_data(current_features, pebble_targets["peb_current"]/current_energy_widths/pebble_surface_area)
    _, flattened_current_unc = flatten_pebble_data(current_features, pebble_targets["peb_current_unc"])

    core_flux_at_energy = []
    for i in range(len(flattened_current_features)):
        idx = (np.abs(ENERGY_GRID_56 - float(flattened_current_features["energy"].iloc[i]))).argmin()
        core_flux_at_energy += [meshes["core_56group_flux"]["data"][0,0,idx-1]]
    
    flattened_current_features["core_flux_at_energy"] = core_flux_at_energy

    interpolated_power = interpolate_core_power(flattened_current_features[["radius","height"]], 
                          meshes["subzone_power"]["data"], 
                          512,
                          do_plot=True)

    flattened_current_features["interpolated_power"] = interpolated_power
    
    interpolated_flux = interpolate_core_flux_3d(flattened_current_features[["radius","height","energy"]], 
                          meshes["coarse_7group_flux"]["data"], 
                          1024,
                          plot_energy=1.7650e-08)

    flattened_current_features["interpolated_flux"] = interpolated_flux

    xs_dict = extract_xs_from_mdx(f"{directory}/gFHR_equilibrium_training_{step}.serpent_mdx0.m")
    xs_inputs, xs_outputs = match_flux_to_xs_and_flatten(zone_features, pebble_targets["peb_flux"], xs_dict, unc_threshold=xs_threshold)
    
    conc_features = aux_features[aux_features.columns.difference(current_aux_features)]
    flattened_flux_features, flattened_flux = flatten_pebble_data(conc_features, pebble_targets["peb_flux"]/current_energy_widths/pebble_kernel_volume)
    _, flattened_flux_unc = flatten_pebble_data(conc_features, pebble_targets["peb_flux_unc"])
    flattened_flux_features["current"] = flattened_current

    precise_flux_indices = np.where((flattened_flux_unc<flux_threshold) & (flattened_flux_unc!=0.0))
    precise_current_indices = np.where((flattened_current_unc<current_threshold) & (flattened_current_unc!=0.0))
    
    results = (flattened_current_features.iloc[precise_current_indices].reset_index(drop=True), 
               pd.Series(flattened_current[precise_current_indices], name="current"), 
               flattened_flux_features.iloc[precise_flux_indices].reset_index(drop=True), 
               pd.Series(flattened_flux[precise_flux_indices], name="flux"),
               xs_inputs,
               xs_outputs)
    return results

def log_scale_data(features, log_scale_features, targets = None):
    if log_scale_features == "all":
        log_features = features.copy()
        log_features = np.log10(features+1e-100).copy()
    else:
        log_features = features.copy()
        log_features[log_scale_features] = np.log10(features[log_scale_features]+1e-100).copy()
    if targets is not None:
        log_targets = targets.copy()
        log_targets = np.log10(targets+1e-100)
        return log_features, log_targets
    else:
        return log_features




def create_forecast_dataset(features, targets, steps_ahead = 5, operating_parameters = 
                            ["power", "burnup_step", "graphite_insertion_fraction", 
                             "control_rod_position", "fima_discharge_threshold", 
                             "fima_discharge_rel_std"]):
    filtered_data = []
    filtered_targets = []
    last_row_index = 0
    match_counter = 0
    for index, row in features.iterrows():
        if index == 0:
            continue
        if features[operating_parameters].iloc[last_row_index].equals(row[operating_parameters]):
            match_counter += 1
            if match_counter >= steps_ahead:
                filtered_data += [features.iloc[last_row_index]]
                filtered_targets += [targets.iloc[index]]
                last_row_index += 1
            else:
                last_row_index = index
                match_counter = 0
    filtered_features = pd.concat(filtered_data).reset_index(drop=True)
    filtered_targets = pd.concat(filtered_targets).reset_index(drop=True)
    return filtered_features, filtered_targets

def create_training_split(features, targets, train_split, log_scale_features = [], log_scale_targets=True, 
                          data_mean = [], data_std = [], seed=42):
    np.random.seed(seed)

    if len(log_scale_features) > 0:
        if log_scale_targets:
            features, targets = log_scale_data(features, log_scale_features, targets)
        else:
            features = log_scale_data(features, log_scale_features)
    
    num_data = len(features)
    training_size = int(num_data*train_split)
    testing_size = num_data - training_size
    data_indices = np.arange(num_data)
    training_indices = np.random.choice(num_data, training_size, replace=False)
    testing_indices = data_indices[np.in1d(data_indices, training_indices, invert=True)]

    if len(data_mean) == 0 and len(data_std) == 0:
        training_data, data_mean, data_std = standardize(features.iloc[training_indices])
        testing_data, _, _  = standardize(features.iloc[testing_indices], mean=data_mean, std=data_std)
    else:
        training_data, _, _ = standardize(features.iloc[training_indices], mean=data_mean, std=data_std)
        testing_data, _, _  = standardize(features.iloc[testing_indices], mean=data_mean, std=data_std)
        
    training_target, target_mean, target_std = standardize(targets.iloc[training_indices])
        
    testing_target, _, _  = standardize(targets.iloc[testing_indices], mean=target_mean, std=target_std)
    data = {}
    data["training_features"] = training_data
    data["testing_features"] = testing_data
    data["feature_mean"] = data_mean
    data["feature_std"] = data_std
    data["training_target"] = training_target
    data["testing_target"] = testing_target
    data["target_mean"] = target_mean
    data["target_std"] = target_std
    return data