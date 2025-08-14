from .material import Material
import random
from .core import get_zone
import pickle
import pandas as pd
import numpy as np
from pearlsim.ml_utilities import *
from pearlsim.depletion_utilities import *
from pearlsim.ml_model_wrappers import SingleModelWrapper, XSModelWrapper
from pearlsim.isotope_rename_key import ISOTOPE_RENAME_KEY, ISOTOPE_RENAME_KEY_INVERTED
from pearlsim.group_definitions import *
from copy import deepcopy
import json
#import openmc.deplete
openmc = None
from .get_fima import get_fima
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import time
import multiprocessing as mp

CRAM_func = np.sin #openmc.deplete.cram.CRAM48

class Pebble():
    def __init__(self, x, y, z, material_source, id=random.randint(1,2147483647)):
        self.x = x
        self.y = y
        self.r = np.sqrt(x**2+y**2)
        self.z = z
        self.last_x = 0
        self.last_y = 0
        self.last_r = 0
        self.last_z = 0
        self.fima = 0
        self.id = id
        self.pass_num = 1
        self.intial_fuel = material_source
        self.in_core = True
        self.material = Material(f"pebble{id}", material_source)
        if "92235<lib>" in self.material.concentrations:
            self.is_fuel = True
        else:
            self.is_fuel = False

    def reinsert(self, distribution):
        x_val, y_val, z_val = distribution(1)
        self.x = x_val
        self.y = y_val
        self.z = 60
        self.pass_num += 1
        self.in_core = True
        self.r = np.sqrt(self.x**2 + self.y**2)


class Pebble_Model():
    def __init__(self):
        self.pebbles = []
        self.num_pebbles = 0
        self.num_cores = 1
        self.material_counter = -1
        self.max_z = 369.47
        self.debug = 0
        self.flux_num_neighbors = 9*9*4
        self.feature_search_radius = 20.0
        self.insertion_distribution = gFHR_insertion_distribution
        self.fresh_materials = {}
        self.axial_zone_bounds = []
        self.radial_bound_points = []
        self.discharge_indices = []
        self.discard_indices = []
        self.reinsert_indices = []
        self.velocity_profile = []
        self.current_library = ""
        self.initial_actinides = 0.00466953+0.0189728 # u235 x u238
        self.burnup_library = ""
        self.current_model = None
        self.flux_model = None
        self.xs_models = None
        self.reaction_keys = []
        self.split_pebble_indices = []
        self.flux_nuclide_labels = []
        

    def load_velocity_model(self, velocity_data_path):
        """
        Read velocity profile from file
        Format is starting z value for bin followed by either a constant A
        or a function of the form z = A*r + B
        (Units are cm / s)
        """
        self.velocity_profile = []
        with open(velocity_data_path, 'r') as f:
            for line in f:
                if line[0] == "#":
                    continue
                line = line.split()

                # Constant velocity provided
                if len(line) == 2:
                    self.velocity_profile += [ (float(line[0]),
                                               lambda r : float(line[1]) ) ]
                # Linear function provided
                elif len(line) == 3:
                    self.velocity_profile += [(float(line[0]),
                                                lambda r: float(line[1])*r+float(line[2]) )]

    def load_current_model(self, current_model_path):
        with open(current_model_path, 'rb') as f:
            self.current_model = pickle.loads(f.read())
        self.current_model.model.n_jobs = -1

    def load_flux_model(self, flux_model_path):
        with open(flux_model_path, 'rb') as f:
            self.flux_model = pickle.loads(f.read())
        self.flux_model.model.n_jobs = -1
        self.flux_nuclide_labels = list(self.flux_model.input_column_labels)
        self.flux_nuclide_labels.remove("energy")
        self.flux_nuclide_labels.remove("current")
            
    def load_xs_model(self, xs_model_path):
        with open(xs_model_path, 'rb') as f:
            self.xs_models = pickle.loads(f.read())
        self.xs_models.flux_df = pd.DataFrame()
        self.reaction_keys = self.xs_models.reaction_keys
        for key in self.reaction_keys:
            self.xs_models.model_dict[key].n_jobs = -1


    def get_velocity(self, z, r):
        for i in reversed(range(len(self.velocity_profile))):
            if z >= self.velocity_profile[i][0]:
                return self.velocity_profile[i][1](r)

    def run_xs_model_batch(self, reaction_key):
        if self.debug > 0:
            print(f"Predicting {reaction_key}")
        return self.xs_models.predict(self.xs_models.flux_df, reaction_key)


    def update_model(self, iteration, simulation, burn_day_step, insertion_ratios, threshold, debug):
        self.debug = debug
        time_step = burn_day_step*86400

        # Set up some constants / parallelization variables
        self.samples_to_run = np.arange(self.num_pebbles)             
        self.split_pebble_indices = np.array_split( self.samples_to_run, self.num_cores)
        self.num_energies = len(ENERGY_GRID_56)-1
        openmc.deplete.pool.USE_MULTIPROCESSING = False
        
        # Track removal of pebble types for feedback into zone model
        pebbles_modeled_by_zone = {}
        pebbles_removed_by_zone = {}
        
        # Track pebbles indices that are being discarded and discharged accordingly
        discharge_data_index = 0
        discharge_data = {}
        discard_data_index = 0
        discard_data = {}

        ######
        # First burnup step (n0)
        ######

        simulation.load_from_step(iteration, 0, debug=debug)
        zone_model_positions = simulation.get_zone_model_pebble_locations()
        zone_model_materials = simulation.get_zone_model_materials()



        print(f"Step 1.1) Collecting position, local FIMA/graphite fractions for {self.num_pebbles} pebbles")
        time_start = time.time()

        current_data, pebble_conc_0_df = self.collect_pebble_data(zone_model_positions, 
                                                zone_model_materials)
        
        time_taken = time.time() - time_start
        print(f"Completed in {time_taken}s")


        if debug > 0:
            individual_u235_values = pebble_conc_0_df['92235<lib>']
            zone_u235_values = []
            zone_fima_values = []

            for index, row in zone_model_positions.iterrows():
                mat_key = row["material"]
                zone_u235_values += [zone_model_materials[mat_key].concentrations['92235<lib>']]  
                zone_fima_values += [get_fima(zone_model_materials[mat_key].concentrations, self.initial_actinides)]  

            zone_u235_values = np.array(zone_u235_values)
            num_bins = 1+int(np.cbrt(self.num_pebbles)*2)

            peb_bin_counts, u_bin_edges = np.histogram(individual_u235_values, bins=num_bins)
            fig, ax1 = plt.subplots()
            ax1.hist(individual_u235_values, u_bin_edges, label="Individual", color="lightblue")
            ax1.set_xlabel("U-235 Concentration")
            ax1.set_ylabel("Number of individual pebbles")

            ax2 = ax1.twinx()
            ax2.hist(zone_u235_values, u_bin_edges, label="Zone", 
                     facecolor="none", hatch="//", edgecolor="black")
            ax2.set_ylabel("Number of zone pebbles")
            #plt.vlines(u235_zone_concentrations, ymin = 0, ymax=scaled_zone_counts, colors="orange")
            plt.savefig(f"u235_distributions_preburn_{iteration}.png") 

            peb_fima_values = []
    
            for p in range(self.num_pebbles):
                peb = self.pebbles[p] 
                if peb.in_core and peb.is_fuel:
                    peb_fima_values += [peb.fima] 

            fima_bin_edges = np.linspace(0,20,num_bins+1)
            
            peb_fima_values = np.array(peb_fima_values)
            #peb_bin_counts, fima_bin_edges = np.histogram(peb_fima_values, bins=num_bins)

            fig, ax1 = plt.subplots()
            ax1.hist(peb_fima_values, fima_bin_edges, label="Individual", color="lightblue")
            ax1.set_xlabel("%FIMA")
            ax1.set_xlim(0,20)
            ax1.set_ylabel("Number of individual pebbles")

            ax2 = ax1.twinx()
            ax2.hist(zone_fima_values, fima_bin_edges, label="Zone", 
                     facecolor="none", hatch="//", edgecolor="black")
            ax2.set_ylabel("Number of zone pebbles")
            ax2.set_xlim(0,20)
            #plt.vlines(u235_zone_concentrations, ymin = 0, ymax=scaled_zone_counts, colors="orange")
            plt.savefig(f"fima_distributions_preburn_{iteration}.png") 
        
        print(f"Step 2.1) Collect data from zone model flux/core meshes...")
        time_start = time.time()
        
        current_features = self.collect_mesh_based_data(iteration,
                                                        0, 
                                                        current_data, 
                                                        zone_model_positions, 
                                                        zone_model_materials)
            
        if self.debug > 0:
            plt.figure()
            plt.scatter(current_data["radius"], current_data["height"])
            plt.title(f"Pebble positions Step = {iteration}")
            plt.xlabel("Radial Position (cm)")
            plt.ylabel("Height (cm)")
            plt.savefig(f"discrete_pebble_positions_{iteration}.png")  

        del(current_data)
        
        time_taken = time.time() - time_start
        print(f"Completed in {time_taken}s")

        
        print(f"Step 3.1) Predicting pebblewise current and flux using ML models for {self.num_pebbles} pebbles...")
        time_start = time.time()

        flux_df = self.run_flux_prediction(current_features, pebble_conc_0_df)
        del(current_features)
        
        time_taken = time.time() - time_start
        print(f"Completed in {time_taken}s using 1 core")

        
        print(f"Step 4.1) Running one group cross sections for all transmutation reactions for {self.num_pebbles} pebbles...")
        time_start = time.time()
        
        xs_predicted_0_df, total_flux_0_series = self.run_xs_prediction(flux_df)
        del(flux_df)
        
        time_taken = time.time() - time_start
        print(f"Completed in {time_taken}s")


        print(f"Step 5.1) Running OpenMC depletion {self.num_pebbles} pebbles...")
        time_start = time.time()

        pebble_conc_1_df = self.run_openmc_depletion(pebble_conc_0_df, xs_predicted_0_df, total_flux_0_series)

        time_taken = time.time() - time_start
        print(f"Completed in {time_taken}s using {self.num_cores} core")


        ######
        # Second burnup step
        ######
        num_zone_substeps = len(simulation.core.substep_schema)
        simulation.load_from_step(iteration, num_zone_substeps, debug=debug)
        zone_model_positions = simulation.get_zone_model_pebble_locations()
        zone_model_materials = simulation.get_zone_model_materials()

        print(f"Step 1.2) Collecting position, local FIMA/graphite fractions for {self.num_pebbles} pebbles")
        time_start = time.time()

        current_data, _ = self.collect_pebble_data(zone_model_positions, 
                                                zone_model_materials)
        
        time_taken = time.time() - time_start
        print(f"Completed in {time_taken}s")

        
        print(f"Step 2.2) Collect data from zone model flux/core meshes...")
        time_start = time.time()
        
        current_features = self.collect_mesh_based_data(iteration,
                                                        num_zone_substeps, 
                                                        current_data, 
                                                        zone_model_positions, 
                                                        zone_model_materials)
             

        del(current_data)
        
        time_taken = time.time() - time_start
        print(f"Completed in {time_taken}s")


        print(f"Step 3.2) Predicting pebblewise current and flux using ML models for {self.num_pebbles} pebbles...")
        time_start = time.time()

        flux_df = self.run_flux_prediction(current_features, pebble_conc_1_df)
        del(current_features)
        
        time_taken = time.time() - time_start
        print(f"Completed in {time_taken}s using 1 core")

        
        print(f"Step 4.2) Running one group cross sections for all transmutation reactions for {self.num_pebbles} pebbles...")
        time_start = time.time()
        
        xs_predicted_1_df, total_flux_1_series = self.run_xs_prediction(flux_df)
        del(flux_df)
        
        time_taken = time.time() - time_start
        print(f"Completed in {time_taken}s")


        print(f"Step 5.2) Running OpenMC depletion {self.num_pebbles} pebbles...")
        time_start = time.time()

        total_flux_avg_series = total_flux_0_series/2 + total_flux_1_series/2
        xs_predicted_avg_df = xs_predicted_1_df/2 + xs_predicted_0_df/2

        pebble_conc_f_df = self.run_openmc_depletion(pebble_conc_0_df, xs_predicted_avg_df, total_flux_avg_series)

        time_taken = time.time() - time_start
        print(f"Completed in {time_taken}s using {self.num_cores} core")


        if debug > 0:
            individual_u235_values = pebble_conc_f_df['92235<lib>']
            zone_u235_values = []

            zone_fima_values = []

            for index, row in zone_model_positions.iterrows():
                mat_key = row["material"]
                zone_u235_values += [zone_model_materials[mat_key].concentrations['92235<lib>']] 
                zone_fima_values += [get_fima(zone_model_materials[mat_key].concentrations, self.initial_actinides)]
            


            zone_u235_values = np.array(zone_u235_values)
            
            #num_bins = 1+int(np.cbrt(self.num_pebbles)*2)
            #peb_bin_counts, bin_edges = np.histogram(individual_u235_values, bins=num_bins)

            fig, ax1 = plt.subplots()
            ax1.hist(individual_u235_values, u_bin_edges, label="Individual", color="lightblue")
            ax1.set_xlabel("U-235 Concentration")
            ax1.set_ylabel("Number of individual pebbles")

            ax2 = ax1.twinx()
            ax2.hist(zone_u235_values, u_bin_edges, label="Zone", 
                     facecolor="none", hatch="//", edgecolor="black")
            ax2.set_ylabel("Number of zone pebbles")
            #plt.vlines(u235_zone_concentrations, ymin = 0, ymax=scaled_zone_counts, colors="orange")
            plt.savefig(f"u235_distributions_postburn_{iteration}.png") 

            
        peb_fima_values = []

        print("Updating fuel pebble concentrations...")
        for p in range(self.num_pebbles):
            peb = self.pebbles[p] 
            if peb.in_core and peb.is_fuel:
                peb.material.concentrations = pebble_conc_f_df.iloc[p].to_dict()
                peb.fima = get_fima(peb.material.concentrations, self.initial_actinides)
                peb_fima_values += [peb.fima] 
        del(pebble_conc_f_df)

        if debug > 0:
            peb_fima_values = np.array(peb_fima_values)
            #peb_bin_counts, bin_edges = np.histogram(peb_fima_values, bins=num_bins)

            fig, ax1 = plt.subplots()
            ax1.hist(peb_fima_values, fima_bin_edges, label="Individual", color="lightblue")
            ax1.set_xlabel("%FIMA")
            ax1.set_ylabel("Number of individual pebbles")
            ax1.set_xlim(0,20)
            
            ax2 = ax1.twinx()
            ax2.hist(zone_fima_values, fima_bin_edges, label="Zone", 
                     facecolor="none", hatch="//", edgecolor="black")
            ax2.set_ylabel("Number of zone pebbles")
            ax2.set_xlim(0,20)
            #plt.vlines(u235_zone_concentrations, ymin = 0, ymax=scaled_zone_counts, colors="orange")
            plt.savefig(f"fima_distributions_postburn_{iteration}.png") 

        # After the burnup, move the pebbles
        for p in range(self.num_pebbles):
            peb = self.pebbles[p] 
            vz = self.get_velocity(peb.z, peb.r)
            peb.z = peb.z + vz*time_step
            if debug > 2:
                print(f"Moving pebble {p} from x={peb.last_x},y={peb.last_y},z={peb.last_z} to x={peb.x},y={peb.y},z={peb.z}") 

            if peb.z > self.max_z:
                if debug > 1:
                    print(f"Flagged discharge pebble {p}: Movement from {peb.last_z} to {peb.z}")
                self.discharge_indices += [p]
                peb.in_core = False

        if debug > 0:
            print("Removing pebbles above threshold from top zone...")
        pebbles_modeled_by_zone, pebbles_removed_by_zone = self.remove_spent_pebbles(threshold,
                                                                                     pebbles_modeled_by_zone,
                                                                                     pebbles_removed_by_zone,
                                                                                     True)
        if debug > 0:
            print("Writing discharged and discarded pebble data...")
            
        for p in self.discharge_indices:
            peb = self.pebbles[p]
            # Skip graphite pebbles.
            if not peb.is_fuel:
                continue
            if p in self.discard_indices:
                discard_data[discard_data_index] = {
                    "features": {
                        "radius": peb.r,
                        "id": p,
                        "pass": peb.pass_num
                    },
                    "concentration": peb.material.concentrations
                }
                discard_data_index += 1

            discharge_data[discharge_data_index] = {
                "features": {
                    "radius": peb.r,
                    "id": p,
                    "pass": peb.pass_num
                },
                "concentration": peb.material.concentrations
            }
            discharge_data_index += 1

        with open(f"discharge_pebbles_{iteration}.json", 'w') as file:
            json.dump(discharge_data, file, indent=2)
        with open(f"discard_pebbles_{iteration}.json", 'w') as file:
            json.dump(discard_data, file, indent=2)
        del(discharge_data)
        del(discard_data)

        # Reinsert non-discarded pebbles and replace discarded pebbles with fresh ones, all at the bottom
        self.reinsert_pebbles(insertion_ratios, time_step, debug)

        # Calculate fractions of pebbles to remove from the zone model
        removal_fractions = {}
        for key in pebbles_modeled_by_zone.keys():
            if key in pebbles_removed_by_zone.keys():
                removal_fractions[key] = pebbles_removed_by_zone[key]/pebbles_modeled_by_zone[key]
            else:
                removal_fractions[key] = 0
        return removal_fractions

    def initialize_pebbles(self, num_pebbles, pebble_points, initial_inventory, debug=0):
        self.pebbles = [None for _ in range(num_pebbles)]
        self.num_pebbles = num_pebbles
        indices = np.random.choice(range(len(pebble_points)), size=num_pebbles, replace=False)
        x_vals = pebble_points.iloc[indices]['x'].values
        y_vals = pebble_points.iloc[indices]['y'].values
        z_vals = pebble_points.iloc[indices]['z'].values
        if debug > 0:
            print(f"Generating {num_pebbles} pebbles for individual pebble model.")
        for i in range(num_pebbles):
            x = x_vals[i]
            y = y_vals[i]
            z = z_vals[i]
            mat_name = random.choices(list(initial_inventory.keys()), weights=initial_inventory.values(), k=1)[0]
            if debug > 1:
                print(f"Generating {mat_name} pebble at x = {x}, y = {y}, z = {z}")
            self.pebbles[i] = Pebble(x, y, z, self.fresh_materials[mat_name].concentrations.copy())
    
    def sample_pebbles(self, num_pebbles, pebble_points, core_materials, debug=0):
        self.pebbles = [None for _ in range(num_pebbles)]
        self.num_pebbles = num_pebbles
        indices = np.random.choice(range(len(pebble_points)), size=num_pebbles, replace=False)
        x_vals = pebble_points.iloc[indices]['x'].values
        y_vals = pebble_points.iloc[indices]['y'].values
        z_vals = pebble_points.iloc[indices]['z'].values
        materials_names = pebble_points.iloc[indices]['material'].values
        if debug > 0:
            print(f"Generating {num_pebbles} pebbles for individual pebble model.")
        for i in range(num_pebbles):
            x = x_vals[i]
            y = y_vals[i]
            z = z_vals[i]
            mat_name = materials_names[i]
            mat_data = core_materials[mat_name].concentrations.copy()
            self.pebbles[i] = Pebble(x, y, z, mat_data)
            self.pebbles[i].fima = get_fima(mat_data, self.initial_actinides)
            if debug > 1:
                print(f"Generating {mat_name} pebble at {x} {y} {z}.")


    def remove_spent_pebbles(self, threshold, pebbles_modeled_by_zone, pebbles_removed_by_zone, graphite_removal_flag):
        """
        Check all invividual pebbles in top zones to see what fraction of them should be removed.
        :param radial_bound_points:
        :param axial_zone_bounds:
        :param threshold:
        :return:
        """
        self.discarded_indices = []
        for p in self.discharge_indices:
            peb = self.pebbles[p]
            # Get the previous zone of the freshly discharged pebble
            rad_zone, axial_zone = get_zone(peb.r, peb.last_z, self.radial_bound_points, self.axial_zone_bounds)

            # Relate this pebble to its corresponding zone material
            zone_material_name = f"{peb.intial_fuel}_R{rad_zone}Z{axial_zone}P{peb.pass_num}"
            pebbles_modeled_by_zone[zone_material_name] = \
                1+pebbles_modeled_by_zone.get(zone_material_name, 0)

            # Check if pebble's cesium concentration is above threshold
            # If so, mark it for discard and count it for the zone model
            fima = get_fima(peb.material.concentrations, self.initial_actinides)
            peb.fima = fima
            if p % 200 == 0:
                print(fima)
            if fima > threshold:
                pebbles_removed_by_zone[zone_material_name] = \
                    1 + pebbles_removed_by_zone.get(zone_material_name, 0)
                self.discard_indices += [p]

            # Check to see if the pebble is graphite and remove it without tracking it
            elif not peb.is_fuel:
                self.discard_indices += [p]
            # Otherwise mark the pebble to be reinserted
            else:
                self.reinsert_indices += [p]
        return pebbles_removed_by_zone, pebbles_modeled_by_zone

    def reinsert_pebbles(self, insertion_ratios, time_step, debug):
        for i in self.reinsert_indices:
            peb = self.pebbles[i]
            peb.reinsert(self.insertion_distribution)
            vz = self.get_velocity(0, peb.r)
            peb.z += np.random.uniform(0,1)*vz * time_step
            if debug > 1:
                print(f"Reinserting pebble {peb.id} at x = {peb.x}, y = {peb.y}, z = {peb.z} (pass {peb.pass_num})")
        self.reinsert_indices = []
        self.discharge_indices = []
        for i in self.discard_indices:
            x,y,z = self.insertion_distribution(1)
            mat_name = random.choices(list(insertion_ratios.keys()), weights=insertion_ratios.values(), k=1)[0]
            material = self.fresh_materials[mat_name].concentrations.copy()
            self.pebbles[i] = Pebble(x, y, z, material)
            vz = self.get_velocity(0, self.pebbles[i].r)
            self.pebbles[i].z = np.random.uniform(0, 1) * vz * time_step
            if debug > 1:
                print(f"Regenerating discarded pebble {peb.id} as {material.name} at x = {x}, y = {y}, z = {self.pebbles[i].z}")
        self.discard_indices = []





    def collect_pebble_data(self, zone_model_positions, zone_model_materials):
        
        pebble_batch_list = []
        
        for process_num in range(self.num_cores):
            indices = self.split_pebble_indices[process_num]
            data_batch = {}
            data_batch["pebbles"] = [self.pebbles[i] for i in indices]
            data_batch["zone_model_positions"] = zone_model_positions
            data_batch["zone_model_materials"] = zone_model_materials
            data_batch["feature_search_radius"] = self.feature_search_radius
            data_batch["initial_actinides"] = self.initial_actinides
            data_batch["indices"] = indices
            data_batch["process"] = process_num
            pebble_batch_list += [data_batch]

        with mp.Pool(self.num_cores) as pebble_pool:
            pebble_data_chunked = pebble_pool.map(collect_pebble_data_helper, pebble_batch_list)

        # Initialize arrays for ML input dataframes
        radius_array = np.array([])
        height_array = np.array([])
        local_graphite_fraction_array = np.array([])
        local_fima_array = np.array([])
        list_of_conc_dicts = []
        for results_chunk in pebble_data_chunked:
            radius_array = np.hstack([radius_array, results_chunk["radius"]])
            height_array = np.hstack([height_array, results_chunk["height"]])
            local_graphite_fraction_array = np.hstack([local_graphite_fraction_array, 
                                                      results_chunk["local_graphite_fraction"]])
            local_fima_array = np.hstack([local_fima_array,
                                         results_chunk["local_fima"]])
            list_of_conc_dicts += results_chunk["conc"]
    
        
        pebble_conc_df = pd.DataFrame(list_of_conc_dicts)

        current_data = pd.DataFrame({
            "radius": radius_array,
            "height": height_array,
            "local_fima": local_fima_array,
            "local_graphite_frac": local_graphite_fraction_array,
        })
        
        del(radius_array)
        del(height_array)
        del(local_fima_array)
        del(local_graphite_fraction_array)

        return current_data, pebble_conc_df


    def collect_mesh_based_data(self, iteration, sub_step, current_data, zone_model_positions, zone_model_materials):
        det_file_name = f"gFHR_equilibrium_{iteration}.serpent_det{sub_step}.m"
        meshes = read_det_file(det_file_name, 
                               read_pebbles=False,
                               meshes_to_read=
                                       ["coarse_7group_flux",
                                        "core_56group_flux",
                                        "subzone_power"], 
                               mesh_grids_rze = [(RADIUS_GRID_3, HEIGHT_GRID_8, ENERGY_GRID_7),
                                                  ([0.0, 120.0], [60.0, 369.47], ENERGY_GRID_56),
                                                  (RADIUS_GRID_8, HEIGHT_GRID_20, [0,20]) ] )
        
        current_features = expand_pebble_features_by_energy(current_data)
        core_flux_at_energy = []
        for i in range(len(current_features)):
            idx = (np.abs(ENERGY_GRID_56 - float(current_features["energy"].iloc[i]))).argmin()
            core_flux_at_energy += [meshes["core_56group_flux"]["data"][0,0,idx-1]]
        
        interp_batch_list = []
        
        for process_num in range(self.num_cores):
            peb_indices = self.split_pebble_indices[process_num]
            feature_indices = np.arange(peb_indices[0]*self.num_energies, (peb_indices[-1]+1)*self.num_energies)
            data_batch = {}
            data_batch["features"] = current_features.iloc[feature_indices]
            data_batch["power_mesh"] = meshes["subzone_power"]["data"]
            data_batch["flux_mesh"] = meshes["coarse_7group_flux"]["data"]
            data_batch["zone_model_positions"] = zone_model_positions
            data_batch["zone_model_materials"] = zone_model_materials
            data_batch["power_num_neighbors"] = 512
            data_batch["flux_num_neighbors"] = 1024
            data_batch["points_per_cm"] = 1
            data_batch["flux_radius_grid"] = RADIUS_GRID_3
            data_batch["flux_height_grid"] = HEIGHT_GRID_8
            data_batch["power_radius_grid"] = RADIUS_GRID_8
            data_batch["power_height_grid"] = HEIGHT_GRID_20
            data_batch["energy_grid"] = ENERGY_GRID_7
            data_batch["plot_energy"] = 1.7650e-08
            data_batch["interp_energy_grid"] = ENERGY_CENTERS_56
            data_batch["do_plot"] = True
            data_batch["process_num"] = process_num
            interp_batch_list += [data_batch]
        
        with mp.Pool(self.num_cores) as interp_pool:
            interp_power_chunked = interp_pool.map(interpolate_core_power, interp_batch_list)
            interp_flux_chunked = interp_pool.map(interpolate_core_flux_3d, interp_batch_list)

        current_features["interpolated_power"] = np.concatenate(interp_power_chunked)
        current_features["interpolated_flux"] = np.concatenate(interp_flux_chunked)
        current_features["core_flux_at_energy"] = core_flux_at_energy
        del(core_flux_at_energy)
        del(interp_flux_chunked)
        del(interp_power_chunked)
        return current_features

    
    def run_flux_prediction(self, current_features, pebble_conc_df):
        predicted_flux_series_list = [[] for _ in range(self.num_pebbles)]

        current_df = self.current_model.predict(current_features)
        flux_feature_chunk_list = []
        for i in range(self.num_pebbles):
            start_ind = i*self.num_energies
            end_ind = (i+1)*self.num_energies

            conc_series = pebble_conc_df[self.flux_nuclide_labels].iloc[i]
            
            flux_feature_chunk = pd.concat([conc_series] * self.num_energies, axis=1).T
            flux_feature_chunk["energy"] = current_features["energy"].iloc[start_ind:end_ind].values
            flux_feature_chunk["current"] = current_df.iloc[start_ind:end_ind].values
            flux_feature_chunk_list += [flux_feature_chunk]
            
        flux_features = pd.concat(flux_feature_chunk_list).reset_index(drop=True)
        flux = self.flux_model.predict(flux_features)
    
        predicted_flux_series_list = []
        for i in range(self.num_pebbles):
            start_ind = i*self.num_energies
            end_ind = (i+1)*self.num_energies
            predicted_flux_series_list += [flux.iloc[start_ind:end_ind].reset_index(drop=True)]

        flux_df = pd.DataFrame(predicted_flux_series_list)
        flux_df = flux_df*np.diff(ENERGY_GRID_56)*PEBBLE_KERNEL_VOLUME
        flux_df.columns=self.xs_models.input_column_labels
        return flux_df

    def run_xs_prediction(self, flux_df):
        predicted_xs_list = []
        for reaction_key in self.reaction_keys:
            predicted_xs_list += [self.xs_models.predict(flux_df, reaction_key)]

        xs_predicted_df = pd.concat(predicted_xs_list, axis=1)
        total_flux_series = xs_predicted_df.pop("total_flux")
        return xs_predicted_df, total_flux_series


    def run_openmc_depletion(self, pebble_conc_df, xs_predicted_df, total_flux_series):
        nuclide_df, rename_map_nuclides = nuclide_labels_serpent_to_openMC(pebble_conc_df)
        rename_map_nuclides_inv = {v: k for k, v in rename_map_nuclides.items()}
        nuclide_df = nuclide_df.fillna(0.0)

        xs_df, rename_map_xs = nuclide_labels_serpent_to_openMC(xs_predicted_df)
        
        replace_keys = {"Am342":"Am242_m1", "Am344":"Am244_m1"}
        drop_keys = ["Ag310", "Pm348", "Gd153_m1"]

        chain_file = "/global/home/users/ikolaja/openmc_data_old/chain_endfb71_pwr.xml"
        chain = openmc.deplete.Chain.from_xml(chain_file)
        xs_lib = openmc.data.DataLibrary.from_xml("/global/home/groups/co_nuclear/openmc-lib/endfb71_hdf5/cross_sections.xml")
        MTS = [102, 16, 103, 107, 17, 37, 18]
        nuclide_df = nuclide_df.rename(columns=replace_keys).drop(columns=drop_keys, errors="ignore")
        global serpent_chain
        serpent_chain = chain.reduce(list(nuclide_df.columns),0)
        serpent_chain.export_to_xml("pearlsim_chain.xml")


        microXS_list = create_MicroXS_batch(xs_df, chain, nuclide_df.columns.to_list())

        material_list = []
        total_flux_list = []
        material_xs_list = []
        total_atom_list = []
        n_0_conditions = []
        n_f_results = []

        for i in self.samples_to_run:
            self.material_counter += 1
            nuclides = nuclide_df.iloc[i].to_dict()
            total_flux_list += [total_flux_series[i]]
            total_atom_list += [sum(nuclides.values())]
            material_xs_list += [microXS_list[i]]
        
            openmc_material = openmc.Material(material_id=self.material_counter)
            openmc_material.add_components(nuclides, "ao")
            openmc_material.set_density('atom/b-cm', total_atom_list[-1])
            openmc_material.volume = PEBBLE_KERNEL_VOLUME
            material_list += [openmc_material]

        data_batch_list = []
        
        for process_num in range(self.num_cores):
            indices = self.split_pebble_indices[process_num]
            data_batch = {}
            data_batch["materials"] = openmc.Materials()
            for i in indices:
                data_batch["materials"].append(material_list[i])
            data_batch["total_flux"] = [total_flux_list[i] for i in indices]
            data_batch["xs"] = [material_xs_list[i] for i in indices]
            data_batch["indices"] = indices
            data_batch["process"] = process_num
            data_batch_list += [data_batch]

        with mp.Pool(self.num_cores) as depletion_pool:
            n_f_results_chunked = depletion_pool.map(run_openMC_depletion_batch, data_batch_list)
            
        # n_f_results: list of num_pebbles length, each entry is another list with isotopes after depletion
        n_f_results = [item for sublist in n_f_results_chunked for item in sublist[0]]
        
        conc_post_burn_series_list = []
        for i in range(self.num_pebbles):
            conc_post_burn_series_list += [pd.Series(n_f_results[i], index=serpent_chain.nuclide_dict.keys())]
        conc_post_burn_df = pd.DataFrame(conc_post_burn_series_list)*1e-24/PEBBLE_KERNEL_VOLUME
        conc_post_burn_df[conc_post_burn_df<0] = 0.0
        replace_keys_inv = {"Am242_m1":"Am342", "Am244_m1":"Am344"}
        conc_post_burn_df = conc_post_burn_df.rename(columns=replace_keys_inv)
        conc_post_burn_df = conc_post_burn_df.rename(columns=rename_map_nuclides_inv)
        return conc_post_burn_df


def run_openMC_depletion_batch(data_batch):
    depleter = openmc.deplete.IndependentOperator(data_batch["materials"], 
                                                  data_batch["total_flux"],
                                                  data_batch["xs"], 
                                                  chain_file="pearlsim_chain.xml",
                                                  normalization_mode="source-rate")
    n_0 = depleter.initial_condition()
    res = depleter(n_0, 1)
    rates = res.rates
    n_f_results = [openmc.deplete.pool.deplete(CRAM_func, serpent_chain, n_0, rates, 6.525*86400)]
    return n_f_results

def gFHR_insertion_distribution(num_pebbles):
    if num_pebbles > 1:
        angles = np.random.uniform(low=0, high=2*np.pi, size=num_pebbles)
        r_vals = 118*np.random.power(2.5, num_pebbles)
        x_vals = r_vals * np.cos(angles)
        y_vals = r_vals * np.sin(angles)
        z_vals = np.full(num_pebbles, 60)
    else:
        angle = np.random.uniform(low=0, high=2 * np.pi, size=num_pebbles)[0]
        r_val = 118*np.random.power(2.5, num_pebbles)[0]
        x_vals = r_val * np.cos(angle)
        y_vals = r_val * np.sin(angle)
        z_vals = np.full(num_pebbles, 60)[0]
    return x_vals, y_vals, z_vals


def collect_pebble_data_helper(data_batch):
    num_pebbles = len(data_batch["pebbles"])

    # Initialize sub arrays for data
    radius_array = np.zeros(num_pebbles)
    height_array = np.zeros(num_pebbles)
    list_of_local_graphite_fraction = np.zeros(num_pebbles)
    list_of_local_fima = np.zeros(num_pebbles)
    list_of_conc_dicts = [[] for _ in range(num_pebbles)]
    
    for p in range(num_pebbles):
        peb = data_batch["pebbles"][p]
        radius_array[p] = peb.r
        height_array[p] = peb.z
        peb.last_x, peb.last_y, peb.last_r, peb.last_z = peb.x, peb.y, peb.r, peb.z
        
        list_of_conc_dicts[p] = peb.material.concentrations
        
        # Collect local values
        local_indices = []
        
        xsq = (peb.x - data_batch["zone_model_positions"]['x'])**2
        ysq = (peb.y - data_batch["zone_model_positions"]['y'])**2
        zsq = (peb.z - data_batch["zone_model_positions"]['z'])**2
        dist = np.sqrt(xsq+ysq+zsq)
        local_indices = list(dist[dist<data_batch["feature_search_radius"]].index)
        
        fima_values = []
        is_graphite = []
        for ind in local_indices:
            data = data_batch["zone_model_positions"].iloc[ind]
            material_id = data['material']
            if "fuel" in material_id:
                material_conc = data_batch["zone_model_materials"][material_id].concentrations
                fima_values += [ get_fima(material_conc, data_batch["initial_actinides"]) ]
                is_graphite += [0]
            else:
                is_graphite += [1]
        list_of_local_fima[p] = np.mean(fima_values)
        list_of_local_graphite_fraction[p] = np.mean(is_graphite)
    results = {"radius":radius_array,
               "height":height_array,
               "local_graphite_fraction":list_of_local_graphite_fraction,
               "local_fima":list_of_local_fima,
               "conc":list_of_conc_dicts}
    return results