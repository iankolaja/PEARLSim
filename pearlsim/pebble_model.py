from .material import Material
import random
from .core import get_zone
import pickle
import pandas as pd
import numpy as np
from pearlsim.ml_utilities import standardize, unstandardize
from copy import deepcopy
import json

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
        self.max_z = 369.47
        self.insertion_distribution = gFHR_insertion_distribution
        self.fresh_materials = {}
        self.axial_zone_bounds = []
        self.radial_bound_points = []
        self.discharge_indices = []
        self.discard_indices = []
        self.reinsert_indices = []
        self.velocity_profile = []
        self.current_library = ""
        self.current_data_mean = []
        self.current_data_std = []
        self.current_target_mean = []
        self.current_target_std = []
        self.burnup_library = ""
        self.burnup_data_mean = []
        self.burnup_data_std = []
        self.burnup_target_mean = []
        self.burnup_target_std = []

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

    def load_current_model(self, current_model_path, num_cores,
                 current_data_mean_path = None,
                 current_data_std_path = None,
                 current_target_mean_path = None,
                 current_target_std_path = None,
                 library = "sklearn"):
        if current_data_mean_path is None:
            current_data_mean_path = current_model_path.replace(".pkl","") + "_data_mean.csv"
        if current_data_std_path is None:
            current_data_std_path = current_model_path.replace(".pkl","") + "_data_std.csv"
        if current_target_mean_path is None:
            current_target_mean_path = current_model_path.replace(".pkl","") + "_target_mean.csv"
        if current_target_std_path is None:
            current_target_std_path = current_model_path.replace(".pkl","") + "_target_std.csv"

        with open(current_model_path, 'rb') as f:
            self.current_model = pickle.loads(f.read())
        if library == "sklearn":
            self.current_model.set_params(**{"n_jobs":num_cores*2})
        self.current_library = library

        self.current_data_mean = pd.read_csv(current_data_mean_path, index_col=0).iloc[:,0]
        self.current_data_std = pd.read_csv(current_data_std_path, index_col=0).iloc[:,0]
        self.current_target_mean = pd.read_csv(current_target_mean_path, index_col=0).iloc[:,0]
        self.current_target_std = pd.read_csv(current_target_std_path, index_col=0).iloc[:,0]


    def load_burnup_model(self, burnup_model_path, num_cores,
                 burnup_data_mean_path = None,
                 burnup_data_std_path = None,
                 burnup_target_mean_path = None,
                 burnup_target_std_path = None,
                 library = "sklearn"):
        if burnup_data_mean_path is None:
            burnup_data_mean_path = burnup_model_path.replace(".pkl","") + "_data_mean.csv"
        if burnup_data_std_path is None:
            burnup_data_std_path = burnup_model_path.replace(".pkl","") + "_data_std.csv"
        if burnup_target_mean_path is None:
            burnup_target_mean_path = burnup_model_path.replace(".pkl","") + "_target_mean.csv"
        if burnup_target_std_path is None:
            burnup_target_std_path = burnup_model_path.replace(".pkl","") + "_target_std.csv"

        with open(burnup_model_path, 'rb') as f:
            self.burnup_model = pickle.loads(f.read())
        if library == "sklearn":
            self.burnup_model.set_params(**{"n_jobs":num_cores*2})
        self.burnup_library = library

        self.burnup_data_mean = pd.read_csv(burnup_data_mean_path, index_col=0).iloc[:,0]
        self.burnup_data_std = pd.read_csv(burnup_data_std_path, index_col=0).iloc[:,0]
        self.burnup_target_mean = pd.read_csv(burnup_target_mean_path, index_col=0).iloc[:,0]
        self.burnup_target_std = pd.read_csv(burnup_target_std_path, index_col=0).iloc[:,0]
    def get_velocity(self, z, r):
        for i in reversed(range(len(self.velocity_profile))):
            if z >= self.velocity_profile[i][0]:
                return self.velocity_profile[i][1](r)


    def update_model(self, iteration, day_step, num_substeps, core_flux, insertion_ratios, threshold, debug):
        time_step = day_step/num_substeps*86400
        burn_step_day = day_step/num_substeps
        num_pebbles = len(self.pebbles)

        # Track removal of pebble types for feedback into zone model
        pebbles_modeled_by_zone = {}
        pebbles_removed_by_zone = {}

        depletion_time_array = np.full(num_pebbles, burn_step_day)
        for sub_step in range(1,1+num_substeps):
            if debug > 0:
                print(f"Performing substep {sub_step}.")
            # Initialize arrays for ML input dataframes
            radius_array = np.zeros(num_pebbles)
            height_array = np.zeros(num_pebbles)
            cs137_array = np.zeros(num_pebbles)
            xe135_array = np.zeros(num_pebbles)
            isfuel_array = np.ones(num_pebbles)
            #u235_array = np.zeros(num_pebbles)
            
            conc_array = [None]*num_pebbles
            
            discharge_data_index = 0
            discharge_data = {}
            discard_data_index = 0
            discard_data = {}
            for p in range(num_pebbles):
                peb = self.pebbles[p]
                radius_array[p] = peb.r
                height_array[p] = peb.z
                peb.last_x, peb.last_y, peb.last_r, peb.last_z = peb.x, peb.y, peb.r, peb.z
                concentrations = peb.material.concentrations
                conc_array[p] = concentrations
                if "55137<lib>" in concentrations.keys():
                    cs137_array[p] = peb.material.concentrations["55137<lib>"]
                else:
                    cs137_array[p] = 0
                if '54135<lib>' in concentrations.keys():
                    xe135_array[p] = peb.material.concentrations['54135<lib>']
                else:
                    xe135_array[p] = 0
                #u235_array[p] = peb.material.concentrations['92235<lib>']

            # Put together df for current ML model to use and predict
            if debug > 0:
                print("Creating current dataframe...")
            current_df = pd.DataFrame({"radius": radius_array,
                                      "height": height_array,
                                      "cs137": cs137_array,
                                      "xe135": xe135_array,
                                      "is_fuel": isfuel_array})
            del(radius_array)
            del(height_array)
            del(cs137_array)
            del(xe135_array)
            del(isfuel_array)
            core_flux_df = core_flux.iloc[core_flux.index.repeat(len(current_df))].reset_index(drop=True)
            current_df = pd.concat([current_df,core_flux_df], axis=1)
            if debug > 0:
                print("Standardizing current dataframe...")
            current_df, _, _ = standardize(current_df,
                                           self.current_data_mean,
                                           self.current_data_std)
            if debug > 0:
                print("Predicting pebble currents...")
            if self.current_library == "sklearn":
                predicted_currents = self.current_model.predict(current_df)
                predicted_currents = pd.DataFrame(predicted_currents, columns=self.current_target_mean.axes[0].tolist())
            else:
                predicted_currents = self.current_model.predict(current_df)
            del(current_df)
             
            predicted_currents = unstandardize(predicted_currents,
                                               self.current_target_mean,
                                               self.current_target_std)
            # Use predicted currents + other features like concentrations for burnup model
            if debug > 0:
                print("Creating burnup dataframe...")
            burnup_df = pd.DataFrame({"depletion_time": depletion_time_array})
            predicted_current_df = predicted_currents.drop(columns="power") 
            burnup_df = pd.concat([burnup_df, predicted_current_df], axis=1)
            conc_df = pd.DataFrame(conc_array, columns=self.burnup_target_mean.axes[0].tolist()).fillna(0)
            burnup_df = pd.concat([burnup_df, conc_df], axis=1)
            burnup_df['power'] = predicted_currents['power']
            
            if debug > 0:
                print("Log standardizing burnup dataframe...")
            burnup_df_log = burnup_df.apply(lambda x: np.log10(x + 1))
            burnup_df_log['power'] = burnup_df['power']
            burnup_df_log['depletion_time'] = burnup_df['depletion_time']
            del(burnup_df)
            burnup_df_stan, _, _ = standardize(burnup_df_log, 
                                          self.burnup_data_mean,
                                          self.burnup_data_std)
            del(burnup_df_log)
            if debug > 0:
                print("Predicting concentration changes from burnup...")
            if self.burnup_library == "sklearn":
                predicted_conc_stan = self.burnup_model.predict(burnup_df_stan)
                predicted_conc_stan = pd.DataFrame(predicted_conc_stan, columns=self.burnup_target_mean.axes[0].tolist())
            else:
                predicted_conc_stan = self.burnup_model.predict(predicted_conc_stan)
            del(burnup_df_stan)
            predicted_conc_log = unstandardize(predicted_conc_stan,
                                           self.burnup_target_mean,
                                           self.burnup_target_std)
            predicted_conc = predicted_conc_log.apply(lambda x:10**(x)-1)
            del(predicted_conc_log)
            
            if debug > 0:
                print("Updating fuel pebble concentrations...")
            for p in range(num_pebbles):
                peb = self.pebbles[p] 
                if peb.in_core and peb.is_fuel:
                    peb.material.concentrations = predicted_conc.iloc[p].to_dict()
            del(predicted_conc)

            # After the burnup, move the pebbles
            for p in range(num_pebbles):
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

            with open(f"discharge_pebbles_{iteration}_{sub_step}.json", 'w') as file:
                json.dump(discharge_data, file, indent=2)
            with open(f"discard_pebbles_{iteration}_{sub_step}.json", 'w') as file:
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
            if peb.material.concentrations.get("55137<lib>", 0) > threshold:
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
            peb.z = np.random.uniform(0,1)*vz * time_step
            if debug > 1:
                print(f"Reinserting pebble {peb.id} at x = {peb.x}, y = {peb.y}, z = {peb.z} (pass {peb.pass_num})")
        self.reinsert_indices = []
        self.discharge_indices = []
        for i in self.discard_indices:
            x,y,z = self.insertion_distribution(1)
            mat_name = random.choices(list(insertion_ratios.keys()), weights=insertion_ratios.values(), k=1)[0]
            material = self.fresh_materials[mat_name].concentrations.copy()
            self.pebbles[i] = Pebble(x, y, z, material)
            vz = self.get_velocity(0, peb.r)
            self.pebbles[i].z = np.random.uniform(0, 1) * vz * time_step
            if debug > 1:
                print(f"Regenerating discarded pebble {peb.id} as {material.name} at x = {x}, y = {y}, z = {self.pebbles[i].z}")
        self.discard_indices = []


def gFHR_insertion_distribution(num_pebbles):
    if num_pebbles > 1:
        angles = np.random.uniform(low=0, high=2*np.pi, size=num_pebbles)
        r_vals = np.random.uniform(low=0, high=118, size=num_pebbles)
        x_vals = r_vals * np.cos(angles)
        y_vals = r_vals * np.sin(angles)
        z_vals = np.full(num_pebbles, 60)
    else:
        angle = np.random.uniform(low=0, high=2 * np.pi, size=num_pebbles)[0]
        r_val = np.random.uniform(low=0, high=118, size=num_pebbles)[0]
        x_vals = r_val * np.cos(angle)
        y_vals = r_val * np.sin(angle)
        z_vals = np.full(num_pebbles, 60)[0]
    return x_vals, y_vals, z_vals