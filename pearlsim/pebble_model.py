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
        self.id = id
        self.pass_num = 1
        self.intial_fuel = material_source
        self.in_core = True
        self.material = Material(f"pebble{id}", material_source)

    def reinsert(self, distribution):
        x_val, y_val, z_val = distribution(1)
        self.x = x_val
        self.y = y_val
        self.z = z_val
        self.pass_num += 1
        self.r = np.sqrt(self.x**2 + self.y**2)


class Pebble_Model():
    def __init__(self):
        self.pebbles = []
        self.max_z = 429
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

    def load_current_model(self, current_model_path,
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
        self.current_library = library

        self.current_data_mean = pd.read_csv(current_data_mean_path, index_col=0)
        self.current_data_std = pd.read_csv(current_data_std_path, index_col=0)
        self.current_target_mean = pd.read_csv(current_target_mean_path, index_col=0)
        self.current_target_std = pd.read_csv(current_target_std_path, index_col=0)


    def load_burnup_model(self, burnup_model_path,
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

        self.burnup_data_mean = pd.read_csv(burnup_data_mean_path, index_col=0)
        self.burnup_data_std = pd.read_csv(burnup_data_std_path, index_col=0)
        self.burnup_target_mean = pd.read_csv(burnup_target_mean_path, index_col=0)
        self.burnup_target_std = pd.read_csv(burnup_target_std_path, index_col=0)
    def get_velocity(self, z, r):
        for i in reversed(range(len(self.velocity_profile))):
            if z > self.velocity_profile[i][0]:
                return self.velocity_profile[i][1](r)


    def update_model(self, iteration, time, num_steps, core_flux, insertion_ratios, threshold, debug):
        time_step = time/num_steps
        day_step = time_step/86400
        num_pebbles = len(self.pebbles)

        # Track removal of pebble types for feedback into zone model
        pebbles_modeled_by_zone = {}
        pebbles_removed_by_zone = {}

        # Initialize arrays for ML input dataframes
        radius_array = np.zeros(num_pebbles)
        height_array = np.zeros(num_pebbles)
        cs137_array = np.zeros(num_pebbles)
        xe135_array = np.zeros(num_pebbles)
        isfuel_array = np.ones(num_pebbles)

        conc_array = [None]*num_pebbles
        depletion_time_array = np.full(num_pebbles, day_step)

        discharge_data_index = 0
        discharge_data = {}
        discard_data_index = 0
        discard_data = {}

        for sub_step in range(num_steps):
            for p in range(num_pebbles):
                peb = self.pebbles[p]

                # Get the velocity for the pebble and update its location
                vz = self.get_velocity(peb.z, peb.r)
                peb.z = peb.z + vz*time_step

                radius_array[p] = peb.r
                height_array[p] = peb.z
                if peb.z > self.max_z:
                    self.discharge_indices += [p]
                    peb.in_core = False
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

            # Put together df for current ML model to use and predict
            current_df = pd.DataFrame({"radius": radius_array,
                                      "height": height_array,
                                      "cs137": cs137_array,
                                      "xe135": xe135_array,
                                      "isfuel": isfuel_array})
            print(current_df)
            core_flux = core_flux.iloc[core_flux.index.repeat(len(current_df))].reset_index(drop=True)
            print(core_flux)
            current_df = pd.concat([current_df, core_flux], axis=1)
            current_df, _, _ = standardize(current_df,
                                           self.current_data_mean,
                                           self.current_data_std)
            print(current_df)
            if self.current_library == "sklearn":
                predicted_currents = self.current_model.predict(current_df)
            else:
                predicted_currents = self.current_model.predict(current_df)
            predicted_currents = unstandardize(predicted_currents,
                                               self.current_target_mean,
                                               self.current_target_std)

            # Use predicted currents + other features like concentrations for burnup model
            burnup_df = pd.DataFrame(pd.DataFrame({"depletion_time": depletion_time_array}))
            burnup_df.join(predicted_currents)
            conc_df = pd.DataFrame(conc_array).fillna(0)
            burnup_df.join(conc_df)

            if self.burnup_library == "sklearn":
                predicted_conc = self.burnup_model.predict(burnup_df)
            else:
                predicted_conc = self.burnup_model.predict(burnup_df)
            predicted_conc, _, _ = unstandardize(predicted_conc,
                                           self.burnup_target_mean,
                                           self.burnup_target_std)

            # Iterate through pebbles again to update concentrations
            for p in range(num_pebbles):
                if self.pebbles[p].in_core:
                    self.pebbles[p].material.concentrations = predicted_conc.iloc[p].to_dict()
            pebbles_modeled_by_zone, pebbles_removed_by_zone = self.remove_spent_pebbles(threshold,
                                                                                         pebbles_modeled_by_zone,
                                                                                         pebbles_removed_by_zone,
                                                                                         True)

            # Track and write discharge pebble data on each sub-iteration
            for p in self.discharge_indices:
                peb = self.pebbles.p
                if p in self.discard_indices:
                    discard_data[discard_data_index] = {
                        "features": {
                            "radius": peb.r
                        },
                        "concentration": peb.Material.concentrations
                    }
                    discard_data_index += 1
                else:
                    discharge_data[discharge_data_index] = {
                        "features": {
                            "radius": peb.r
                        },
                        "concentration": peb.Material.concentrations
                    }
                    discharge_data_index += 1

            with open(f"discharge_pebbles_{iteration}_{sub_step}.json", 'w') as file:
                json.dump(discharge_data, file)
            with open(f"discard_pebbles_{iteration}_{sub_step}.json", 'w') as file:
                json.dump(discard_data, file)

            # Reinsert non-discarded pebbles and replace discarded pebbles with fresh ones, all at the bottom
            self.reinsert_pebbles(insertion_ratios, debug)

        # Calculate fractions of pebbles to remove from the zone model
        removal_fractions = {}
        for key in pebbles_modeled_by_zone.keys():
            if key in pebbles_removed_by_zone.keys():
                removal_fractions[key] = pebbles_removed_by_zone[key]/pebbles_modeled_by_zone[key]
            else:
                removal_fractions[key] = 0
        return removal_fractions

    def initialize_pebbles(self, num_pebbles, pebble_points, initial_inventory, debug=0):
        indices = random.choices(range(len(pebble_points)), k=num_pebbles)
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
            material = deepcopy(self.fresh_materials[mat_name])
            print(f"Generating {material.name} pebble at x = {x}, y = {y}, z = {z}")
            self.pebbles += [Pebble(x, y, z, material)]


    def remove_spent_pebbles(self, threshold, pebbles_modeled_by_zone, pebbles_removed_by_zone, graphite_removal_flag):
        """
        Check all invividual pebbles in top zones to see what fraction of them should be removed.
        :param radial_bound_points:
        :param axial_zone_bounds:
        :param threshold:
        :return:
        """
        self.discarded_indices = []
        self.reinsert_indices = []
        for i in range(len(self.pebbles)):
            pebble = self.pebbles[i]
            rad_zone, axial_zone = get_zone(pebble.r, pebble.z, self.radial_bound_points, self.axial_zone_bounds)

            # Check if the pebble is in a top-most zone
            if axial_zone == len(self.axial_zone_bounds[rad_zone-1])-1:
                zone_material_name = f"{pebble.intial_fuel}_R{rad_zone}Z{axial_zone}P{pebble.pass_num}"
                pebbles_modeled_by_zone[zone_material_name] = \
                    1+pebbles_modeled_by_zone.get(zone_material_name, 0)

                # Check if pebble's cesium concentration is above threshold
                if pebble.material.concentrations.get("55137<lib>", 0) > threshold:
                    pebbles_removed_by_zone[zone_material_name] = \
                        1 + pebbles_removed_by_zone.get(zone_material_name, 0)
                    self.discard_indices += [i]

                # Check to see if its a graphite pebble and if it should be removed
                elif graphite_removal_flag and "pebgraph" in pebble.material.name:
                    pebbles_removed_by_zone[zone_material_name] = \
                        1 + pebbles_removed_by_zone.get(zone_material_name, 0)

                    self.discard_indices += [i]
                # Otherwise reinsert the pebble
                else:
                    self.reinsert_indices += [i]
        return pebbles_removed_by_zone, pebbles_modeled_by_zone

    def reinsert_pebbles(self, insertion_ratios, debug):
        for i in self.reinsert_indices:
            self.pebbles[i].reinsert(self.insertion_distribution)
            if debug > 1:
                peb = self.pebbles[i]
                print(f"Reinserting pebble {peb.id} at x = {peb.x}, y = {peb.y}, z = {peb.z} (pass {peb.pass_num})")
        for i in self.discard_indices:
            x,y,z = self.insertion_distribution(1)
            mat_name = random.choices(list(insertion_ratios.keys()), weights=insertion_ratios.values(), k=1)[0]
            material = deepcopy(self.fresh_materials[mat_name])
            if debug > 1:
                print(f"Generating {material.name} pebble at x = {x}, y = {y}, z = {z}")
            self.pebbles[i] = Pebble(x, y, z, material)

def gFHR_insertion_distribution(num_pebbles):
    if num_pebbles > 1:
        angles = np.random.uniform(low=0, high=2*np.pi, size=num_pebbles)
        r_vals = np.random.uniform(low=0, high=118, size=num_pebbles)
        x_vals = r_vals * np.cos(angles)
        y_vals = r_vals * np.sin(angles)
        z_vals = np.zeros(num_pebbles)
    else:
        angle = np.random.uniform(low=0, high=2 * np.pi, size=num_pebbles)[0]
        r_val = np.random.uniform(low=0, high=118, size=num_pebbles)[0]
        x_vals = r_val * np.cos(angle)
        y_vals = r_val * np.sin(angle)
        z_vals = np.zeros(num_pebbles)[0]
    return x_vals, y_vals, z_vals