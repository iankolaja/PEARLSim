from .zone import Zone, Out_Of_Core_Bin
from .material import Material
import numpy as np
import random
from copy import deepcopy
import pandas as pd
import os
import json

def get_zone(r, z, radial_bound_points, axial_zone_bounds):
    num_radial_channels = len(radial_bound_points)
    radial_boundaries = []
    for r_zone in range(num_radial_channels):
        points = radial_bound_points[r_zone]
        radial_boundaries += [np.interp(z, points[:,0], points[:,1])]
    radial_boundaries = np.array(radial_boundaries)
    rad_zone = np.searchsorted(radial_boundaries, r)
    axial_zone = np.searchsorted(axial_zone_bounds[rad_zone], z)
    return (rad_zone, axial_zone)

def _get_fuel_type(mat_name):
    name = mat_name.split("_")
    if len(name) > 2:
        fuel_type = name[1]
    else:
        fuel_type = name[0]
    return fuel_type

def _get_fuel_types(inventory):
    fuel_types = []
    for key in inventory.keys():
        fuel_types += [_get_fuel_type(key)]
    return list(set(fuel_types))


class Core():
    def __init__(self):
        self.zones = []
        self.simulation_name = "gFHR_core"
        self.radial_bound_points = []
        self.axial_zone_bounds = []
        self.num_r = 0
        self.num_z = []
        self.static_fuel_materials = {}
        self.max_passes = 8
        self.iteration = 1
        self.pebble_radius = 2.0
        self.burnup_time = 6.525
        self.num_top_zone_pebbles = 0
        self.materials = {}
        self.fresh_pebbles = {}
        self.pebble_locations = pd.DataFrame([], columns=["x","y","z","r","zone_r","zone_z"])
        self.core_geometry = ""
        self.discharge_inventory = Out_Of_Core_Bin()
        self.reinsert_inventory = Out_Of_Core_Bin()
        self.always_remove_graphite = True

    def define_zones(self, zone_file, pebble_file, debug=0):
        pebble_locations = []
        with open(zone_file, 'r') as zone_f:
            radial_channel_num = 0
            radial_zones = []
            axial_zone_num = 0
            for line in zone_f:
                line = line.split()
                if len(line) > 0:
                    keyword = line[0]
                else:
                    keyword = ""
                if keyword == "radial_channel":
                    radial_channel_num += 1
#                    if radial_channel_num > 1:
#                        self.zones += [radial_zones]
#                        self.axial_zone_counts += [axial_zone_num]
#                    axial_zone_num = 0
#                    radial_zones = []
                    points = []
                if keyword == "z_divisions":
                    axial_bounds = np.array(line[1:]).astype(float)
                if keyword == "point":
                    points += [(float(line[1]), float(line[2]))]
                if keyword == "end":
                    self.radial_bound_points += [np.array(points)]
                    self.axial_zone_bounds += [np.array(axial_bounds)]

                    #axial_zone_num += 1
                    #radial_zones += [Zone(1, radial_channel_num, axial_zone_num)]
        self.radial_bound_points = self.radial_bound_points
        self.axial_zone_bounds = self.axial_zone_bounds
        self.num_r = len(self.radial_bound_points)
        self.num_z = []
        self.zones = []
        for r in range(self.num_r):
            self.num_z += [len(self.axial_zone_bounds[r])+1]
            self.zones += [[]]
            for z in range(self.num_z[-1]):
                self.zones[r] += [Zone(1, r+1, z+1)]
        with open(pebble_file, 'r') as pebble_f:
            for line in pebble_f:
                line = line.split()
                x = float(line[0])
                y = float(line[1])
                z = float(line[2])
                r = np.sqrt(x**2+y**2)
                zone_r, zone_z = get_zone(r, z, self.radial_bound_points, self.axial_zone_bounds)
                if debug >= 2:
                    print(f"Pebble at {x},{y},{z} (r={r} placed in zone R{zone_r+1}Z{zone_z+1}")
                self.zones[zone_r][zone_z].pebble_locations += [(x, y, z, r)]
                pebble_locations += [(x,y,z,r,zone_r,zone_z)]
        for r in range(self.num_r):
            self.num_z += [len(self.axial_zone_bounds[r])+1]
            self.num_top_zone_pebbles += len(self.zones[r][-1].pebble_locations)
        self.pebble_locations = pd.DataFrame(pebble_locations, columns=["x","y","z","r","zone_r","zone_z"])
    def rename_materials(self, rename_map):
        for key in rename_map.keys():
            new_material = rename_map[key]
            if key in self.materials.keys():
                self.materials[new_material] = deepcopy(self.materials[key])
                self.materials[new_material].name = new_material
            elif key in self.fresh_pebbles.keys():
                self.materials[new_material] = deepcopy(self.fresh_pebbles[key])
                self.materials[new_material].name = new_material
                self.materials[new_material].rgb = (np.random.randint(50, 150),
                                                    np.random.randint(50, 150),
                                                    np.random.randint(50, 150))
            else:
                raise Exception(f"Error: Missing material {key} to copy to {new_material}.")


    def insert(self, insert_pebbles, threshold, pebble_model, debug=0):
        num_radial_channels = len(self.zones)
        num_bottom_zone_pebbles = 0

        if debug >= 1:
            print("Clearing discharge inventory...")

        self.reinsert_inventory = deepcopy(self.discharge_inventory)
        rename_map = self.reinsert_inventory.increment_pass()
        self.rename_materials(rename_map)
        if self.reinsert_inventory.num_pebbles > 0:
            reinsert_fracs = dict(zip(self.reinsert_inventory.inventory.keys(),
                                      map(lambda x: x / self.num_top_zone_pebbles,
                                          self.reinsert_inventory.inventory.values())))
        else:
            reinsert_fracs = {}

        self.discharge_inventory.clear()


        for r_zone in range(num_radial_channels):
            num_axial_zones = len(self.axial_zone_bounds[r_zone])+1

            if debug >= 1:
                print(f"Extracting from top zone radial channel {r_zone}...")
            # Extract top zones
            rename_map = self.discharge_inventory.extract_top_zone(self.zones[r_zone][-1])
            self.rename_materials(rename_map)

            if debug >= 1:
                print(f"Propagating materials up radial channel {r_zone}...")
            # Starting from the top, move materials from the zone directly below
            for z_zone in reversed(range(1, num_axial_zones)):
                rename_map = self.zones[r_zone][z_zone].propagate_from(self.zones[r_zone][z_zone-1])
                self.rename_materials(rename_map)
            num_bottom_zone_pebbles += self.zones[r_zone][0].num_pebbles

        if debug >= 1:
            print(f"Removing spent pebbles from discrete model...")
        removal_fractions, removed_pebbles = pebble_model.remove_spent_pebbles(self.radial_bound_points,
                                                                               self.axial_zone_bounds,
                                                                               threshold,
                                                                               self.always_remove_graphite)
        if debug >= 1:
            print(f"Removing spent pebbles from top zones...")
        removed_pebbles = self.discharge_inventory.remove_fractions(removal_fractions, self.always_remove_graphite)
        if debug >= 1:
            print(f"Removed {removed_pebbles} pebbles based on fraction of comparable simulated pebbles crossing threshold.")
        removed_pebbles = self.discharge_inventory.remove_max_passes(self.max_passes)
        if debug >= 1:
            print(f"Removed {removed_pebbles} pebbles based on zone materials exceeding hard limit.")

        if debug >= 1:
            print(f"Volume averaging discharge inventory by pass...")
        self.volume_average_by_pass(self.discharge_inventory)

        if debug >= 1:
            print(f"Removing spend pebbles from top zones...")
        for r_zone in range(num_radial_channels):
            # How many discharge pebbles belong to this bottom zone
            #fraction_in_zone = self.zones[r_zone][0].num_pebbles / num_bottom_zone_pebbles
            #print(fraction_in_zone)
            #reinsert_fracs = {}
            #for key in self.reinsert_inventory.inventory.keys():
            #    #
            #    reinsert_fracs[key] = self.reinsert_inventory.inventory[key]*fraction_in_zone/self.reinsert_inventory.num_pebbles
            #print(reinsert_fracs)
            #print("\n\n\n")
            #print(insert_pebbles)
            rename_map = self.zones[r_zone][0].insert(insert_pebbles, reinsert_fracs)
            self.rename_materials(rename_map)
        if debug > 0:
            for r_zone in range(num_radial_channels):
                num_axial_zones = len(self.axial_zone_bounds[r_zone]) + 1
                for z_zone in range(num_axial_zones):
                    self.zones[r_zone][z_zone].print_status()
            print(self.discharge_inventory.inventory)

        # Perform out of core burn calculations here


    def volume_average_by_pass(self, out_of_core_bin):
        averaged_materials = {}
        averaged_materials_fractions = {}
        fuel_types = _get_fuel_types(out_of_core_bin.inventory)

        # Go through each type and pass of fuel, a "group"
        for p in range(1, 1+self.max_passes):
            for fuel_type in fuel_types:
                grouped_materials = {}
                total_group_pebbles = 0

                # Iterate through all bin pebbles to see if they're that pebble type
                for key in out_of_core_bin.inventory.keys():
                    _,pass_num = key.split("P")

                    # If so, add them to the grouped materials dictionary and count them
                    if int(pass_num) == p and fuel_type in key:
                        grouped_materials[key] = out_of_core_bin.inventory[key]
                        total_group_pebbles += out_of_core_bin.inventory[key]

                # If this pebble group is nonzero, create a new material to volume average
                if total_group_pebbles > 0:
                    averaged_grouped_material_key = f"avgdischarge_{fuel_type}_P{p}"
                    averaged_grouped_material = Material(averaged_grouped_material_key, {},
                                              temperature=0, pass_num=pass_num, density="sum fix")
                    average_temperature = 0
                    for key in grouped_materials.keys():
                        pebbles_in_material = grouped_materials[key]
                        volume_fraction = pebbles_in_material/total_group_pebbles
                        averaged_grouped_material = averaged_grouped_material + (self.materials[key]*volume_fraction)
                        average_temperature += self.materials[key].temperature * volume_fraction
                    averaged_materials[averaged_grouped_material_key] = total_group_pebbles
                    averaged_grouped_material.temperature = average_temperature
                    self.materials[averaged_grouped_material_key] = averaged_grouped_material

        # Reassign the inventories as just the new averaged material
        out_of_core_bin.inventory = averaged_materials

    def initialize_materials(self, zone_inventory):
        for key in zone_inventory.keys():
            fuel_type = key.split("_")[0]
            self.materials[key] = Material(key, self.fresh_pebbles[fuel_type])

    def generate_pebble_locations(self, filename):
        num_radial_channels = len(self.zones)
        pebble_text = ""
        assigned_pebbles = pd.DataFrame()
        for r_zone in range(num_radial_channels):
            num_axial_zones = len(self.axial_zone_bounds[r_zone]) + 1
            for z_zone in range(num_axial_zones):
                pebble_s, zone_inventory = self.zones[r_zone][z_zone].shuffle_pebbles(self.pebble_radius)
                pebble_text += pebble_s
                assigned_pebbles = pd.concat([assigned_pebbles, zone_inventory])
        with open(filename, 'w') as f:
            f.write(pebble_text)
        return assigned_pebbles

    def generate_pebble_detectors(self, num_detectors, assigned_pebbles):
        pebble_ids = np.random.choice(len(assigned_pebbles), num_detectors, replace=False)
        detector_text = ""
        detector_id = 0

        # Extra features to track per pebble
        temperature_array = []
        cs137_array = []
        fuel_flag_array = []

        for i in pebble_ids:
            detector_id += 1
            data = assigned_pebbles.iloc[i]
            mat_name = data['material']
            detector_text += f"surf peb{detector_id}_s sph {data['x']} {data['y']} {data['z']} {self.pebble_radius}\n"
            temperature_array += [self.materials[mat_name].temperature]
            if "fuel" in mat_name:
                fuel_flag_array += [1]
                if "551370" in self.materials[mat_name].concentrations.keys():
                    cs137_array += [self.materials[mat_name].concentrations['551370']]
                else:
                    cs137_array += [0]
            else:
                fuel_flag_array += [0]
                cs137_array += [0]

            detector_text += f"det peb_{i}_{round(data['r'],4)}_{round(data['z'],4)}_ ds peb{detector_id}_s -1 de standard_grid\n"

        auxiliary_features = pd.DataFrame({"temperature": temperature_array,
                                           "cs137": cs137_array,
                                           "is_fuel": fuel_flag_array})
        auxiliary_features.to_csv(f"current_auxiliary_features{self.iteration}.csv")
        return detector_text

    def get_volumes(self, fuel_kernel_per_pebble_volume = 0.36263376):
        volumes = {}
        for radial_channel in self.zones:
            for zone in radial_channel:
                for mat_name in zone.inventory.keys():
                    if "fuel" in mat_name:
                        volumes[mat_name] = zone.inventory[mat_name]*fuel_kernel_per_pebble_volume
        return volumes


    def generate_input(self, serpent_settings, num_training_data, debug):
        input_str = self.core_geometry
        if num_training_data == 0:
            input_str += f"dep daystep {self.burnup_time}\n"
        for setting in serpent_settings.keys():
            input_str += f"set {setting} {serpent_settings[setting]}\n"

        input_str += "\n\n%%%%%%%% Pebble Universe Definition \n\n"
        peb_file_name = f"pebble_positions_{self.iteration}.csv"
        assigned_pebbles = self.generate_pebble_locations(peb_file_name)
        if num_training_data > 0:
            input_str += f"pbed u_pb u_flibe \"{peb_file_name}\"  pow\n"
        else:
            input_str += f"pbed u_pb u_flibe \"{peb_file_name}\" \n"

        input_str += "\n\n%%%%%%%% Material and Pebble Definitions \n\n"
        triso_counter = 1
        for material in self.materials.values():
            if "discharge" not in material.name:
                input_str += material.write_input(triso_counter, self.static_fuel_materials ,debug)
                triso_counter += 1

        input_str += "\n\n%%%%%%%% Material Definitions \n\n"
        input_str += "set mvol\n"
        volumes = self.get_volumes()
        for mat_name in volumes.keys():
            input_str += f"  {mat_name} 1 {volumes[mat_name]}\n"

        if num_training_data > 0:
            input_str += "\n\n%%%%%%%% Pebble Current Detectors for ML model \n\n"
            detector_str = self.generate_pebble_detectors(num_training_data, assigned_pebbles)
            input_str += detector_str

        file_name = f"{self.simulation_name}_{self.iteration}.serpent"
        with open(file_name, 'w') as f:
            f.write(input_str)
        return file_name

    def save_zone_maps(self, file_name):
        zone_map = {}
        for radial_channel in self.zones:
            for zone in radial_channel:
                zone_id = f"zoneR{zone.radial_num}Z{zone.axial_num}"
                zone_map[zone_id] = zone.inventory
        zone_str = json.dumps(zone_map)
        with open(file_name, 'w') as f:
            f.write(zone_str)
        return zone_str


    def update_from_bumat(self, debug):
        bumat_name = f"{self.simulation_name}_{self.iteration}.serpent.bumat1"
        if debug > 0:
            print(f"Reading {bumat_name}")
        with open(bumat_name, 'r') as f:
            reading = False
            first_mat = True
            for line in f:
                line = line.split()
                if len(line) == 0:
                    continue
                if line[0] == "mat":
                    if not first_mat:
                        if debug > 0:
                            print(f"Updating {current_mat_name}")
                        self.materials[current_mat_name].concentrations = current_conc
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
                    amount = float(line[1].replace("\n",""))
                    current_conc[nuclide] = amount
        if debug > 0:
            print(f"Updating {current_mat_name}")
        self.materials[current_mat_name].concentrations = current_conc
