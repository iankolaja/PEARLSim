from .zone import Zone, Out_Of_Core_Bin
from .material import Material
import numpy as np
import random
from copy import deepcopy
import pandas as pd
import os
import json
from .get_fima import get_fima
import gzip

BINNING_ATTEMPTS = 20


def get_zone(r, z, radial_bound_points, axial_zone_bounds):
    num_radial_channels = len(radial_bound_points)
    radial_boundaries = []
    for r_zone in range(num_radial_channels):
        points = radial_bound_points[r_zone]
        radial_boundaries += [np.interp(z, points[:, 0], points[:, 1])]
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
    def __init__(self, simulation_name):
        self.zones = []
        self.simulation_name = simulation_name
        self.radial_bound_points = []
        self.axial_zone_bounds = []
        self.num_r = 0
        self.num_z = []
        self.static_fuel_materials = {}
        self.max_passes = 8
        self.fuel_groups = 12
        self.feature_search_radius = 20.0
        self.iteration = 1
        self.pebble_radius = 2.0
        self.num_top_zone_pebbles = 0
        self.substep_schema = [1.0]
        self.averaging_mode = "pass"
        self.initial_actinides = 0.00466953 + 0.0189728  # u235 x u238
        self.materials = {}
        self.fresh_pebbles = {}
        self.pebble_locations = pd.DataFrame([], columns=["x", "y", "z", "r", "zone_r", "zone_z", "material"])
        self.core_geometry = ""
        self.discharge_inventory = Out_Of_Core_Bin()
        self.reinsert_inventory = Out_Of_Core_Bin()
        self.always_remove_graphite = True

    def define_zones(self, zone_file, pebble_file, debug=0):
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
                    points = []
                if keyword == "z_divisions":
                    axial_bounds = np.array(line[1:]).astype(float)
                if keyword == "point":
                    points += [(float(line[1]), float(line[2]))]
                if keyword == "end":
                    self.radial_bound_points += [np.array(points)]
                    self.axial_zone_bounds += [np.array(axial_bounds)]

        self.radial_bound_points = self.radial_bound_points
        self.axial_zone_bounds = self.axial_zone_bounds
        self.num_r = len(self.radial_bound_points)
        self.num_z = []
        self.zones = []
        for r in range(self.num_r):
            self.num_z += [len(self.axial_zone_bounds[r]) + 1]
            self.zones += [[]]
            for z in range(self.num_z[-1]):
                self.zones[r] += [Zone(1, r + 1, z + 1)]
        self.load_pebble_locations(pebble_file, debug)
        for r in range(self.num_r):
            self.num_z += [len(self.axial_zone_bounds[r]) + 1]
            self.num_top_zone_pebbles += len(self.zones[r][-1].pebble_locations)

    def load_pebble_locations(self, pebble_file, debug):
        pebble_locations = []
        if type(self.zones[0][0].pebble_locations) == list:
            loading_zone_locations = True
        else:
            loading_zone_locations = False
        with open(pebble_file, 'r') as pebble_f:
            for line in pebble_f:
                line = line.split()
                x = float(line[0])
                y = float(line[1])
                z = float(line[2])
                r = np.sqrt(x ** 2 + y ** 2)
                material = line[4][1:].replace("\n", "")
                zone_r, zone_z = get_zone(r, z, self.radial_bound_points, self.axial_zone_bounds)
                if debug >= 2:
                    print(f"Pebble at {x},{y},{z} (r={r} placed in zone R{zone_r + 1}Z{zone_z + 1}")
                if loading_zone_locations:
                    self.zones[zone_r][zone_z].pebble_locations += [(x, y, z, r)]
                pebble_locations += [(x, y, z, r, zone_r, zone_z, material)]
        self.pebble_locations = pd.DataFrame(pebble_locations,
                                             columns=["x", "y", "z", "r", "zone_r", "zone_z", "material"])

    def rename_materials(self, rename_map):
        for key in rename_map.keys():
            new_material = rename_map[key]
            if key in self.materials.keys():
                self.materials[new_material] = deepcopy(self.materials[key])
                self.materials[new_material].name = new_material
            elif key in self.fresh_pebbles.keys():
                self.materials[new_material] = deepcopy(self.fresh_pebbles[key])
                self.materials[new_material].name = new_material
            else:
                raise Exception(f"Error: Missing material {key} to copy to {new_material}.")

    def insert(self, insert_pebbles, threshold, threshold_rel_std, pebble_model, debug=0):
        num_radial_channels = len(self.zones)
        num_bottom_zone_pebbles = 0

        if debug >= 1:
            print("Clearing discharge inventory...")

        self.reinsert_inventory, rename_map = self.discharge_inventory.to_reinsert(self.averaging_mode)
        self.rename_materials(rename_map)

        if self.reinsert_inventory.num_pebbles > 0:
            reinsert_fracs = dict(zip(self.reinsert_inventory.inventory.keys(),
                                      map(lambda x: x / self.num_top_zone_pebbles,
                                          self.reinsert_inventory.inventory.values())))
        else:
            reinsert_fracs = {}

        self.discharge_inventory.clear()

        for r_zone in range(num_radial_channels):
            num_axial_zones = len(self.axial_zone_bounds[r_zone]) + 1

            if debug >= 1:
                print(f"Extracting from top zone radial channel {r_zone}...")
            # Extract top zones
            rename_map = self.discharge_inventory.extract_top_zone(self.zones[r_zone][-1])
            self.rename_materials(rename_map)

            if debug >= 1:
                print(f"Propagating materials up radial channel {r_zone}...")
            # Starting from the top, move materials from the zone directly below
            for z_zone in reversed(range(1, num_axial_zones)):
                rename_map = self.zones[r_zone][z_zone].propagate_from(self.zones[r_zone][z_zone - 1])
                self.rename_materials(rename_map)
            num_bottom_zone_pebbles += self.zones[r_zone][0].num_pebbles

        if debug >= 1:
            print(f"Removing spent pebbles from discrete model...")
        removal_fractions = {}
        removed_pebbles = 0
        self.save_discharge_inventory(f"{self.simulation_name}_discharge_inventory{self.iteration}.json",
                                      self.initial_actinides)
        # removal_fractions, removed_pebbles = pebble_model.remove_spent_pebbles(threshold,
        #                                                                       removal_fractions,
        #                                                                       removed_pebbles,
        #                                                                       self.always_remove_graphite)
        if debug >= 1:
            print(f"Removing spent pebbles from top zones...")
        removed_pebbles = self.discharge_inventory.remove_fractions(removal_fractions, self.always_remove_graphite)
        if debug >= 1:
            print(
                f"Removed {removed_pebbles} pebbles based on fraction of comparable simulated pebbles crossing threshold.")
        if self.averaging_mode == "pass":
            removed_pebbles = self.discharge_inventory.remove_max_passes(self.max_passes)
        else:
            removed_pebbles, removed_inventory = self.discharge_inventory.remove_threshold(threshold, threshold_rel_std,
                                                                                           self.materials,
                                                                                           self.initial_actinides,
                                                                                           debug=debug)
        if debug >= 1:
            print(f"Removed {removed_pebbles} pebbles based on zone materials exceeding hard limit.")

        if debug >= 1:
            print(f"Volume averaging discharge inventory by pass...")



        discard_path = f"{self.simulation_name}_discard_inventory{self.iteration}.json"
        with open(discard_path, 'w') as f:
            json.dump(removed_inventory, f)

        if self.averaging_mode == "pass":
            self.volume_average_by_pass(self.discharge_inventory)
        elif self.averaging_mode == "burnup":
            self.volume_average_by_burnup(self.discharge_inventory)

        if debug >= 1:
            print(f"Inserting pebbles at bottom of core...")
        for r_zone in range(num_radial_channels):
            rename_map = self.zones[r_zone][0].insert(insert_pebbles, reinsert_fracs)
            self.rename_materials(rename_map)
            for mat_name in self.zones[r_zone][0].inventory.keys():                 
                material_conc = self.materials[mat_name].concentrations
                fima = get_fima(material_conc, self.initial_actinides)
                self.materials[mat_name].fima = fima
        if debug > 0:
            for r_zone in range(num_radial_channels):
                num_axial_zones = len(self.axial_zone_bounds[r_zone]) + 1
                for z_zone in range(num_axial_zones):
                    self.zones[r_zone][z_zone].print_status()
            print(self.discharge_inventory.inventory)

        # Perform out of core burn calculations here

    def volume_average_by_burnup(self, out_of_core_bin, ):
        averaged_materials = {}
        averaged_materials_fractions = {}
        fuel_types = _get_fuel_types(out_of_core_bin.inventory)

        fima_values = []
        material_keys = []
        for key in out_of_core_bin.inventory.keys():
            if "fuel" in key:
                material_keys += [key]
                fima_values += [get_fima(self.materials[key].concentrations, self.initial_actinides)]

        # Assign pebbles to groups based on which bin their concentration
        # of the selected isotope falls in

        # Because there may be many zero bins in some situations, keep making
        # the bin structure finer until the desired number of nonzero bins
        # exists.
        num_assigned_bins = 0
        num_attempts = 0
        while (num_assigned_bins < self.fuel_groups - 1) and (num_attempts < BINNING_ATTEMPTS):
            counts, bins = np.histogram(np.array(fima_values), bins=self.fuel_groups - 1 + num_attempts)
            num_assigned_bins = np.sum(counts > 0)
            num_attempts += 1
            print(fima_values)
            print(bins)

        # Do one final histogram with nonzero bins removed so that group labels
        # increase from 2 to the desired number of groups with no gaps
        nonzero_bins_indices = np.where(counts != 0)[0]
        nonzero_bins = bins[nonzero_bins_indices]
        counts, bins = np.histogram(fima_values, bins=nonzero_bins)

        if len(fima_values) > 0:
            bins[-1] = bins[-1] + 1e-5  # Make sure the largest value is included in the last bin
        group_assignments = np.digitize(fima_values, bins) + 1
        print("fima_values")
        print(fima_values)
        print("final bins")
        print(bins)
        print("group assignments")
        print(group_assignments)

        # Go through each type of fuel (if applicable), and each group
        # as defined by binned isotope values
        for g in range(2, 1 + self.fuel_groups):
            print(f"group {g}")
            for fuel_type in fuel_types:
                grouped_materials = {}
                total_group_pebbles = 0

                pebbles_in_group = np.where(group_assignments == g)[0]
                print(pebbles_in_group)
                # Iterate through all pebbles in a group and select the ones
                # of the fuel type currently being constructed
                if len(pebbles_in_group) > 0:
                    for index in pebbles_in_group:
                        key = material_keys[index]
                        if fuel_type in key:
                            grouped_materials[key] = out_of_core_bin.inventory[key]
                            total_group_pebbles += out_of_core_bin.inventory[key]

                # If this pebble group is nonzero, create a new material to volume average
                if total_group_pebbles > 0:
                    averaged_grouped_material_key = f"avgdischarge_{fuel_type}_G{g}"
                    averaged_grouped_material = Material(averaged_grouped_material_key, {},
                                                         temperature=0, density="sum fix")
                    average_temperature = 0
                    for key in grouped_materials.keys():
                        pebbles_in_material = grouped_materials[key]
                        volume_fraction = pebbles_in_material / total_group_pebbles
                        averaged_grouped_material = averaged_grouped_material + (self.materials[key] * volume_fraction)
                        average_temperature += self.materials[key].temperature * volume_fraction
                    averaged_materials[averaged_grouped_material_key] = total_group_pebbles
                    averaged_grouped_material.temperature = average_temperature
                    self.materials[averaged_grouped_material_key] = averaged_grouped_material

        # Reassign the inventories as just the new averaged material
        out_of_core_bin.inventory = averaged_materials

    def volume_average_by_pass(self, out_of_core_bin):
        averaged_materials = {}
        averaged_materials_fractions = {}
        fuel_types = _get_fuel_types(out_of_core_bin.inventory)

        # Go through each type and pass of fuel, a "group"
        for p in range(1, 1 + self.max_passes):
            for fuel_type in fuel_types:
                grouped_materials = {}
                total_group_pebbles = 0

                # Iterate through all bin pebbles to see if they're that pebble type
                for key in out_of_core_bin.inventory.keys():
                    _, pass_num = key.split("G")

                    # If so, add them to the grouped materials dictionary and count them
                    if int(pass_num) == p and fuel_type in key:
                        grouped_materials[key] = out_of_core_bin.inventory[key]
                        total_group_pebbles += out_of_core_bin.inventory[key]

                # If this pebble group is nonzero, create a new material to volume average
                if total_group_pebbles > 0:
                    averaged_grouped_material_key = f"avgdischarge_{fuel_type}_G{p}"
                    averaged_grouped_material = Material(averaged_grouped_material_key, {},
                                                         temperature=0, density="sum fix")
                    average_temperature = 0
                    for key in grouped_materials.keys():
                        pebbles_in_material = grouped_materials[key]
                        volume_fraction = pebbles_in_material / total_group_pebbles
                        averaged_grouped_material = averaged_grouped_material + (self.materials[key] * volume_fraction)
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

    def generate_pebble_locations(self, filename, do_shuffle=True):
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

    def generate_sample_pebbles(self, num_detectors, pebble_file, triso_counter, debug, 
        do_average=True, fuel_kernel_per_pebble_volume=0.36263376):
        is_fuel_array = self.pebble_locations['material'].str.contains("fuel")
        fuel_pebble_ids = self.pebble_locations[is_fuel_array].index.values
        
        # Set seed based on step/number of samples
        seed = num_detectors
        for c in pebble_file:
            seed += ord(c)
        np.random.seed(seed)
        
        pebble_ids = np.random.choice(fuel_pebble_ids, num_detectors, replace=False)
        material_assignments = self.pebble_locations["material"].to_list()
        sample_names = []
        sample_text = ""
        detector_id = 0

        volumes = self.get_volumes()



        # Extra features to track per pebble
        list_of_local_graphite_fraction = []
        list_of_local_fima = []
        list_of_conc_dicts = []

        for i in pebble_ids:
            detector_id += 1
            data = self.pebble_locations.iloc[i]
            x = round(data['x'], 4)
            y = round(data['y'], 4)
            z = round(data['z'], 4)
            mat_name = data['material']
            volumes[mat_name] -= fuel_kernel_per_pebble_volume
            sample_mat_name = f"samplefuel{detector_id}_R0Z0G0"
            volumes[sample_mat_name] = fuel_kernel_per_pebble_volume
            material_assignments[i] = sample_mat_name
            sample_names += [sample_mat_name]
            sample_text += f"surf peb{detector_id}_s sph {data['x']} {data['y']} {data['z']} {self.pebble_radius}\n"
            sample_material = deepcopy(self.materials[mat_name])
            sample_material.name = sample_mat_name

            # Collect local values
            local_indices = []

            xsq = (x - self.pebble_locations['x']) ** 2
            ysq = (y - self.pebble_locations['y']) ** 2
            zsq = (z - self.pebble_locations['z']) ** 2
            dist = np.sqrt(xsq + ysq + zsq)
            local_indices = list(dist[dist < self.feature_search_radius].index)

            fima_values = []
            is_graphite = []
            for ind in local_indices:
                data = self.pebble_locations.iloc[ind]
                material_id = data['material']
                if "fuel" in material_id:
                    material_conc = self.materials[material_id].concentrations
                    fima_values += [get_fima(material_conc, self.initial_actinides)]
                    is_graphite += [0]
                else:
                    is_graphite += [1]
            local_fima = np.mean(fima_values)
            local_graphite = np.mean(is_graphite)

            list_of_conc_dicts += [deepcopy(sample_material.concentrations)]
            list_of_local_fima += [local_fima]
            list_of_local_graphite_fraction += [local_graphite]

            sample_text += f"det pebcurrent_{i}_{x}_{y}_{z}_ ds peb{detector_id}_s -1 de finegroup\n"
            sample_text += f"det pebflux_{i}_{x}_{y}_{z}_ dm {sample_mat_name} de finegroup dv {fuel_kernel_per_pebble_volume}\n"

            sample_text += sample_material.write_input(triso_counter, self.static_fuel_materials, debug) + "\n"
            triso_counter += 1

        # Generate cross sections for averaged materials        
        if do_average:
            for mat_name in self.materials.keys():
                if "fuel" in mat_name:
                    mat_volume = volumes[mat_name]
                    if mat_volume > 0:
                        vol_str = np.format_float_scientific(mat_volume,precision=8).replace("e","E")
                        sample_text += f"\n\nset mdep u{mat_name} {vol_str} \n1 {mat_name}  \n10030 16\n30060 102\n30060 103\n30070 16\n30070 102\n50100 102\n50100 103\n50100 107\n50110 16\n50110 102\n50110 103\n50110 107\n60120 102\n60120 103\n60120 107\n80160 16\n80160 102\n80160 103\n80160 107\n340740 16\n340740 102\n340740 103\n340740 107\n340800 16\n340800 17\n340800 102\n340800 103\n340800 107\n340820 16\n340820 17\n340820 102\n340820 103\n340820 107\n350810 16\n350810 17\n350810 102\n350810 103\n350810 107\n360800 16\n360800 102\n360800 103\n360800 107\n360820 16\n360820 17\n360820 102\n360820 103\n360820 107\n360830 16\n360830 17\n360830 102\n360830 103\n360830 107\n360840 16\n360840 17\n360840 102\n360840 103\n360840 107\n360850 16\n360850 17\n360850 102\n360850 103\n360850 107\n360860 16\n360860 17\n360860 102\n360860 103\n360860 107\n370850 16\n370850 17\n370850 102\n370850 103\n370850 107\n370860 16\n370860 17\n370860 102\n370860 103\n370860 107\n370870 16\n370870 17\n370870 102\n370870 103\n370870 107\n380840 16\n380840 102\n380840 103\n380840 107\n380860 16\n380860 102\n380860 103\n380860 107\n380870 16\n380870 102\n380870 103\n380870 107\n380880 16\n380880 17\n380880 102\n380880 103\n380880 107\n380890 16\n380890 17\n380890 102\n380890 103\n380890 107\n380900 16\n380900 17\n380900 102\n380900 103\n380900 107\n390890 16\n390890 102\n390890 103\n390890 107\n390900 16\n390900 17\n390900 102\n390910 16\n390910 17\n390910 102\n390910 103\n390910 107\n400900 16\n400900 102\n400900 103\n400900 107\n400910 16\n400910 17\n400910 102\n400910 103\n400910 107\n400920 16\n400920 17\n400920 102\n400920 103\n400920 107\n400930 16\n400930 17\n400930 102\n400930 103\n400930 107\n400940 16\n400940 17\n400940 102\n400940 103\n400940 107\n400950 16\n400950 17\n400950 102\n400950 103\n400950 107\n400960 16\n400960 17\n400960 102\n400960 103\n400960 107\n410930 16\n410930 17\n410930 102\n410930 103\n410930 107\n410940 16\n410940 17\n410940 102\n410940 103\n410940 107\n410950 16\n410950 17\n410950 102\n410950 103\n410950 107\n420920 16\n420920 102\n420920 103\n420920 107\n420940 16\n420940 17\n420940 102\n420940 103\n420940 107\n420950 16\n420950 17\n420950 102\n420950 103\n420950 107\n420960 16\n420960 17\n420960 102\n420960 103\n420960 107\n420970 16\n420970 17\n420970 102\n420970 103\n420970 107\n420980 16\n420980 17\n420980 102\n420980 103\n420980 107\n420990 16\n420990 17\n420990 102\n420990 103\n420990 107\n421000 16\n421000 17\n421000 102\n421000 103\n421000 107\n430990 16\n430990 17\n430990 102\n430990 103\n430990 107\n440960 16\n440960 102\n440960 103\n440960 107\n440980 16\n440980 17\n440980 102\n440980 103\n440980 107\n440990 16\n440990 17\n440990 102\n440990 103\n440990 107\n441000 16\n441000 17\n441000 102\n441000 103\n441000 107\n441010 16\n441010 17\n441010 102\n441010 103\n441010 107\n441020 16\n441020 17\n441020 102\n441020 103\n441020 107\n441030 16\n441030 17\n441030 102\n441030 103\n441030 107\n441040 16\n441040 17\n441040 102\n441040 103\n441040 107\n441050 16\n441050 17\n441050 102\n441050 103\n441050 107\n441060 16\n441060 17\n441060 102\n441060 103\n441060 107\n451030 16\n451030 17\n451030 102\n451030 103\n451030 107\n451050 16\n451050 17\n451050 102\n451050 103\n451050 107\n461020 16\n461020 17\n461020 102\n461020 103\n461020 107\n461040 16\n461040 17\n461040 102\n461040 103\n461040 107\n461050 16\n461050 17\n461050 102\n461050 103\n461050 107\n461060 16\n461060 17\n461060 102\n461060 103\n461060 107\n461070 16\n461070 17\n461070 102\n461070 103\n461070 107\n461080 16\n461080 17\n461080 102\n461080 103\n461080 107\n461100 16\n461100 17\n461100 102\n461100 103\n461100 107\n471070 16\n471070 102\n471070 103\n471070 107\n471090 16\n471090 17\n471090 102\n471090 103\n471090 107\n471101 16\n471101 17\n471101 102\n471101 103\n471101 107\n481060 16\n481060 17\n481060 102\n481060 103\n481060 107\n481080 16\n481080 17\n481080 102\n481080 103\n481080 107\n481100 16\n481100 17\n481100 102\n481100 103\n481100 107\n481110 16\n481110 17\n481110 102\n481110 103\n481110 107\n501120 16\n501120 17\n501120 102\n501120 103\n501120 107\n501260 16\n501260 17\n501260 102\n501260 103\n501260 107\n511240 16\n511240 17\n511240 102\n511240 103\n511240 107\n521300 16\n521300 17\n521300 102\n521300 103\n521300 107\n521320 16\n521320 17\n521320 102\n521320 103\n521320 107\n531300 16\n531300 17\n531300 102\n531300 103\n531300 107\n531310 16\n531310 17\n531310 102\n531310 103\n531310 107\n531350 16\n531350 17\n531350 102\n531350 103\n531350 107\n541230 16\n541230 17\n541230 102\n541230 103\n541230 107\n541240 16\n541240 17\n541240 102\n541240 103\n541240 107\n541260 16\n541260 17\n541260 102\n541260 103\n541260 107\n541300 16\n541300 17\n541300 102\n541300 103\n541300 107\n541310 16\n541310 17\n541310 102\n541310 103\n541310 107\n541320 16\n541320 17\n541320 102\n541320 103\n541320 107\n541330 16\n541330 17\n541330 102\n541330 103\n541330 107\n541340 16\n541340 17\n541340 102\n541340 103\n541340 107\n541350 16\n541350 17\n541350 102\n541350 103\n541350 107\n541360 16\n541360 17\n541360 102\n541360 103\n541360 107\n551330 16\n551330 17\n551330 102\n551330 103\n551330 107\n551340 16\n551340 17\n551340 102\n551340 103\n551340 107\n551350 16\n551350 17\n551350 102\n551350 103\n551350 107\n551360 16\n551360 17\n551360 102\n551360 103\n551360 107\n551370 16\n551370 17\n551370 102\n551370 103\n551370 107\n561300 16\n561300 17\n561300 102\n561300 103\n561300 107\n561320 16\n561320 17\n561320 102\n561320 103\n561320 107\n561330 16\n561330 17\n561330 102\n561330 103\n561330 107\n561340 16\n561340 17\n561340 102\n561340 103\n561340 107\n561350 16\n561350 17\n561350 102\n561350 103\n561350 107\n561360 16\n561360 17\n561360 102\n561360 103\n561360 107\n561370 16\n561370 17\n561370 102\n561370 103\n561370 107\n561380 16\n561380 17\n561380 102\n561380 103\n561380 107\n561400 16\n561400 17\n561400 102\n561400 103\n561400 107\n571380 16\n571380 17\n571380 102\n571380 103\n571380 107\n571390 16\n571390 17\n571390 102\n571390 103\n571390 107\n571400 16\n571400 17\n571400 102\n571400 103\n571400 107\n581360 16\n581360 17\n581360 102\n581360 103\n581360 107\n581380 16\n581380 17\n581380 102\n581380 103\n581380 107\n581390 16\n581390 17\n581390 102\n581390 103\n581390 107\n581400 16\n581400 17\n581400 102\n581400 103\n581400 107\n581410 16\n581410 17\n581410 102\n581410 103\n581410 107\n581420 16\n581420 17\n581420 102\n581420 103\n581420 107\n581430 16\n581430 17\n581430 102\n581430 103\n581430 107\n581440 16\n581440 17\n581440 102\n581440 103\n581440 107\n591410 16\n591410 17\n591410 102\n591410 103\n591410 107\n591420 16\n591420 17\n591420 102\n591420 103\n591420 107\n591430 16\n591430 17\n591430 102\n591430 103\n591430 107\n601420 16\n601420 17\n601420 102\n601420 103\n601420 107\n601430 16\n601430 17\n601430 102\n601430 103\n601430 107\n601440 16\n601440 17\n601440 102\n601440 103\n601440 107\n601450 16\n601450 17\n601450 102\n601450 103\n601450 107\n601460 16\n601460 17\n601460 102\n601460 103\n601460 107\n601470 16\n601470 17\n601470 102\n601470 103\n601470 107\n601480 16\n601480 17\n601480 102\n601480 103\n601480 107\n601500 16\n601500 17\n601500 102\n601500 103\n601500 107\n611470 16\n611470 17\n611470 102\n611470 103\n611470 107\n611480 16\n611480 17\n611480 102\n611480 103\n611480 107\n611481 16\n611481 17\n611481 102\n611481 103\n611481 107\n611490 16\n611490 17\n611490 102\n611490 103\n611490 107\n621440 16\n621440 17\n621440 102\n621440 103\n621440 107\n621470 16\n621470 17\n621470 102\n621470 103\n621470 107\n621480 16\n621480 17\n621480 102\n621480 103\n621480 107\n621490 16\n621490 17\n621490 102\n621490 103\n621490 107\n621500 16\n621500 17\n621500 102\n621500 103\n621500 107\n621510 16\n621510 17\n621510 102\n621510 103\n621510 107\n631540 16\n631540 17\n631540 102\n631540 103\n631540 107\n661580 16\n661580 17\n661580 102\n661580 103\n661580 107\n681640 16\n681640 17\n681640 102\n681640 103\n681640 107\n902270 16\n902270 17\n902270 18\n902270 37\n902270 102\n902280 16\n902280 17\n902280 18\n902280 102\n902290 16\n902290 17\n902290 18\n902290 37\n902290 102\n902300 16\n902300 17\n902300 18\n902300 102\n902320 16\n902320 17\n902320 18\n902320 102\n902330 16\n902330 17\n902330 18\n902330 102\n902340 16\n902340 17\n902340 18\n902340 102\n912310 16\n912310 17\n912310 18\n912310 102\n912320 16\n912320 17\n912320 18\n912320 37\n912320 102\n912330 16\n912330 17\n912330 18\n912330 102\n922320 16\n922320 17\n922320 102\n922330 16\n922330 17\n922330 18\n922330 37\n922330 102\n922340 16\n922340 17\n922340 37\n922340 102\n922350 16\n922350 17\n922350 18\n922350 37\n922350 102\n922360 16\n922360 17\n922360 37\n922360 102\n922370 16\n922370 17\n922370 18\n922370 37\n922370 102\n922380 16\n922380 17\n922380 18\n922380 37\n922380 102\n922390 16\n922390 17\n922390 18\n922390 37\n922390 102\n922400 16\n922400 17\n922400 37\n922400 102\n922410 16\n922410 17\n922410 18\n922410 37\n922410 102\n932350 16\n932350 17\n932350 18\n932350 102\n932360 16\n932360 17\n932360 18\n932360 102\n932370 16\n932370 17\n932370 18\n932370 37\n932370 102\n932380 16\n932380 17\n932380 18\n932380 102\n932390 16\n932390 17\n932390 18\n932390 102\n942360 16\n942360 17\n942360 18\n942360 102\n942370 16\n942370 17\n942370 102\n942380 16\n942380 17\n942380 102\n942390 16\n942390 17\n942390 18\n942390 37\n942390 102\n942400 16\n942400 17\n942400 102\n942410 16\n942410 17\n942410 18\n942410 102\n942420 16\n942420 17\n942420 18\n942420 102\n942430 16\n942430 17\n942430 18\n942430 37\n942430 102\n942440 16\n942440 17\n942440 37\n942440 102\n952410 16\n952410 17\n952410 18\n952410 37\n952410 102\n952420 16\n952420 17\n952420 18\n952420 102\n952421 16\n952421 17\n952421 18\n952421 102\n952430 16\n952430 17\n952430 18\n952430 37\n952430 102\n952440 16\n952440 17\n952440 18\n952440 37\n952440 102\n952441 16\n952441 17\n952441 18\n952441 37\n952441 102\n962400 16\n962400 17\n962400 18\n962400 102\n962410 16\n962410 17\n962410 102\n962420 16\n962420 17\n962420 102\n962430 16\n962430 17\n962430 102\n962440 16\n962440 17\n962440 18\n962440 102\n962450 16\n962450 17\n962450 18\n962450 102\n962460 16\n962460 17\n962460 102\n962470 16\n962470 17\n962470 18\n962470 102\n962480 16\n962480 17\n962480 37\n962480 102\n962490 16\n962490 17\n962490 18\n962490 102\n962500 16\n962500 17\n962500 18\n962500 102\n"
                        sample_text += f"det avgflux_{mat_name} dm {mat_name} de finegroup dv {mat_volume}\n"

        pebble_conc_df = pd.DataFrame(list_of_conc_dicts).fillna(0.0)

        auxiliary_features = pd.DataFrame({"local_fima": list_of_local_fima,
                                           "local_graphite_frac": list_of_local_graphite_fraction})
        auxiliary_features = pd.concat([auxiliary_features, pebble_conc_df], axis=1)
        auxiliary_features.to_csv(f"current_auxiliary_features{self.iteration}.csv")

        pebble_s = ""
        for i in range(len(self.pebble_locations)):
            peb = self.pebble_locations.iloc[i]
            pebble_s += f"{peb['x']} {peb['y']} {peb['z']} {self.pebble_radius} u{material_assignments[i]}\n"
        with open(pebble_file, "w") as f:
            f.write(pebble_s)

        #sample_name_str = " ".join(sample_names)
        for i in range(1, 1+num_detectors):
            sample_text += f"\n\nset mdep usamplefuel{i}_R0Z0G0 3.626338E-01 \n1 samplefuel{i}_R0Z0G0  \n10030 16\n30060 102\n30060 103\n30070 16\n30070 102\n50100 102\n50100 103\n50100 107\n50110 16\n50110 102\n50110 103\n50110 107\n60120 102\n60120 103\n60120 107\n80160 16\n80160 102\n80160 103\n80160 107\n340740 16\n340740 102\n340740 103\n340740 107\n340800 16\n340800 17\n340800 102\n340800 103\n340800 107\n340820 16\n340820 17\n340820 102\n340820 103\n340820 107\n350810 16\n350810 17\n350810 102\n350810 103\n350810 107\n360800 16\n360800 102\n360800 103\n360800 107\n360820 16\n360820 17\n360820 102\n360820 103\n360820 107\n360830 16\n360830 17\n360830 102\n360830 103\n360830 107\n360840 16\n360840 17\n360840 102\n360840 103\n360840 107\n360850 16\n360850 17\n360850 102\n360850 103\n360850 107\n360860 16\n360860 17\n360860 102\n360860 103\n360860 107\n370850 16\n370850 17\n370850 102\n370850 103\n370850 107\n370860 16\n370860 17\n370860 102\n370860 103\n370860 107\n370870 16\n370870 17\n370870 102\n370870 103\n370870 107\n380840 16\n380840 102\n380840 103\n380840 107\n380860 16\n380860 102\n380860 103\n380860 107\n380870 16\n380870 102\n380870 103\n380870 107\n380880 16\n380880 17\n380880 102\n380880 103\n380880 107\n380890 16\n380890 17\n380890 102\n380890 103\n380890 107\n380900 16\n380900 17\n380900 102\n380900 103\n380900 107\n390890 16\n390890 102\n390890 103\n390890 107\n390900 16\n390900 17\n390900 102\n390910 16\n390910 17\n390910 102\n390910 103\n390910 107\n400900 16\n400900 102\n400900 103\n400900 107\n400910 16\n400910 17\n400910 102\n400910 103\n400910 107\n400920 16\n400920 17\n400920 102\n400920 103\n400920 107\n400930 16\n400930 17\n400930 102\n400930 103\n400930 107\n400940 16\n400940 17\n400940 102\n400940 103\n400940 107\n400950 16\n400950 17\n400950 102\n400950 103\n400950 107\n400960 16\n400960 17\n400960 102\n400960 103\n400960 107\n410930 16\n410930 17\n410930 102\n410930 103\n410930 107\n410940 16\n410940 17\n410940 102\n410940 103\n410940 107\n410950 16\n410950 17\n410950 102\n410950 103\n410950 107\n420920 16\n420920 102\n420920 103\n420920 107\n420940 16\n420940 17\n420940 102\n420940 103\n420940 107\n420950 16\n420950 17\n420950 102\n420950 103\n420950 107\n420960 16\n420960 17\n420960 102\n420960 103\n420960 107\n420970 16\n420970 17\n420970 102\n420970 103\n420970 107\n420980 16\n420980 17\n420980 102\n420980 103\n420980 107\n420990 16\n420990 17\n420990 102\n420990 103\n420990 107\n421000 16\n421000 17\n421000 102\n421000 103\n421000 107\n430990 16\n430990 17\n430990 102\n430990 103\n430990 107\n440960 16\n440960 102\n440960 103\n440960 107\n440980 16\n440980 17\n440980 102\n440980 103\n440980 107\n440990 16\n440990 17\n440990 102\n440990 103\n440990 107\n441000 16\n441000 17\n441000 102\n441000 103\n441000 107\n441010 16\n441010 17\n441010 102\n441010 103\n441010 107\n441020 16\n441020 17\n441020 102\n441020 103\n441020 107\n441030 16\n441030 17\n441030 102\n441030 103\n441030 107\n441040 16\n441040 17\n441040 102\n441040 103\n441040 107\n441050 16\n441050 17\n441050 102\n441050 103\n441050 107\n441060 16\n441060 17\n441060 102\n441060 103\n441060 107\n451030 16\n451030 17\n451030 102\n451030 103\n451030 107\n451050 16\n451050 17\n451050 102\n451050 103\n451050 107\n461020 16\n461020 17\n461020 102\n461020 103\n461020 107\n461040 16\n461040 17\n461040 102\n461040 103\n461040 107\n461050 16\n461050 17\n461050 102\n461050 103\n461050 107\n461060 16\n461060 17\n461060 102\n461060 103\n461060 107\n461070 16\n461070 17\n461070 102\n461070 103\n461070 107\n461080 16\n461080 17\n461080 102\n461080 103\n461080 107\n461100 16\n461100 17\n461100 102\n461100 103\n461100 107\n471070 16\n471070 102\n471070 103\n471070 107\n471090 16\n471090 17\n471090 102\n471090 103\n471090 107\n471101 16\n471101 17\n471101 102\n471101 103\n471101 107\n481060 16\n481060 17\n481060 102\n481060 103\n481060 107\n481080 16\n481080 17\n481080 102\n481080 103\n481080 107\n481100 16\n481100 17\n481100 102\n481100 103\n481100 107\n481110 16\n481110 17\n481110 102\n481110 103\n481110 107\n501120 16\n501120 17\n501120 102\n501120 103\n501120 107\n501260 16\n501260 17\n501260 102\n501260 103\n501260 107\n511240 16\n511240 17\n511240 102\n511240 103\n511240 107\n521300 16\n521300 17\n521300 102\n521300 103\n521300 107\n521320 16\n521320 17\n521320 102\n521320 103\n521320 107\n531300 16\n531300 17\n531300 102\n531300 103\n531300 107\n531310 16\n531310 17\n531310 102\n531310 103\n531310 107\n531350 16\n531350 17\n531350 102\n531350 103\n531350 107\n541230 16\n541230 17\n541230 102\n541230 103\n541230 107\n541240 16\n541240 17\n541240 102\n541240 103\n541240 107\n541260 16\n541260 17\n541260 102\n541260 103\n541260 107\n541300 16\n541300 17\n541300 102\n541300 103\n541300 107\n541310 16\n541310 17\n541310 102\n541310 103\n541310 107\n541320 16\n541320 17\n541320 102\n541320 103\n541320 107\n541330 16\n541330 17\n541330 102\n541330 103\n541330 107\n541340 16\n541340 17\n541340 102\n541340 103\n541340 107\n541350 16\n541350 17\n541350 102\n541350 103\n541350 107\n541360 16\n541360 17\n541360 102\n541360 103\n541360 107\n551330 16\n551330 17\n551330 102\n551330 103\n551330 107\n551340 16\n551340 17\n551340 102\n551340 103\n551340 107\n551350 16\n551350 17\n551350 102\n551350 103\n551350 107\n551360 16\n551360 17\n551360 102\n551360 103\n551360 107\n551370 16\n551370 17\n551370 102\n551370 103\n551370 107\n561300 16\n561300 17\n561300 102\n561300 103\n561300 107\n561320 16\n561320 17\n561320 102\n561320 103\n561320 107\n561330 16\n561330 17\n561330 102\n561330 103\n561330 107\n561340 16\n561340 17\n561340 102\n561340 103\n561340 107\n561350 16\n561350 17\n561350 102\n561350 103\n561350 107\n561360 16\n561360 17\n561360 102\n561360 103\n561360 107\n561370 16\n561370 17\n561370 102\n561370 103\n561370 107\n561380 16\n561380 17\n561380 102\n561380 103\n561380 107\n561400 16\n561400 17\n561400 102\n561400 103\n561400 107\n571380 16\n571380 17\n571380 102\n571380 103\n571380 107\n571390 16\n571390 17\n571390 102\n571390 103\n571390 107\n571400 16\n571400 17\n571400 102\n571400 103\n571400 107\n581360 16\n581360 17\n581360 102\n581360 103\n581360 107\n581380 16\n581380 17\n581380 102\n581380 103\n581380 107\n581390 16\n581390 17\n581390 102\n581390 103\n581390 107\n581400 16\n581400 17\n581400 102\n581400 103\n581400 107\n581410 16\n581410 17\n581410 102\n581410 103\n581410 107\n581420 16\n581420 17\n581420 102\n581420 103\n581420 107\n581430 16\n581430 17\n581430 102\n581430 103\n581430 107\n581440 16\n581440 17\n581440 102\n581440 103\n581440 107\n591410 16\n591410 17\n591410 102\n591410 103\n591410 107\n591420 16\n591420 17\n591420 102\n591420 103\n591420 107\n591430 16\n591430 17\n591430 102\n591430 103\n591430 107\n601420 16\n601420 17\n601420 102\n601420 103\n601420 107\n601430 16\n601430 17\n601430 102\n601430 103\n601430 107\n601440 16\n601440 17\n601440 102\n601440 103\n601440 107\n601450 16\n601450 17\n601450 102\n601450 103\n601450 107\n601460 16\n601460 17\n601460 102\n601460 103\n601460 107\n601470 16\n601470 17\n601470 102\n601470 103\n601470 107\n601480 16\n601480 17\n601480 102\n601480 103\n601480 107\n601500 16\n601500 17\n601500 102\n601500 103\n601500 107\n611470 16\n611470 17\n611470 102\n611470 103\n611470 107\n611480 16\n611480 17\n611480 102\n611480 103\n611480 107\n611481 16\n611481 17\n611481 102\n611481 103\n611481 107\n611490 16\n611490 17\n611490 102\n611490 103\n611490 107\n621440 16\n621440 17\n621440 102\n621440 103\n621440 107\n621470 16\n621470 17\n621470 102\n621470 103\n621470 107\n621480 16\n621480 17\n621480 102\n621480 103\n621480 107\n621490 16\n621490 17\n621490 102\n621490 103\n621490 107\n621500 16\n621500 17\n621500 102\n621500 103\n621500 107\n621510 16\n621510 17\n621510 102\n621510 103\n621510 107\n631540 16\n631540 17\n631540 102\n631540 103\n631540 107\n661580 16\n661580 17\n661580 102\n661580 103\n661580 107\n681640 16\n681640 17\n681640 102\n681640 103\n681640 107\n902270 16\n902270 17\n902270 18\n902270 37\n902270 102\n902280 16\n902280 17\n902280 18\n902280 102\n902290 16\n902290 17\n902290 18\n902290 37\n902290 102\n902300 16\n902300 17\n902300 18\n902300 102\n902320 16\n902320 17\n902320 18\n902320 102\n902330 16\n902330 17\n902330 18\n902330 102\n902340 16\n902340 17\n902340 18\n902340 102\n912310 16\n912310 17\n912310 18\n912310 102\n912320 16\n912320 17\n912320 18\n912320 37\n912320 102\n912330 16\n912330 17\n912330 18\n912330 102\n922320 16\n922320 17\n922320 102\n922330 16\n922330 17\n922330 18\n922330 37\n922330 102\n922340 16\n922340 17\n922340 37\n922340 102\n922350 16\n922350 17\n922350 18\n922350 37\n922350 102\n922360 16\n922360 17\n922360 37\n922360 102\n922370 16\n922370 17\n922370 18\n922370 37\n922370 102\n922380 16\n922380 17\n922380 18\n922380 37\n922380 102\n922390 16\n922390 17\n922390 18\n922390 37\n922390 102\n922400 16\n922400 17\n922400 37\n922400 102\n922410 16\n922410 17\n922410 18\n922410 37\n922410 102\n932350 16\n932350 17\n932350 18\n932350 102\n932360 16\n932360 17\n932360 18\n932360 102\n932370 16\n932370 17\n932370 18\n932370 37\n932370 102\n932380 16\n932380 17\n932380 18\n932380 102\n932390 16\n932390 17\n932390 18\n932390 102\n942360 16\n942360 17\n942360 18\n942360 102\n942370 16\n942370 17\n942370 102\n942380 16\n942380 17\n942380 102\n942390 16\n942390 17\n942390 18\n942390 37\n942390 102\n942400 16\n942400 17\n942400 102\n942410 16\n942410 17\n942410 18\n942410 102\n942420 16\n942420 17\n942420 18\n942420 102\n942430 16\n942430 17\n942430 18\n942430 37\n942430 102\n942440 16\n942440 17\n942440 37\n942440 102\n952410 16\n952410 17\n952410 18\n952410 37\n952410 102\n952420 16\n952420 17\n952420 18\n952420 102\n952421 16\n952421 17\n952421 18\n952421 102\n952430 16\n952430 17\n952430 18\n952430 37\n952430 102\n952440 16\n952440 17\n952440 18\n952440 37\n952440 102\n952441 16\n952441 17\n952441 18\n952441 37\n952441 102\n962400 16\n962400 17\n962400 18\n962400 102\n962410 16\n962410 17\n962410 102\n962420 16\n962420 17\n962420 102\n962430 16\n962430 17\n962430 102\n962440 16\n962440 17\n962440 18\n962440 102\n962450 16\n962450 17\n962450 18\n962450 102\n962460 16\n962460 17\n962460 102\n962470 16\n962470 17\n962470 18\n962470 102\n962480 16\n962480 17\n962480 37\n962480 102\n962490 16\n962490 17\n962490 18\n962490 102\n962500 16\n962500 17\n962500 18\n962500 102\n"

        return sample_text, triso_counter, volumes

    def get_volumes(self, fuel_kernel_per_pebble_volume=0.36263376):
        volumes = {}
        for radial_channel in self.zones:
            for zone in radial_channel:
                for mat_name in zone.inventory.keys():
                    if "fuel" in mat_name:
                        volumes[mat_name] = zone.inventory[mat_name] * fuel_kernel_per_pebble_volume
        return volumes

    def generate_input(self, serpent_settings, num_training_data, burnup_time_step, debug, cr_position):
        input_str = self.core_geometry
        input_str = input_str.replace("<insertion_position>", str(cr_position))
        if num_training_data == 0:
            input_str += f"dep daystep"
            for value in self.substep_schema:
                input_str += f" {round(value * burnup_time_step, 6)}"
            input_str += "\n"
        for setting in serpent_settings.keys():
            input_str += f"set {setting} {serpent_settings[setting]}\n"

        input_str += "\n\n%%%%%%%% Pebble Universe Definition \n\n"

        if num_training_data == 0:
            peb_file_name = f"pebble_positions_{self.iteration}.csv"
            file_name = f"{self.simulation_name}_{self.iteration}.serpent"
            self.generate_pebble_locations(peb_file_name)
            input_str += f"pbed u_pb u_flibe \"{peb_file_name}\" \n"
        else:
            peb_file_name = f"pebble_positions_{self.iteration}_training.csv"
            file_name = f"{self.simulation_name}_training_{self.iteration}.serpent"
            input_str += f"pbed u_pb u_flibe \"{peb_file_name}\"  pow\n"

        input_str += "\n\n%%%%%%%% Material and Pebble Definitions \n\n"
        triso_counter = 1
        for material in self.materials.values():
            if self.is_nonzero_in_core(material.name):
                input_str += material.write_input(triso_counter, self.static_fuel_materials, debug)
                triso_counter += 1



        if num_training_data > 0:
            input_str += "\n\n%%%%%%%% Pebble Current Detectors for ML model \n\n"
            sample_str, triso_counter, modified_volumes = self.generate_sample_pebbles(num_training_data, peb_file_name, triso_counter, debug)
            input_str += sample_str
            volumes = modified_volumes
        else:
            volumes = self.get_volumes()
            
        input_str += "\n\n%%%%%%%% Volume Definitions \n\n"
        input_str += "set mvol\n"
        for mat_name in volumes.keys():
            if volumes[mat_name] > 0:
                vol_str = np.format_float_scientific(volumes[mat_name],precision=8).replace("e","E")
                input_str += f"  {mat_name} 1 {vol_str}\n"

        
        

        with open(file_name, 'w') as f:
            f.write(input_str)

        return file_name

    def is_nonzero_in_core(self, material_name):
        if ("discharge" in material_name) or ("reinsert" in material_name):
            return False
        # Iterate through all zones and return when the material exists in
        # one with a count greater than 1
        for radial_channel in self.zones:
            for zone in radial_channel:
                if material_name in zone.inventory.keys():
                    if zone.inventory[material_name] > 0:
                        return True
        return False

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

    def save_all_materials(self, file_name):
        data_dict = {}
        for key in self.materials.keys():
            data_dict[key] = self.materials[key].concentrations
        if "gz" in file_name:
            json_str = json.dumps(data_dict, indent=1)
            with gzip.open(file_name, 'w') as fout:
                fout.write(json_str.encode())
        else:
            with open(file_name, 'w') as f:
                json.dump(data_dict, f)

    def load_all_materials(self, file_name):
        data_dict = {}
        if "gz" in file_name:
            with gzip.open(f"{file_name}", 'r') as fin:
                data_dict = json.loads(fin.read())
        else:
            with open(file_name, 'r') as f:
                data_dict = json.loads(f)
        for mat_name in data_dict.keys():
            mat_conc = data_dict[mat_name]
            if mat_name in self.materials.keys():
                self.materials[mat_name].concentrations = mat_conc
            else:
                self.materials[mat_name] = Material(mat_name, mat_conc)

    def save_operating_parameters(self, file_name, parameters):
        data_dict = {}
        for key in parameters.keys():
            data_dict[key] = parameters[key]
        with open(file_name, 'w') as f:
            json.dump(data_dict, f)

    def save_discharge_inventory(self, file_name, initial_atoms):
        data_dict = {}
        for key in self.discharge_inventory.inventory.keys():
            data_dict[key] = {}
            data_dict[key]['count'] = self.discharge_inventory.inventory[key]
            previous_fima = self.materials[key].fima
            fima = get_fima(self.materials[key].concentrations, initial_atoms)
            self.materials[key].fima = fima
            data_dict[key]['FIMA'] = fima
            data_dict[key]['FIMA_last_pass'] = fima-previous_fima
            data_dict[key]['concentration'] = self.materials[key].concentrations
        with open(file_name, 'w') as f:
            json.dump(data_dict, f)

    def load_zone_maps(self, file_name):
        with open(file_name, 'r') as f:
            zone_map = json.load(f)
        for zone_key in zone_map.keys():
            # ex: {"zoneR1Z1": {"graphpeb_R1Z1G1": 3281, "fuel20_R1Z1G1": 447},
            radial_zone = int(zone_key.split("R")[1].split("Z")[0]) - 1
            axial_zone = int(zone_key.split("R")[1].split("Z")[1]) - 1
            self.zones[radial_zone][axial_zone].inventory = zone_map[zone_key]

    def update_from_bumat(self, debug, iteration=None, step=1):
        if iteration is None:
            iteration = self.iteration
        bumat_name = f"{self.simulation_name}_{iteration}.serpent.bumat{step}"
        if debug > 0:
            print(f"Reading {bumat_name} and updating zone materials...")
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
                        if current_mat_name in self.materials.keys():
                            self.materials[current_mat_name].concentrations = current_conc
                        else:
                            self.materials[current_mat_name] = Material(current_mat_name, current_conc)
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
        # Handles the last-read material block
        # TODO: Make this block obselete or make these 6 lines of code a function
        if debug > 1:
            print(f"Updating {current_mat_name}")
        if debug > 0:
            print("Complete.")
        if current_mat_name in self.materials.keys():
            self.materials[current_mat_name].concentrations = current_conc
        else:
            self.materials[current_mat_name] = Material(current_mat_name, current_conc)
