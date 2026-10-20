from .zone import Zone, Out_Of_Core_Bin
from .material import Material, shared_component_materials
import numpy as np
from copy import deepcopy
import pandas as pd
import os
import json
from .get_fima import get_fima, heavy_metal_density
from .nuclear_data import ENDF7_V1, CoreConditions, ThermRegistry, render_template
from . import replicas
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


def _fuel_type_and_group(key):
    """('fuel20', 3) from a recirculating material key such as 'avgreinsert_fuel20_G3' or 'held_...'."""
    if key.startswith("held_"):
        key = key[len("held_"):]
    return _get_fuel_type(key), int(key.rsplit("G", 1)[1])


def _largest_remainder(total, weights):
    """Split integer ``total`` in proportion to ``weights`` so the parts sum exactly to ``total``."""
    weights = np.asarray(weights, dtype=float)
    exact = total * weights / weights.sum()
    parts = np.floor(exact).astype(int)
    shortfall = int(total - parts.sum())
    order = np.argsort(-(exact - parts), kind="stable")
    parts[order[:shortfall]] += 1
    return parts


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
        # heavy-metal atom density of fresh fuel; set from the fresh fuel material (set_fresh_fuel)
        self.initial_actinides = 0.00466953 + 0.0189728  # fuel20 U-235 + U-238
        self.materials = {}
        self.fresh_pebbles = {}
        self.pebble_locations = pd.DataFrame([], columns=["x", "y", "z", "r", "zone_r", "zone_z", "material"])
        self.core_geometry = ""
        self.discharge_inventory = Out_Of_Core_Bin()
        self.reinsert_inventory = Out_Of_Core_Bin()
        # recirculating pebbles that did not fit at the bottom; reinserted first next step
        self.held_inventory = {}
        self.always_remove_graphite = True
        # nuclear-data profile and component temperatures used to render decks;
        # conditions None = v1 behaviour (see pearlsim.nuclear_data)
        self.nuclear_data = ENDF7_V1
        self.conditions = None
        # numpy Generator for pebble assignment and shuffling; Simulation seeds it per step
        self.rng = np.random.default_rng()

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

    def insert(self, insert_pebbles, threshold, threshold_rel_std, debug=0):
        num_radial_channels = len(self.zones)
        num_bottom_zone_pebbles = 0

        if debug >= 1:
            print("Clearing discharge inventory...")

        self.reinsert_inventory, rename_map = self.discharge_inventory.to_reinsert(self.averaging_mode)
        self.rename_materials(rename_map)

        # held-out pebbles from earlier steps go back in first
        reinsert_pool = list(self.held_inventory.items()) + self._to_bottom_zone_pebbles(
            list(self.reinsert_inventory.inventory.items()))

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
                rename_map = self.zones[r_zone][z_zone].propagate_from(self.zones[r_zone][z_zone - 1], self.rng)
                self.rename_materials(rename_map)
            num_bottom_zone_pebbles += self.zones[r_zone][0].num_pebbles

        if debug >= 1:
            print(f"Removing spent pebbles from discrete model...")
        removal_fractions = {}
        removed_pebbles = 0
        self.save_discharge_inventory(f"{self.simulation_name}_discharge_inventory{self.iteration}.json",
                                      self.initial_actinides)
        if debug >= 1:
            print(f"Removing spent pebbles from top zones...")
        removed_pebbles = self.discharge_inventory.remove_fractions(removal_fractions, self.always_remove_graphite)
        if debug >= 1:
            print(
                f"Removed {removed_pebbles} pebbles based on fraction of comparable simulated pebbles crossing threshold.")
        if self.averaging_mode == "pass":
            removed_pebbles, removed_inventory = self.discharge_inventory.remove_max_passes(self.max_passes,
                                                                                           self.materials,
                                                                                           self.initial_actinides)
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
        self.held_inventory = self.fill_bottom_zones(insert_pebbles, reinsert_pool)
        with open(f"{self.simulation_name}_held_inventory{self.iteration}.json", "w") as f:
            json.dump(self.held_inventory, f)
        for r_zone in range(num_radial_channels):
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

    def _to_bottom_zone_pebbles(self, pool):
        """Rescale (key, count) pairs discharged from the top zones to bottom-zone pebbles (review #39).

        Zones propagate by fractions, each refilled to its own size, so per step the top zones
        discharge sum(top sizes) pebbles while the bottom zones hold sum(bottom sizes). Without
        this rescaling the difference piles up as held inventory and fresh fuel stops entering.
        """
        top = sum(channel[-1].num_pebbles for channel in self.zones)
        bottom = sum(channel[0].num_pebbles for channel in self.zones)
        total = sum(count for _, count in pool)
        if not pool or top == bottom or total == 0:
            return pool
        counts = _largest_remainder(int(round(total * bottom / top)), [count for _, count in pool])
        return [(key, int(n)) for (key, _), n in zip(pool, counts) if n]

    def fill_bottom_zones(self, insert_pebbles, reinsert_pool):
        """Refill the bottom zone of every radial channel, conserving recirculating pebbles.

        insert_pebbles: ordered (name, fraction) instructions, e.g.
            [("graphpeb", g), ("reinsert", 1.0), ("fuel20", 1.0)].
            A fresh-material fraction is a share of each bottom zone, capped by the space
            left; "reinsert" places the recirculating pool.
        reinsert_pool: ordered (material key, pebble count) pairs; earlier entries have priority.

        The pool is split between channels in proportion to zone size with
        largest-remainder rounding (review #37). Pebbles that do not fit are returned
        as the held inventory, re-keyed ``held_<key>``, to be reinserted first next step
        instead of being lost (review #7). Materials that land on the same zone
        material name are volume-averaged by pebble count.
        """
        bottoms = [channel[0] for channel in self.zones]
        capacity = np.array([zone.num_pebbles for zone in bottoms])
        remaining = capacity.copy()
        sources = [dict() for _ in bottoms]  # zone material name -> {source material key: pebbles}
        held = {}

        def place(r, source_key, fuel_type, group, n):
            n = int(n)
            name = f"{fuel_type}_R{bottoms[r].radial_num}Z{bottoms[r].axial_num}G{group}"
            sources[r].setdefault(name, {})
            sources[r][name][source_key] = sources[r][name].get(source_key, 0) + n
            remaining[r] -= n

        for name, fraction in insert_pebbles:
            if fraction == 0:
                continue
            if name == "reinsert":
                for key, count in reinsert_pool:
                    split = _largest_remainder(count, capacity)
                    leftover = 0
                    for r in range(len(bottoms)):
                        n = min(split[r], remaining[r])
                        leftover += split[r] - n
                        if n:
                            place(r, key, *_fuel_type_and_group(key), n)
                    for r in np.argsort(-remaining, kind="stable"):  # spare room elsewhere
                        n = min(leftover, remaining[r])
                        if n:
                            place(r, key, *_fuel_type_and_group(key), n)
                            leftover -= n
                    if leftover:
                        held_key = key if key.startswith("held_") else f"held_{key}"
                        held[held_key] = held.get(held_key, 0) + int(leftover)
            else:
                for r in range(len(bottoms)):
                    n = min(int(round(fraction * capacity[r])), remaining[r])
                    if n:
                        place(r, name, name, 1, n)

        for r, zone in enumerate(bottoms):
            zone.reset()
            for zone_name, contributions in sources[r].items():
                self._set_mixture(zone_name, contributions)
                zone.inventory[zone_name] = sum(contributions.values())
            zone.num_pebbles_assigned = sum(zone.inventory.values())
            zone.random_assign_remaining(self.rng)  # only if the instructions leave the zone short
        for held_key in held:
            if held_key not in self.materials:
                self.materials[held_key] = deepcopy(self.materials[held_key[len("held_"):]])
                self.materials[held_key].name = held_key
        return held

    def _set_mixture(self, name, contributions):
        """Make ``self.materials[name]`` the pebble-weighted mixture of the contributing materials."""
        lookup = {key: (self.fresh_pebbles[key] if key in self.fresh_pebbles else self.materials[key])
                  for key in contributions}
        if len(contributions) == 1:
            (key,) = contributions
            material = deepcopy(lookup[key])
        else:
            total = sum(contributions.values())
            first = lookup[next(iter(contributions))]
            material = Material(name, {}, temperature=0, density=first.density)
            temperature = 0.0
            for key, n in contributions.items():
                material = material + lookup[key] * (n / total)
                temperature += lookup[key].temperature * n / total
            material.temperature = temperature
        material.name = name
        self.materials[name] = material

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

        # Group = rank of the occupied histogram bin a value falls in, starting at 2
        # (group 1 is fresh fuel). Every occupied bin becomes a group, so no fuel is
        # dropped (review #36: v1 lost fuel when only one bin was occupied, and
        # dropped groups above fuel_groups when the binning overshot).
        if len(fima_values) > 0:
            bin_index = np.clip(np.searchsorted(bins, fima_values, side="right") - 1, 0, len(counts) - 1)
            occupied = sorted(set(bin_index.tolist()))
            group_of_bin = {b: rank + 2 for rank, b in enumerate(occupied)}
            group_assignments = np.array([group_of_bin[b] for b in bin_index.tolist()])
        else:
            group_assignments = np.array([], dtype=int)

        # Go through each type of fuel (if applicable), and each group
        # as defined by binned isotope values
        for g in sorted(set(group_assignments.tolist())):
            for fuel_type in fuel_types:
                grouped_materials = {}
                total_group_pebbles = 0

                pebbles_in_group = np.where(group_assignments == g)[0]
                # Iterate through all pebbles in a group and select the ones
                # of the fuel type currently being constructed
                if len(pebbles_in_group) > 0:
                    for index in pebbles_in_group:
                        key = material_keys[index]
                        if _get_fuel_type(key) == fuel_type:  # exact: 'fuel1' must not match 'fuel15'
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
                    pass_num = key.rsplit("G", 1)[1]

                    # If so, add them to the grouped materials dictionary and count them
                    # (exact fuel type: 'fuel1' must not match 'fuel15')
                    if int(pass_num) == p and _get_fuel_type(key) == fuel_type:
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

    def set_fresh_material(self, key, material):
        """Register a fresh pebble material; fuel sets the initial heavy-metal density used for FIMA."""
        self.fresh_pebbles[key] = material
        if heavy_metal_density(material.concentrations) > 0:
            self.initial_actinides = heavy_metal_density(material.concentrations)

    def initialize_materials(self, zone_inventory):
        for key in zone_inventory.keys():
            fuel_type = key.split("_")[0]
            self.materials[key] = Material(key, self.fresh_pebbles[fuel_type])

    def generate_pebble_locations(self, filename, do_shuffle=True, rng=None):
        rng = self.rng if rng is None else rng
        num_radial_channels = len(self.zones)
        pebble_text = ""
        assigned_pebbles = pd.DataFrame()
        for r_zone in range(num_radial_channels):
            num_axial_zones = len(self.axial_zone_bounds[r_zone]) + 1
            for z_zone in range(num_axial_zones):
                pebble_s, zone_inventory = self.zones[r_zone][z_zone].shuffle_pebbles(self.pebble_radius, rng)
                pebble_text += pebble_s
                assigned_pebbles = pd.concat([assigned_pebbles, zone_inventory])
        with open(filename, 'w') as f:
            f.write(pebble_text)
        return assigned_pebbles

    def get_volumes(self, fuel_kernel_per_pebble_volume=0.36263376):
        volumes = {}
        for radial_channel in self.zones:
            for zone in radial_channel:
                for mat_name in zone.inventory.keys():
                    if "fuel" in mat_name:
                        volumes[mat_name] = zone.inventory[mat_name] * fuel_kernel_per_pebble_volume
        return volumes

    def generate_input(self, serpent_settings, burnup_time_step, debug, cr_position, replica=None, rng=None):
        """Write this step's Serpent deck and pebble file; return the deck name.

        replica: None for the single-replica (v1) file names, else the replica index.
        rng: generator for the pebble shuffle (default: self.rng).
        """
        marker = "%%PEARLSIM_THERM_CARDS%%"
        conditions = self.conditions or CoreConditions()
        registry = ThermRegistry(self.nuclear_data, conditions)
        input_str = render_template(self.core_geometry, self.nuclear_data, conditions, registry, marker)
        input_str = input_str.replace("<insertion_position>", str(cr_position))
        input_str += f"dep daystep"
        for value in self.substep_schema:
            input_str += f" {round(value * burnup_time_step, 6)}"
        input_str += "\n"
        for setting in serpent_settings.keys():
            input_str += f"set {setting} {serpent_settings[setting]}\n"

        input_str += "\n\n%%%%%%%% Pebble Universe Definition \n\n"

        peb_file_name = replicas.pebble_file(self.iteration, replica)
        file_name = replicas.stem(self.simulation_name, self.iteration, replica)
        self.generate_pebble_locations(peb_file_name, rng=rng)
        input_str += f"pbed u_pb u_flibe \"{peb_file_name}\" \n"

        input_str += "\n\n%%%%%%%% Material and Pebble Definitions \n\n"
        if self.conditions is not None and self.conditions.shared_fuel_components and self.static_fuel_materials:
            input_str += shared_component_materials(self.static_fuel_materials, self.nuclear_data, self.conditions,
                                                    registry)
        triso_counter = 1
        for material in self.materials.values():
            if self.is_nonzero_in_core(material.name):
                input_str += material.write_input(triso_counter, self.static_fuel_materials, debug,
                                                  profile=self.nuclear_data, conditions=self.conditions,
                                                  registry=registry)
                triso_counter += 1



        volumes = self.get_volumes()
            
        input_str += "\n\n%%%%%%%% Volume Definitions \n\n"
        input_str += "set mvol\n"
        for mat_name in volumes.keys():
            if volumes[mat_name] > 0:
                vol_str = np.format_float_scientific(volumes[mat_name],precision=8).replace("e","E")
                input_str += f"  {mat_name} 1 {vol_str}\n"

        
        

        if marker in input_str:
            input_str = input_str.replace(marker, registry.cards())
        elif registry.cards():
            raise ValueError("core template has no 'therm grph 900 gre6.18t gre6.20t' line to replace")
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
                data_dict = json.load(f)
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

    def update_from_bumat(self, debug, iteration=None, step=1, num_replicas=1):
        """Set material compositions from a step's .bumat output, averaged over its replicas.

        Returns the replica spread summary ({} for one replica).
        """
        if iteration is None:
            iteration = self.iteration
        names = [f"{replicas.stem(self.simulation_name, iteration, r)}.bumat{step}"
                 for r in replicas.replica_ids(num_replicas)]
        if debug > 0:
            print(f"Reading {', '.join(names)} and updating zone materials...")
        averaged, spread = replicas.average_compositions([replicas.read_bumat(name) for name in names])
        for name, concentrations in averaged.items():
            if name in self.materials:
                self.materials[name].concentrations = concentrations
            else:
                self.materials[name] = Material(name, concentrations)
        if debug > 0:
            print("Complete.")
        return spread
