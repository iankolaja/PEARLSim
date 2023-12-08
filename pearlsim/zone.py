import pandas as pd
import random
import numpy as np

def _get_fuel_type(mat_name):
    name = mat_name.split("_")
    if len(name) > 2:
        fuel_type = name[1]
    else:
        fuel_type = name[0]
    return fuel_type

class Zone():
    def __init__(self,num_pebbles,radial_num,axial_num):
        self.num_pebbles = num_pebbles
        self.num_pebbles_assigned = 0
        self.radial_num = radial_num
        self.axial_num = axial_num
        self.pebble_locations = []
        self.inventory = {}
        #self.pebble_locations = pd.df
    def get_fractions(self, as_str=False):
        fractions = {}
        for key in self.inventory.keys():
            if as_str:
                fractions[key] = str(round(100*self.inventory[key] / self.num_pebbles,2))
            else:
                fractions[key] = self.inventory[key] / self.num_pebbles
        return fractions

    def reset(self):
        self.num_pebbles_assigned = 0
        self.inventory = {}

    def initialize(self, insertion_fractions, debug=0):
        self.num_pebbles = len(self.pebble_locations)
        self.pebble_locations = pd.DataFrame(self.pebble_locations, columns=["x", "y", "z", "r"])
        self.inventory = {}
        for key in insertion_fractions.keys():
            #if "fuel" in key:
            #for p in range(num_passes):
            fuel_name = f"{key}_R{self.radial_num}Z{self.axial_num}P{1}"
            self.inventory[fuel_name] = int(self.num_pebbles*insertion_fractions[key])
            #else:
            #    mat_name = f"{key}_R{self.radial_num}Z{self.axial_num}"
            #    self.inventory[mat_name] = int(round(self.num_pebbles * insertion_fractions[key]))
        while sum(self.inventory.values()) < self.num_pebbles:
            self.inventory[random.choice( list(self.inventory.keys()) )] += 1
        if debug >= 1:
            self.print_status()
    def set_fractions(self, fractions, reset=False):
        if reset:
            self.reset()
        rename_map = {}
        remaining_unassigned_pebbles = 0
        for key in fractions.keys():
            try:
                pass_number = int(key.split("P")[1])
            except:
                pass_number = 1
            mat_base_name = _get_fuel_type(key)
            new_mat_name = f"{mat_base_name}_R{self.radial_num}Z{self.axial_num}P{pass_number}"
            num_pebbles_to_set = int(round(fractions[key]*self.num_pebbles))
            remaining_unassigned_pebbles = self.num_pebbles-self.num_pebbles_assigned
            if num_pebbles_to_set > remaining_unassigned_pebbles:
                num_pebbles_to_set = remaining_unassigned_pebbles
            self.inventory[new_mat_name] = num_pebbles_to_set
            self.num_pebbles_assigned += num_pebbles_to_set
            rename_map[key] = new_mat_name
        return rename_map



    def remove_graphite_pebbles(self, name_base="graphpeb"):
        removed_pebbles = 0
        for mat_name in self.inventory.keys():
            if name_base in mat_name:
                removed_pebbles = self.inventory[mat_name]
                self.inventory[mat_name] = 0
        return removed_pebbles

    def random_assign_remaining(self):
        random_options = {}
        for key in self.inventory.keys():
            random_options[key] = self.inventory[key]
        while self.num_pebbles > sum(self.inventory.values()):
            key = random.choices( list(random_options.keys()),
                                  weights=list(random_options.values()) )[0]
            self.inventory[key] += 1


    def propagate_from(self, other_zone):
        other_fraction = other_zone.get_fractions()
        rename_map = self.set_fractions(other_fraction, reset=True)
        self.random_assign_remaining()
        return rename_map

    def insert(self, insert_fracs, reinsert_inventory):
        self.reset()
        rename_map = {}
        for instruction in insert_fracs:
            insert_name, insert_amount = instruction
            if insert_amount == 0:
                continue
            if insert_name == "reinsert":
                rename_step = self.set_fractions(reinsert_inventory)
            else:
                rename_step = self.set_fractions({insert_name: insert_amount})
            rename_map.update(rename_step)
        self.random_assign_remaining()
        return rename_map

    def print_status(self):
        status_s = f"Zone R{self.radial_num}Z{self.axial_num}: [{self.num_pebbles} pebble points, {sum(self.inventory.values())} assigned]\n"
        status_s += str(self.inventory)
        status_s += "\n"
        status_s += str(self.get_fractions(as_str=True))
        print(status_s)

    def shuffle_pebbles(self, peb_radius):
        pebble_s = ""
        pebble_choices = []
        for key in self.inventory.keys():
            pebble_choices += [key]*self.inventory[key]
        assignments = np.random.choice(pebble_choices, self.num_pebbles, replace=False)
        self.pebble_locations["material"] = assignments
        for i in range(len(self.pebble_locations)):
            peb = self.pebble_locations.iloc[i]
            pebble_s += f"{peb['x']} {peb['y']} {peb['z']} {peb_radius} u{peb['material']}\n"
        return pebble_s, self.pebble_locations

class Out_Of_Core_Bin():
    def __init__(self):
        self.inventory = {}
        self.num_pebbles = 0

   # def increment_pass(self):
   #     rename_map = {}
   #     new_inventory = {}
   #     for key in list(self.inventory.keys()):
   #         split_key = key.split("P")
   #         if len(split_key) > 1:
   #             pass_number = int(split_key[1]) + 1
   #             new_key = f"{split_key[0]}P{pass_number}"
   #             new_inventory[new_key] = self.inventory[key]
   #             rename_map[key] = new_key
   #     self.inventory = new_inventory
   #     self.num_pebbles = sum(self.inventory.values())
   #     return rename_map

    def to_reinsert(self, do_increment=True):
        reinsert_bin = Out_Of_Core_Bin()
        rename_map = {}
        new_inventory = {}
        for key in list(self.inventory.keys()):
            split_key = key.split("P")
            new_name = split_key[0].replace("discharge","reinsert")
            if len(split_key) > 1:
                pass_number = int(split_key[1])
                if do_increment:
                     pass_number += 1
                new_key = f"{new_name}P{pass_number}"
                new_inventory[new_key] = self.inventory[key]
                rename_map[key] = new_key
        reinsert_bin.inventory = new_inventory
        reinsert_bin.num_pebbles = sum(new_inventory.values())
        return reinsert_bin, rename_map
        

    def clear(self):
        self.inventory = {}
        self.num_pebbles = 0

    def extract_top_zone(self, other_zone):
        rename_map = {}
        for key in other_zone.inventory.keys():
            discharge_mat_name = "discharge_" + key
            self.inventory[discharge_mat_name] = other_zone.inventory[key]
            self.num_pebbles += other_zone.inventory[key]
            rename_map[key] = discharge_mat_name
        return rename_map

    def __mul__(self, coeff):
        multiplied_inventory = {}
        for key in self.inventory.keys():
            multiplied_inventory[key] = self.inventory[key]*coeff
        return multiplied_inventory

    def remove_fractions(self, fractions, graphite_removal_flag, base_graphite_name = "graphpeb"):
        removed_pebbles = 0
        for key in self.inventory.keys():
            if key in fractions.keys():
                starting_pebbles = self.inventory[key]
                self.inventory[key] = int(round(self.inventory[key]*(1-fractions[key])))
                removed_pebbles += starting_pebbles - self.inventory[key]
            if base_graphite_name in key and graphite_removal_flag:
                #removed_pebbles += self.inventory[key]
                self.inventory[key] = 0
        self.num_pebbles -= removed_pebbles
        return removed_pebbles


    def remove_max_passes(self, max_passes):
        removed_pebbles = 0
        new_inventory = {}
        for key in self.inventory.keys():
            split_key = key.split("P")
            if len(split_key) > 1:
                pass_number = int(split_key[1]) + 1
                if pass_number <= max_passes:
                    new_inventory[key] = self.inventory[key]
                else:
                    removed_pebbles += self.inventory[key]
        self.num_pebbles -= removed_pebbles
        self.inventory = new_inventory
        return removed_pebbles
