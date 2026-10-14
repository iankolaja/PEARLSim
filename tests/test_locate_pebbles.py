from pearlsim import *
import unittest
import os
import json

class TestLocatePebbles(unittest.TestCase):
    def test_locate_pebbles(self):
        print(os.getcwd())
        directory_path = "gFHR_sequence" # Look at gFHR_equilibrium directory
        assert os.path.isdir(directory_path), "gFHR_sequence is not a "+\
            "directory that exists. Please run gFHR_sequence.inp."
            
        # Determine the number of iterations dynamically and collect the pebble
        # point files.
        files_in_directory = os.listdir(directory_path)
        pebble_position_files = [] 
        iteration_list = []
        for file_name in files_in_directory:
            if "pebble_positions" in file_name:
                pebble_position_files += [file_name]
                iteration_list += [int("".join(filter(str.isdigit, file_name)))]

        for iteration in iteration_list:
            pebble_file = f"{directory_path}/pebble_positions_{iteration}.csv"
            
            pebble_position_counts = {}
            
            # Identify which material and zone every point in pebble_positions
            # has been assigned.
            with open(pebble_file, 'r') as f: 
                for line in f:  # iterate thru each line
                    line = line.replace("\n","").split()
                    universe_name = line[-1]                # ufuel20_R1Z1P6
                    zone_pass_id = universe_name.split('_')[1]   # R1Z1P6
                    zone_id = "zone"+zone_pass_id[:-2]                # zoneR1Z1
                    material_key = universe_name.split('_')[0][1:] # fuel20_R1Z1P6
                    
                    if zone_id not in pebble_position_counts.keys(): 
                        pebble_position_counts[zone_id] = {} 
                        pebble_position_counts[zone_id][material_key] = 1
                    else:
                        if material_key in pebble_position_counts[zone_id].keys():
                            pebble_position_counts[zone_id][material_key] += 1
                        else:
                            pebble_position_counts[zone_id][material_key] = 1

            # Check the zone_map file, which records the internal representation
            # of zone inventories on each time step.
            zone_file = f"{directory_path}/zone_map{iteration}.json"
            with open(zone_file, 'r') as json_file:  # zone_map1
                zone_map_data = json.load(json_file)
            
            # Iterate through the zone map data and match it against the
            # pebble point data
            zone_map_keys = zone_map_data.keys()  # zoneR1Z1
            for zone_key in zone_map_keys: 
                zone_data = zone_map_data[zone_key]
                for material_key in zone_data.keys():
                    zone_map_count = zone_data[material_key]
                    peb_file_count = pebble_position_counts[zone_key][material_key]
                    error_str = f"[Step {iteration}] Mismatch in {zone_key} with {material_key} -"+ \
                        "{zone_map_count} in zone_map, {peb_file_count} in pebble_positions."
                    assert zone_map_count == peb_file_count, error_str

if __name__ == '__main__':
    unittest.main()
           
