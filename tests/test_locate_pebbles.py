from pearlsim import *
import unittest
import os
import json

class TestLocatePebbles(unittest.TestCase):
    def test_locate_pebbles(self):
        directory_path = "gFHR_equilibrium/" # Look at gFHR_equilibrium directory
        if os.path.exists(directory_path) and os.path.isdir(directory_path): # Check if the directory exists
            items = os.listdir(directory_path) # List items in the directory
            pebble_position_files = [] # Loop through and add each pebble position file to pebble_position_files list
            for item in items:
                if item.startswith("pebble"):
                    pebble_position_files.append(item) # [pebble_positions_1.csv, ...]
        else:
            print(f"The directory '{directory_path}' does not exist or is not a valid directory.")

        iterations = len(pebble_position_files) # Change # of iterations thru pebble and zone files in gFHR folder
        for i in range(iterations):
            pebble_file = f"gFHR_equilibrium/pebble_positions_{i+1}.csv"
            with open(pebble_file, 'r') as f:  # pebble_positions_1.csv
                pebble_position_counts = {}
                #line = f.readline()
                #if line:  # check if the line is not empty 
                for line in f:  # iterate thru each line
                    line = line.replace("\n","")
                    parts = line.split()
                    last_item = parts[-1]  # ufuel20_R1Z1P6
                    zoneP = last_item.split('_')[1]  # R1Z1P6
                    zone = "zone"+zoneP[:-2]  # zoneR1Z1
                    material = last_item[1:]  # fuel20_R1Z1P6
                    if zone not in pebble_position_counts.keys():  # Create zone
                        pebble_position_counts[zone] = {}  # {R1Z1: {}}


                    if material in pebble_position_counts[zone].keys():
                        pebble_position_counts[zone][material] += 1  # Increment material count by 1
                    else:
                        pebble_position_counts[zone][material] = 1  # {R1Z1: {fuel20_zone: 1}}

            #else:
                #print("No data in pebble file")

            zone_file = f"gFHR_equilibrium/zone_map{i+1}.json"
            with open(zone_file, 'r') as json_file:  # zone_map1
                data = json.load(json_file)
                top_level_keys = data.keys()  # R1Z1
                for z in top_level_keys:  # iterate thru zones
                    subdictionary = data[z]  # {"graphpeb_R1Z1P1": 490, "fuel20_R1Z1P2": 80...}
                    # iterate thru each key in the subdictionary
                    for material in subdictionary.keys():
                        # find the same key in pebble_position_counts[zone][material] and assert the values are equal
                        assert subdictionary[material] == pebble_position_counts[z][material], f"{material}: {subdictionary[material]} in {z} in zone file does not match {material} : {pebble_position_counts[z][material]} in {z} in pebble file in iteration {i}"    

if __name__ == '__main__':
    unittest.main()
           
