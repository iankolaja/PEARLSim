import pickle
import random


from .core import Core
from .material import Material
import numpy as np
from .pebble_model import Pebble_Model
import pandas as pd
import os

class Simulation():
    def __init__(self):
        self.core = Core()
        self.cpu_cores = 1
        self.pebble_model = None
        self.debug = 0
        self.generate_training_data = True
        self.serpent_settings = {"pop": "10000 50 25"}
    def read_input_file(self, input_file):
        with open(input_file, 'r') as f:
            directory_name = input_file.split("/")[-1].split(".")[0]
            try:
                os.chdir(directory_name)
            except:
                os.mkdir(directory_name)
                os.chdir(directory_name)
            for line in f:
                self.read_input_line(line)

    def read_input_line(self, raw_line):
        line = raw_line.split(" ")
        keyword = line[0]

        if keyword == "restart":
            try:
                self.core = pickle.load("../"+line[1])
            except:
                print(f"Failed to load {line[1]}. Does the file exist?")

        if keyword == "fresh":
            fresh_definitions = line[1:]
            num_definitions = int(len(fresh_definitions) / 2)
            for i in range(num_definitions):
                key = fresh_definitions[i * 2]
                file_path = fresh_definitions[i * 2 + 1].replace("\n","")
                self.core.fresh_pebbles[key] = Material(key, "../"+file_path)

        if keyword == "burn_time":
            self.core.burn_time = float(line[1])

        if keyword == "power":
            self.core.power = float(line[1])

        if keyword == "triso_file":
            num_copies = int(line[1])
            file_path = line[2].replace("\n", "")
            try:
                triso_data = pd.read_csv("../"+file_path, sep=" ", names=["x", "y", "z", "r", "id"])
                try:
                    os.mkdir("trisos")
                except:
                    pass
                for i in range(1, 1 + num_copies):
                    triso_data["id"] = f"p{i}"
                    triso_data.to_csv(f"trisos/triso{i}", sep=" ", header=False, index=False)
            except:
                print(f"Failed to load {line[1]}. Does the file exist?")

        if keyword == "static_fuel_materials":
            static_definitions = line[1:]
            num_definitions = int(len(static_definitions) / 2)
            for i in range(num_definitions):
                key = static_definitions[i * 2]
                file_path = static_definitions[i * 2 + 1].replace("\n","")
                self.core.static_fuel_materials[key] = Material(key, "../"+file_path)


        if keyword == "set_seed":
            try:
                print(f"Set random seed to {line[1]}")
                random.seed(int(line[1]))
                np.random.seed(int(line[1]))
            except:
                print(f"Failed to seed random number generator.")

        if keyword == "debug_level":
            try:
                self.debug = int(line[1])
                print(f"Set debug level to {int(line[1])}")
            except:
                print(f"Invalid debug input.")

        if keyword == "define_zones":
            file_path = line[1].replace("\n", "")
            pebble_path = line[2].replace("\n", "")
            #try:
            self.core.define_zones("../"+file_path, "../"+pebble_path, debug=self.debug)
            #except:
            #    print(f"Failed to define zones from {file_path}. Is the file valid?")

        if keyword == "pebble_model":
            model_type = line[1]
            file_path = line[2]
            if model_type == "RFR":
                self.pebble_model = Pebble_Model(model_type, "../"+file_path)

        if keyword == "set_generate_training_data":
            choice = bool(line[1].replace("\n", ""))
            print(f"Training data generation set to {choice}.")
            self.generate_training_data = choice


        if keyword == "core_geometry":
            file_path = line[1].replace("\n", "").replace('\"','')
            try:
                with open("../"+file_path, 'r') as f:
                    self.core.core_geometry = f.read()
                print(f"Loaded {line[1]} core geometry.")
            except:
                    print(f"Failed to load {file_path}. Does the file exist?")

        if keyword == "set_max_passes":
            try:
                self.core.max_passes = int(line[1])
                print(f"Set maximum number of passes to {self.core.max_passes}.")
            except:
                print("Invalid maximum number of passes provided.")

        if keyword == "initialize":
            num_pebbles = int(line[1])
            fuel_ratios = line[2:]
            num_definitions = int(len(fuel_ratios)/2)
            insertion_ratios = {}
            for i in range(num_definitions):
                insertion_ratios[fuel_ratios[i*2]] = float(fuel_ratios[i*2+1])
            for r in range(len(self.core.zones)):
                for zone in self.core.zones[r]:
                    zone.initialize(insertion_ratios, debug=self.debug)
                    self.core.initialize_materials(zone.inventory)
                    print(f"Initialized zone R{zone.radial_num}Z{zone.axial_num}")
            self.pebble_model.distribute_pebbles(num_pebbles, self.core.pebble_locations,
                                                 self.core.fresh_pebbles,insertion_ratios, debug=self.debug)
            print("Finished initializing zones.")


        if keyword == "insertion":
            num_steps = int(line[1])
            threshold = float(line[2])
            run_serpent = bool(int(line[3]))
            fuel_ratios = line[4:]
            num_definitions = int(len(fuel_ratios)/2)
            insertion_ratios = []
            for i in range(num_definitions):
                insertion_ratios += [(fuel_ratios[i * 2], float(fuel_ratios[i * 2 + 1]))]
            for step in range(num_steps):
                self.core.insert(insertion_ratios, threshold, self.pebble_model, debug=self.debug)
                if run_serpent:
                    input_name = self.core.generate_input(self.serpent_settings,
                                                          self.generate_training_data,
                                                          self.debug)
                    os.system(f"sss2_2_0 {input_name} -omp {self.cpu_cores}")
                    self.core.update_from_bumat(self.debug)
                    self.core.iteration += 1

        if keyword == "set":
            setting = line[1]
            value = raw_line.split(setting)[1].replace("\n","")
            if value == "clear":
                self.serpent_settings.pop(setting)
            else:
                self.serpent_settings[setting] = value
