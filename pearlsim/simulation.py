import pickle
import random


from pearlsim.core import Core
from pearlsim.material import Material
from pearlsim.results_processing import read_core_flux
import numpy as np
from pearlsim.pebble_model import Pebble_Model
import pandas as pd
import os

class Simulation():
    def __init__(self, simulation_name):
        self.simulation_name = simulation_name
        self.core = Core(simulation_name)
        self.cpu_cores = 20
        self.num_nodes = 1
        self.days = 0
        self.burnup_time_step = 6.525 # days
        self.fima_discharge_threshold = 17
        self.pebble_model = Pebble_Model()
        self.debug = 0
        self.serpent_settings = {"pop": "10000 50 25"}
    def read_input_file(self, input_file, directory_name=None):
        if directory_name is None:
            directory_name = input_file.split("/")[-1].split(".")[0]
        with open(input_file, 'r') as f:
            try:
                os.chdir(directory_name)
            except:
                os.mkdir(directory_name)
                os.chdir(directory_name)
            for line in f:
                self.read_input_line(line)

    def read_input_line(self, raw_line):
        line = raw_line.split("#")[0] # Ignore comments
        line = line.split(" ")
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
                self.pebble_model.fresh_materials[key] = Material(key, "../"+file_path)

        if keyword == "burn_time":
            self.core.burn_time = float(line[1])

        if keyword == "power":
            self.core.power = float(line[1])

        if keyword == "triso_file":
            num_copies = int(line[1])
            file_path = line[2].replace("\n", "")
            try:
                triso_data = pd.read_csv("../"+file_path, sep=" ", names=["x", "y", "z", "r", "id"])
                if not os.path.exists("trisos"):
                    os.mkdir("trisos")
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
            self.pebble_model.radial_bound_points = self.core.radial_bound_points
            self.pebble_model.axial_zone_bounds = self.core.axial_zone_bounds
            #except:
            #    print(f"Failed to define zones from {file_path}. Is the file valid?")

        if keyword == "load_pebble_model":
            current_model_path = "../" + line[1]
            burnup_model_path = "../" + line[2].replace("\n","")
            print(f"Loading current model from {current_model_path}")
            self.pebble_model.load_current_model(current_model_path, self.cpu_cores)
            print(f"Loading burnup model from {burnup_model_path}")
            self.pebble_model.load_burnup_model(burnup_model_path, self.cpu_cores)

        if keyword == "load_velocity_profile":
            file_path = line[1].replace("\n","")
            print(f"Loading velocity profile from {file_path}")
            self.pebble_model.load_velocity_model("../"+file_path)

        if keyword == "core_geometry":
            file_path = line[1].replace("\n", "").replace('\"','')
            try:
                with open("../"+file_path, 'r') as f:
                    self.core.core_geometry = f.read()
                print(f"Loaded {file_path} core geometry.")
            except:
                    print(f"Failed to load {file_path}. Does the file exist?")

                
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
            self.pebble_model.initialize_pebbles(num_pebbles, self.core.pebble_locations,
                                                insertion_ratios, debug=self.debug)
            print("Finished initializing zones.")


        if keyword == "insertion":
            num_steps = int(line[1])
            couple_flag = get_boolean(line[2])
            serpent_flag = int(line[3]) # 0 for no serpent, 1 to run serpent, 2 to use serpent results that already exist
            insertion_ratios = get_insertion_ratio_pairs(line[4:])
            for step in range(num_steps):
                self.core.insert(insertion_ratios, self.fima_discharge_threshold, self.pebble_model, debug=self.debug)

                bumat_name = f"{self.simulation_name}_{self.core.iteration}.serpent.bumat1"
                
                # If Serpent mode 2 is on, check to see if results for the step
                # already do, and skip it if so.
                if os.path.isfile(bumat_name) and serpent_flag == 2:
                    print(f"{bumat_name} file already exists. Skipping iteration.")
                    self.core.update_from_bumat(self.debug)
                    self.core.iteration += 1
                else:
                    input_name = self.core.generate_input(self.serpent_settings,
                                                          0,
                                                          self.burnup_time_step,
                                                          self.debug)
                    self.run_serpent(input_name)
                    self.core.save_zone_maps(f"zone_map{self.core.iteration}.json")
                    self.core.save_discharge_inventory(f"{self.simulation_name}_discharge_inventory{self.core.iteration}.json")
                    self.core.update_from_bumat(self.debug)
                    self.days += self.burnup_time_step
                    self.core.iteration += 1


        if keyword == "averaging_mode":
            mode = lint[1]
            if mode == "pass":
                self.core.averaging_mode = "pass"
                print("Top zone averaging mode set to pass.")
            elif model == "burnup":
                self.core.averaging_mode = "burnup"
                print("Top zone averaging mode set to burnup.")

        if keyword == "set_threshold":
            self.fima_discharge_threshold = float(line[1])
            print(f"Set discharge fima to {self.fima_discharge_threshold}.")
        
        if keyword == "set_num_fuel_groups":
            self.core.fuel_groups = int(line[1])
            print(f"Set number of fuel groups for averaging to {self.core.fuel_groups}.")

        if keyword == "set_burnup_timestep":
            self.burnup_time_step = float(line[1])
            print(f"Set burnup time step to {self.burnup_time_step} days.")

        if keyword == "set_max_passes":
            try:
                self.core.max_passes = int(line[1])
                print(f"Set maximum number of passes to {self.core.max_passes}.")
            except:
                print("Invalid maximum number of passes provided.")

        if keyword == "load_from_step":
            load_iteration = int(line[1])
            if len(line) > 2:
                burnup_step = int(line[2])
            else:
                burnup_step = 0
            self.load_from_step(load_iteration, burnup_step)


        if keyword == "simulate_pebbles":
            num_steps = int(line[1])
            num_substeps = int(line[2])
            starting_step = int(line[3])
            fuel_ratios = line[4:]
            num_definitions = int(len(fuel_ratios)/2)
            insertion_ratios = {}
            for i in range(num_definitions):
                insertion_ratios[fuel_ratios[i*2]] = float(fuel_ratios[i*2+1])
                
            for i in range(starting_step, num_steps+1):
                print(f"Simulating pebbles (Step {i})")
                core_flux_map, core_flux_avg_unc = read_core_flux(f"{self.simulation_name}_{i}.serpent_det0.m",
                                                                 normalize_and_label=True)
                self.pebble_model.update_model(i, self.burnup_time_step, num_substeps, core_flux_map,
                                               insertion_ratios, self.fima_discharge_threshold, self.debug)

        if keyword == "generate_training_data":
            step = int(line[1])
            self.core.iteration = step
            num_training_data = int(line[2])
            self.load_from_step(step, 0)
            print(f"Generating {num_training_data} points of training data from step {step}.")
            input_name = self.core.generate_input(self.serpent_settings,
                                                  num_training_data,
                                                  self.burnup_time_step,
                                                  self.debug)
            self.run_serpent(input_name)
        
        if keyword == "save":
            core_file = f"{self.simulation_name}_core.pkl"
            pebble_model_file = f"{self.simulation_name}_pebble_model.pkl"
            simulation_file = f"{self.simulation_name}_simulation.pkl"
            
            core_data = pickle.dumps(self.core)
            with open(core_file, 'wb') as f:
                f.write(core_data)
                
            pebmod_data = pickle.dumps(self.pebble_model)
            with open(pebble_model_file, 'wb') as f:
                f.write(pebmod_data)
            
            simulation_data = pickle.dumps(self)
            with open(simulation_file, 'wb') as f:
                f.write(simulation_data)
            


        if keyword == "transport":
            serpent_flag = int(line[1]) # 0 for no serpent, 1 to run serpent, 2 to use serpent results that already exist
            if serpent_flag == 1:
                input_name = self.core.generate_input(self.serpent_settings,
                                                      0,
                                                      self.burnup_time_step,
                                                      self.debug)
                self.run_serpent(input_name)
                self.core.save_zone_maps(f"zone_map{self.core.iteration}.json")
                self.core.update_from_bumat(self.debug)
                self.days += self.burnup_time_step
                self.core.iteration += 1
            elif serpent_flag == 2:
                self.core.update_from_bumat(self.debug)
                self.days += self.burnup_time_step
                self.core.iteration += 1
            elif serpent_flag == 3:
                self.core.update_from_bumat(self.debug)
                self.days += self.burnup_time_step
                input_name = self.core.generate_input(self.serpent_settings,
                                                      0,
                                                      self.burnup_time_step,
                                                      self.debug)
                self.core.save_zone_maps(f"zone_map{self.core.iteration}.json")
                self.core.iteration += 1

        if keyword == "set":
            setting = line[1]
            value = raw_line.split(setting)[1].replace("\n","")
            if value == "clear":
                self.serpent_settings.pop(setting)
            else:
                self.serpent_settings[setting] = value

    def load_from_step(self, load_iteration, burnup_step):
        self.core.load_zone_maps(f"zone_map{load_iteration}.json")
        self.core.load_pebble_locations(f"pebble_positions_{load_iteration}.csv", debug=self.debug)
        for r in range(len(self.core.zones)):
            for zone in self.core.zones[r]:
                zone.initialize({"graphpeb":1}, debug=self.debug)
        print(self.core.materials.keys())
        self.core.update_from_bumat(self.debug, iteration=load_iteration, step=burnup_step)
        print(self.core.materials.keys())

    def run_serpent(self, input_name):
        if self.num_nodes > 1:
            print(f"Running with {self.num_nodes} nodes.")
            os.system(f"mpirun -np {self.num_nodes} --map-by ppr:1:node:pe={self.cpu_cores}"
                      f" sss2_2_0 -omp {self.cpu_cores} {input_name}")
        else:
            print(f"Running with {self.cpu_cores} cores.")
            os.system(f"sss2_2_0 {input_name} -omp {self.cpu_cores}")

def get_boolean(input_argument):
    input_argument = input_argument.replace("\n","")
    if input_argument in ["1", "true", "True", "yes", "Yes", "on"]:
        return True
    elif input_argument in ["0", "false", "False", "no", "No", "off"]:
        return False
    else:
        raise TypeError("Invalid flag value provided.")

def get_insertion_ratio_pairs(list_of_definitions):
    num_definitions = int(len(list_of_definitions) / 2)
    insertion_ratio_pairs = []
    for i in range(num_definitions):
        insertion_ratio_pairs += [(list_of_definitions[i * 2], float(list_of_definitions[i * 2 + 1]))]
    return insertion_ratio_pairs