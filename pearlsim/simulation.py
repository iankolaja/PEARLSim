import pickle
import random


from pearlsim.core import Core
from pearlsim.material import Material
from pearlsim.ml_utilities import read_det_file, add_computed_features
from pearlsim.results_processing import read_core_flux, extract_lstm_input_features
from pearlsim.ml_model_wrappers import SingleModelWrapper, XSModelWrapper
from pearlsim.lstm import LSTM_predictor
import numpy as np
from pearlsim.pebble_model import Pebble_Model
import pandas as pd
import os
import json



def get_reactivity_impacts(lstm, baseline_reactivity, control_feature_labels, endpoint_series, control_series, control_step_series, control_min_series, control_max_series, lstm_input_df):
    reactivity_impact_dict = {}
    for feature_label in control_feature_labels:
        print(feature_label)
        if (control_series[feature_label] == endpoint_series[feature_label]) and \
        (feature_label != "burnup_step"):
            reactivity_impact_dict[feature_label] = 0.0
        elif feature_label in ["control_rod_position"]:
            reactivity_impact_dict[feature_label] = 1
        elif feature_label in ["power"]:
            reactivity_impact_dict[feature_label] = -1
        else:
            sample_series = control_series.copy()
            modified_value = sample_series[feature_label] + control_step_series[feature_label]*0.95
            
            if control_min_series[feature_label] <= modified_value <= control_max_series[feature_label]:
                sample_series[feature_label] = modified_value
                lstm_sample_input_df = lstm_input_df.copy()
                lstm_sample_input_df.iloc[-1] = sample_series
                reactivity_final = lstm.predict(lstm_sample_input_df, 
                                ["reactivity"])["reactivity"].iloc[0]
                reactivity_impact_dict[feature_label] = reactivity_final - baseline_reactivity
            else:
                reactivity_impact_dict[feature_label] = 0.0
    return reactivity_impact_dict


def get_reactivity_impacts_updated(lstm, baseline_reactivity, control_feature_labels, control_series, control_step_series, control_min_series, control_max_series, lstm_input_df):
    reactivity_impact_dict = {}
    for feature_label in control_feature_labels:
        print(feature_label)
        if feature_label in ["power"]:
            reactivity_impact_dict[feature_label] = -1
        else:
            sample_series = control_series.copy()
            modified_value = sample_series[feature_label] + control_step_series[feature_label]*0.95
            
            if control_min_series[feature_label] <= modified_value <= control_max_series[feature_label]:
                sample_series[feature_label] = modified_value
                lstm_sample_input_df = lstm_input_df.copy()
                lstm_sample_input_df.iloc[-1] = sample_series
                reactivity_final = lstm.predict(lstm_sample_input_df, 
                                ["reactivity"])["reactivity"].iloc[0]
                reactivity_impact_dict[feature_label] = reactivity_final - baseline_reactivity
            else:
                reactivity_impact_dict[feature_label] = 0.0
    return reactivity_impact_dict



def get_reactivity_impacts_delayed(lstm, predicted_reactivity, control_feature_labels, endpoint_series, control_series, control_step_series, control_min_series, control_max_series, lstm_input_df):
    reactivity_impact_dict = {}

    dependent_input_labels = ["avg_FIMA", "R1_avg_FIMA_last_pass", 
                              "R2_avg_FIMA_last_pass", "R3_avg_FIMA_last_pass", 
                              "R4_avg_FIMA_last_pass", "num_discarded",
                              "burnup_bin1_count", "burnup_bin2_count",
                              "burnup_bin3_count", "burnup_bin4_count",
                              "burnup_bin5_count", "burnup_bin6_count",
                              "burnup_bin7_count", "burnup_bin8_count",
                              "burnup_bin9_count", "avg_FIMA", "average_power_per_pebble"]
    
    stand_dev = lstm.target_std["reactivity"]
    mean = lstm.target_mean["reactivity"]
    lstm_input_slice = lstm_input_df.drop(columns=["cumulative_time"])[-8:].to_numpy().reshape(1, 8, len(lstm_input_df.columns))
    forecasted_reactivity_series = lstm.forecast(lstm_input_slice, 10, dependent_input_labels, predict_labels=["reactivity"])["reactivity"]*stand_dev+mean
    print("Baseline forecast")
    print(forecasted_reactivity_series)
    
    baseline_reactivity = np.mean(forecasted_reactivity_series)

    
    
    for feature_label in control_feature_labels:
        print(feature_label)
        if (control_series[feature_label] == endpoint_series[feature_label]) and \
        (feature_label != "burnup_step"):
            reactivity_impact_dict[feature_label] = 0.0
        #elif feature_label in ["control_rod_position"]:
        #    reactivity_impact_dict[feature_label] = 1
        #elif feature_label in ["power"]:
        #    reactivity_impact_dict[feature_label] = -1
        else:
            sample_series = control_series.copy()
            modified_value = sample_series[feature_label] + control_step_series[feature_label]*0.95
            
            if control_min_series[feature_label] <= modified_value <= control_max_series[feature_label]:
                sample_series[feature_label] = modified_value
                lstm_sample_input_df = lstm_input_df.copy()
                lstm_sample_input_df.iloc[-1] = sample_series
                print("Perturbed forecast")
                
                lstm_sample_input_slice = lstm_sample_input_df.drop(columns=["cumulative_time"])[-8:].to_numpy().reshape(1, 8, len(lstm_sample_input_df.columns))
                reactivity_final_series = lstm.forecast(lstm_sample_input_slice, 10, dependent_input_labels, predict_labels=["reactivity"])["reactivity"]*stand_dev+mean
                reactivity_final = np.mean(reactivity_final_series)
                reactivity_impact_dict[feature_label] = reactivity_final - baseline_reactivity
            else:
                reactivity_impact_dict[feature_label] = 0.0
    return reactivity_impact_dict


class Simulation():
    def __init__(self, simulation_name):
        self.simulation_name = simulation_name
        self.core = Core(simulation_name)
        self.cpu_cores = 20
        self.num_nodes = 1
        self.days = 0
        self.power = 280e6
        self.serpent_command = "sss2_HxF_dev"
        self.rod_insertion = 429.469 # Practically fully removed
        self.burnup_time_step = 6.525 # days
        self.fima_discharge_threshold = 17
        self.fima_discharge_rel_std = 0.02576
        self.pebble_model = Pebble_Model()
        self.debug = 0
        self.serpent_settings = {"pop": "10000 50 25"}

    def get_zone_model_materials(self):
        return self.core.materials

    def get_zone_model_pebble_locations(self):
        return self.core.pebble_locations
    
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

        if keyword == "steer":
            model_path = "../" + line[1]
            print(f"Set LSTM model path to {model_path}.")
            
            constraint_path = line[2]
            print(f"Reading steering instructions from {constraint_path}.")
            
            serpent_flag = int(line[3]) # 0 for no serpent, 1 to run serpent, 2 to use serpent results that already exist
            print(f"Reading steering instructions from {constraint_path}.")
            
            control_feature_labels =["power","burnup_step","graphite_insertion_fraction", 
                         "fima_discharge_threshold", "control_rod_position", "fima_discharge_rel_std"]

            if len(line) > 4:            
                reactivity_tolerance = float(line[4])
                print(f"reactivity tolerance set to {reactivity_tolerance}.")

                max_iterations = int(line[5])
                print(f"Set max iterations to {max_iterations}.")
            else:
                reactivity_tolerance = 50*1E-5 # 50 pcm
                max_iterations = 250
                print(f"Using default max iterations of {max_iterations}.")

            if len(line) > 6:            
                adjustment_forced_steps = int(line[6])
                adjustment_max_iterations = int(line[7])

                print(f"Using minimum adjustment of {adjustment_forced_steps}, maximum {adjustment_max_iterations}.")
            else:

                adjustment_max_iterations = 100
                adjustment_forced_steps = 10
                print(f"Using default minimum adjustment of {adjustment_forced_steps}, maximum {adjustment_max_iterations}.")

            steer_iteration = 0
            instruction_step = 3
            instruction_df = pd.read_csv(os.getcwd()+"/../"+constraint_path,delimiter=",").astype(float)
            num_instructions = len(instruction_df)
            last_step = len(self.core.substep_schema)
            with open(model_path, "rb") as f:
                self.lstm = pickle.load(f)
                
            operation_history = []
            steering_file = open("steering.log", "w")
            steering_file.close()
            for step in range(self.core.iteration-self.lstm.window, self.core.iteration):
                
                operation_history += [extract_lstm_input_features(os.getcwd(), 
                                                                  step, 
                                                                  run_name=self.simulation_name, 
                                                                  merge=True)]
            operation_history_df = pd.DataFrame(operation_history).astype(float)
            operation_history_df = add_computed_features(operation_history_df)
            
            control_step_series = instruction_df.iloc[0]
            control_min_series = instruction_df.iloc[1]
            control_max_series = instruction_df.iloc[2]
            
            
                
            while instruction_step < num_instructions:
                steering_file = open(f"steering_{self.core.iteration}.log", "w")
                if steer_iteration > max_iterations:
                    break
                    
                control_series = operation_history_df.iloc[-1].copy()
                target_reactivity = instruction_df.iloc[instruction_step]["reactivity_target"] 
                endpoint_series = instruction_df.iloc[instruction_step].drop(columns=["reactivity_target"])
                steering_file.write("control series value: \n"+str(endpoint_series[control_feature_labels].round(4))+"\n")
                compare_headers = control_feature_labels.copy()
                compare_headers.remove("burnup_step")
                if np.allclose(control_series[compare_headers].round(4), 
                               endpoint_series[compare_headers].round(4),
                               atol=0.0001):
                    instruction_step += 1
                    continue
                lstm_input_df = operation_history_df
                lstm_input_df[list(control_series.axes[0])].iloc[-1] = control_series
                predicted_reactivity = self.lstm.predict(lstm_input_df, ["reactivity"])["reactivity"].iloc[0]


                reactivity_margin = predicted_reactivity-target_reactivity
                adjustment_step = 0
                
                while (np.abs(reactivity_margin) > reactivity_tolerance) or (adjustment_step < adjustment_forced_steps):
                    steering_file.write(f"[Step {adjustment_step}] reactivity margin: {reactivity_margin}\n")
                    changed_parameter = False
                    adjustment_step += 1
                    if adjustment_step > adjustment_max_iterations:
                        print("Warning: Max number of adjustments reached.")
                        break

                    reactivity_impact_dict = get_reactivity_impacts(self.lstm, predicted_reactivity, 
                        control_feature_labels, endpoint_series, control_series, control_step_series,  control_min_series, control_max_series, lstm_input_df)
                    steering_file.write(f"Reactivity impacts: {reactivity_impact_dict}\n")

                    # If reactivity is predicted to be too high, increase a negative reactivity effect
                    if reactivity_margin > 0:
                        print("Introducing negative reactivity")
                        for feature_label in control_feature_labels:
                            if (reactivity_impact_dict[feature_label] < 0) and \
                            (feature_label != "burnup_step") and \
                            (control_series[feature_label] != endpoint_series[feature_label]):

                                modified_value = control_series[feature_label] + control_step_series[feature_label]
                                if control_min_series[feature_label] <= modified_value <= control_max_series[feature_label]:
                                    control_series[feature_label] = modified_value
                                    print(f"Changed {feature_label} by {control_step_series[feature_label]} [{modified_value}]")
                                    changed_parameter = True
                                else:
                                    if modified_value < control_min_series[feature_label]:
                                        control_series[feature_label] = control_min_series[feature_label]
                                        print(f"Changed {feature_label} to minimum value of {control_min_series[feature_label]}")
                                    elif control_max_series[feature_label] < modified_value:
                                        control_series[feature_label] = control_max_series[feature_label]
                                        print(f"Changed {feature_label} to maximum value of {control_max_series[feature_label]}")
                                    changed_parameter = True

                                

                        if not changed_parameter:
                            control_series["burnup_step"] += control_step_series["burnup_step"]
                            
                    # If reactivity is predicted to be too low, increase a positive reactivity effect
                    else:
                        print("Introducing positive reactivity")
                        for feature_label in control_feature_labels:
                            if (reactivity_impact_dict[feature_label] > 0) and \
                            (feature_label != "burnup_step") and \
                            (control_series[feature_label] != endpoint_series[feature_label]):

                                modified_value = control_series[feature_label] + control_step_series[feature_label]
                                if control_min_series[feature_label] <= modified_value <= control_max_series[feature_label]:
                                    control_series[feature_label] = modified_value
                                    print(f"Changed {feature_label} by {control_step_series[feature_label]} [{modified_value}]")
                                    changed_parameter = True
                                else:
                                    if modified_value < control_min_series[feature_label]:
                                        control_series[feature_label] = control_min_series[feature_label]
                                        print(f"Changed {feature_label} to minimum value of {control_min_series[feature_label]}")
                                    elif control_max_series[feature_label] < modified_value:
                                        control_series[feature_label] = control_max_series[feature_label]
                                        print(f"Changed {feature_label} to maximum value of {control_max_series[feature_label]}")
                                    changed_parameter = True
                                
                                
                        if (not changed_parameter) and \
                        (control_series["burnup_step"] != endpoint_series[feature_label]):
                            
                            modified_value = control_series["burnup_step"] - control_step_series["burnup_step"]
                            if modified_value > control_min_series["burnup_step"]:
                                control_series["burnup_step"] = modified_value
                            else:
                                print("Cannot reduce burnup step / circulation rate any lower.")
                    

                    lstm_input_df.iloc[-1] = control_series
                    predicted_reactivity = self.lstm.predict(lstm_input_df, ["reactivity"])["reactivity"].iloc[0]
                    
                    steering_file.write("\n"+str(control_series)+"\n")
                    steering_file.write(f"Predicted reactivity: {predicted_reactivity} target reactivity: {target_reactivity}\n")

                    reactivity_margin = predicted_reactivity-target_reactivity
                steering_file.write("Done tuning input.")
                steering_file.close()
                self.power = control_series["power"]
                self.serpent_settings['power'] = control_series["power"]
                self.burnup_time_step = control_series["burnup_step"]
                self.fima_discharge_threshold = control_series["fima_discharge_threshold"]
                self.fima_discharge_rel_std = control_series["fima_discharge_rel_std"]
                self.rod_insertion = control_series["control_rod_position"]
                
                insertion_ratios = [("graphpeb", control_series["graphite_insertion_fraction"]),
                                    ("reinsert", 1.0),
                                    ("fuel20", 1.0)]
                
                insertion_dict = {"graphpeb": control_series["graphite_insertion_fraction"],
                                    "reinsert": 1.0,
                                    "fuel20": 1.0}
                                    
            
                self.core.insert(insertion_ratios, self.fima_discharge_threshold,
                    self.fima_discharge_rel_std, self.pebble_model, debug=self.debug)

                
                bumat_name = f"{self.simulation_name}_{self.core.iteration}.serpent.bumat{last_step}"
                zone_file_name = f"zone_map{self.core.iteration}.json"
                
                # If Serpent mode 2 is on, check to see if results for the step
                # already do, and skip it if so.
                if os.path.isfile(bumat_name) and serpent_flag == 2:
                    print(f"{bumat_name} file already exists. Skipping iteration.")
                    self.core.update_from_bumat(self.debug, step=last_step)
                    self.core.iteration += 1
                    operating_parameters = {"power": self.serpent_settings['power'],
                                            "burnup_step": self.burnup_time_step,
                                            "graphite_insertion_fraction": insertion_dict["graphpeb"], 
                                            "control_rod_position": self.rod_insertion,
                                            "fima_discharge_threshold": self.fima_discharge_threshold,
                                            "fima_discharge_rel_std": self.fima_discharge_rel_std
                                            
                    }

                elif os.path.isfile(zone_file_name) and serpent_flag == 0:
                    print(f"{zone_file_name} file already exists. Skipping iteration.")
                    self.core.iteration += 1
                elif serpent_flag == 0:
                    self.core.save_zone_maps(f"zone_map{self.core.iteration}.json")
                    self.core.save_all_materials(f"core_materials_{self.core.iteration}.json.gz")
                    self.core.iteration += 1
                else:
                    input_name = self.core.generate_input(self.serpent_settings,
                                                          0,
                                                          self.burnup_time_step,
                                                          self.debug,
                                                          cr_position=self.rod_insertion)
                    operating_parameters = {"power": self.serpent_settings['power'],
                                            "burnup_step": self.burnup_time_step,
                                            "graphite_insertion_fraction": insertion_dict["graphpeb"], 
                                            "control_rod_position": self.rod_insertion,
                                            "fima_discharge_threshold": self.fima_discharge_threshold,
                                            "fima_discharge_rel_std": self.fima_discharge_rel_std
                                            
                    }
                    file_name = f"{self.core.simulation_name}_operating_{self.core.iteration}.json"
                    self.core.save_operating_parameters(file_name, operating_parameters)
                    


                    
                    self.run_serpent(input_name)
                    self.core.save_zone_maps(zone_file_name)
                    self.core.save_all_materials(f"core_materials_{self.core.iteration}.json.gz")
                    
                    self.core.update_from_bumat(self.debug, step=last_step)
                    self.days += self.burnup_time_step
                    self.core.iteration += 1
                
                operation_history_df = operation_history_df.iloc[1:]
                operation_history_new = extract_lstm_input_features(os.getcwd(), 
                                                                  self.core.iteration-1, 
                                                                  run_name=self.simulation_name, 
                                                                  merge=True).astype(float)
                operation_history_df.loc[self.lstm.window+1] = operation_history_new
                operation_history_df = operation_history_df.reset_index(drop=True)
                operation_history_df = add_computed_features(operation_history_df)
                
                    
                print(operation_history_df)
        


            
        if keyword == "steer_constant_ascent":
            model_path = "../" + line[1]
            print(f"Set LSTM model path to {model_path}.")
            
            constraint_path = line[2]
            print(f"Reading steering instructions from {constraint_path}.")
            
            serpent_flag = int(line[3]) # 0 for no serpent, 1 to run serpent, 2 to use serpent results that already exist
            print(f"Reading steering instructions from {constraint_path}.")
            
            control_feature_labels =["power","burnup_step","graphite_insertion_fraction", 
                         "fima_discharge_threshold", "control_rod_position", "fima_discharge_rel_std"]

           
            reactivity_tolerance = float(line[4])
            print(f"reactivity tolerance set to {reactivity_tolerance}.")


            ascending_control = str(line[5])
            ascending_adjustment_amount = float(line[6])
            final_ascending_value = float(line[7])
            
            
            max_adjustments = int(line[8])

            print(f"Adjusting {ascending_control} by {ascending_adjustment_amount} up to {final_ascending_value}.")


            steer_iteration = 0
            instruction_step = 3
            instruction_df = pd.read_csv(os.getcwd()+"/../"+constraint_path,delimiter=",").astype(float)
            num_instructions = len(instruction_df)
            last_step = len(self.core.substep_schema)
            with open(model_path, "rb") as f:
                self.lstm = pickle.load(f)
                
            operation_history = []
            steering_file = open("steering.log", "w")
            steering_file.close()
            for step in range(self.core.iteration-self.lstm.window, self.core.iteration):
                
                operation_history += [extract_lstm_input_features(os.getcwd(), 
                                                                  step, 
                                                                  run_name=self.simulation_name, 
                                                                  merge=True)]
            operation_history_df = pd.DataFrame(operation_history).astype(float)
            operation_history_df = add_computed_features(operation_history_df)
            
            
            control_step_series = instruction_df.iloc[0]
            control_min_series = instruction_df.iloc[1]
            control_max_series = instruction_df.iloc[2]
            
            ascending_control_value = operation_history_df[ascending_control].iloc[-1]
                
            perturb_feature_labels = control_feature_labels.copy()
            perturb_feature_labels.remove(ascending_control)
        
            while (np.abs(ascending_control_value - final_ascending_value) > 0.001) and \
                (instruction_step < num_instructions):
                steering_file = open(f"ascending_{self.core.iteration}.log", "w")
                control_series = operation_history_df.iloc[-1].copy()
                control_series[ascending_control] += ascending_adjustment_amount
                target_reactivity = instruction_df.iloc[instruction_step]["reactivity_target"] 
                endpoint_series = instruction_df.iloc[instruction_step].drop(columns=["reactivity_target"])
                steering_file.write("control series value: \n"+str(endpoint_series[control_feature_labels].round(4))+"\n")
                compare_headers = control_feature_labels.copy()
                compare_headers.remove("burnup_step")
                if np.allclose(control_series[compare_headers].round(4), 
                               endpoint_series[compare_headers].round(4),
                               atol=0.0001):
                    instruction_step += 1
                    continue
                lstm_input_df = operation_history_df
                lstm_input_df[list(control_series.axes[0])].iloc[-1] = control_series
                predicted_reactivity = self.lstm.predict(lstm_input_df, ["reactivity"])["reactivity"].iloc[0]

                adjustment_step = 0
                reactivity_margin = predicted_reactivity-target_reactivity
                
                while (np.abs(reactivity_margin) > reactivity_tolerance) and (adjustment_step < max_adjustments):
                    adjustment_step += 1
                    steering_file.write(f"[Step {adjustment_step}] reactivity margin: {reactivity_margin}\n")
                    changed_parameter = False

                    reactivity_impact_dict = get_reactivity_impacts(self.lstm, predicted_reactivity, 
                        perturb_feature_labels, endpoint_series, control_series, control_step_series,  control_min_series, control_max_series, lstm_input_df)
                    steering_file.write(f"Reactivity impacts: {reactivity_impact_dict}\n")
                    
                    # If reactivity is predicted to be too high, increase a negative reactivity effect
                    if reactivity_margin > 0:
                        print("Introducing negative reactivity")
                        for feature_label in perturb_feature_labels:
                            if (reactivity_impact_dict[feature_label] < 0) and \
                            (feature_label != "burnup_step") and \
                            (control_series[feature_label] != endpoint_series[feature_label]):

                                modified_value = control_series[feature_label] + control_step_series[feature_label]
                                if control_min_series[feature_label] <= modified_value <= control_max_series[feature_label]:
                                    control_series[feature_label] = modified_value
                                    print(f"Changed {feature_label} by {control_step_series[feature_label]} [{modified_value}]")
                                    changed_parameter = True
                                else:
                                    if modified_value < control_min_series[feature_label]:
                                        control_series[feature_label] = control_min_series[feature_label]
                                        print(f"Changed {feature_label} to minimum value of {control_min_series[feature_label]}")
                                    elif control_max_series[feature_label] < modified_value:
                                        control_series[feature_label] = control_max_series[feature_label]
                                        print(f"Changed {feature_label} to maximum value of {control_max_series[feature_label]}")
                                    changed_parameter = True

                                

                        if not changed_parameter:
                            control_series["burnup_step"] += control_step_series["burnup_step"]
                            
                    # If reactivity is predicted to be too low, increase a positive reactivity effect
                    else:
                        print("Introducing positive reactivity")
                        for feature_label in perturb_feature_labels:
                            if (reactivity_impact_dict[feature_label] > 0) and \
                            (feature_label != "burnup_step") and \
                            (control_series[feature_label] != endpoint_series[feature_label]):

                                modified_value = control_series[feature_label] + control_step_series[feature_label]
                                if control_min_series[feature_label] <= modified_value <= control_max_series[feature_label]:
                                    control_series[feature_label] = modified_value
                                    print(f"Changed {feature_label} by {control_step_series[feature_label]} [{modified_value}]")
                                    changed_parameter = True
                                else:
                                    if modified_value < control_min_series[feature_label]:
                                        control_series[feature_label] = control_min_series[feature_label]
                                        print(f"Changed {feature_label} to minimum value of {control_min_series[feature_label]}")
                                    elif control_max_series[feature_label] < modified_value:
                                        control_series[feature_label] = control_max_series[feature_label]
                                        print(f"Changed {feature_label} to maximum value of {control_max_series[feature_label]}")
                                    changed_parameter = True
                                
                                
                        if (not changed_parameter) and \
                        (control_series["burnup_step"] != endpoint_series[feature_label]):
                            
                            modified_value = control_series["burnup_step"] - control_step_series["burnup_step"]
                            if modified_value > control_min_series["burnup_step"]:
                                control_series["burnup_step"] = modified_value
                            else:
                                print("Cannot reduce burnup step / circulation rate any lower.")
                    

                    lstm_input_df.iloc[-1] = control_series
                    predicted_reactivity = self.lstm.predict(lstm_input_df, ["reactivity"])["reactivity"].iloc[0]
                    
                    steering_file.write("\n"+str(control_series)+"\n")
                    steering_file.write(f"Predicted reactivity: {predicted_reactivity} target reactivity: {target_reactivity}\n")

                    reactivity_margin = predicted_reactivity-target_reactivity
                steering_file.write("Done tuning input.")
                steering_file.close()
                self.power = control_series["power"]
                self.serpent_settings['power'] = control_series["power"]
                self.burnup_time_step = control_series["burnup_step"]
                self.fima_discharge_threshold = control_series["fima_discharge_threshold"]
                self.fima_discharge_rel_std = control_series["fima_discharge_rel_std"]
                self.rod_insertion = control_series["control_rod_position"]
                
                insertion_ratios = [("graphpeb", control_series["graphite_insertion_fraction"]),
                                    ("reinsert", 1.0),
                                    ("fuel20", 1.0)]
                
                insertion_dict = {"graphpeb": control_series["graphite_insertion_fraction"],
                                    "reinsert": 1.0,
                                    "fuel20": 1.0}
                                    
            
                self.core.insert(insertion_ratios, self.fima_discharge_threshold,
                    self.fima_discharge_rel_std, self.pebble_model, debug=self.debug)

                
                bumat_name = f"{self.simulation_name}_{self.core.iteration}.serpent.bumat{last_step}"
                zone_file_name = f"zone_map{self.core.iteration}.json"
                
                # If Serpent mode 2 is on, check to see if results for the step
                # already do, and skip it if so.
                if os.path.isfile(bumat_name) and serpent_flag == 2:
                    print(f"{bumat_name} file already exists. Skipping iteration.")
                    self.core.update_from_bumat(self.debug, step=last_step)
                    self.core.iteration += 1
                    operating_parameters = {"power": self.serpent_settings['power'],
                                            "burnup_step": self.burnup_time_step,
                                            "graphite_insertion_fraction": insertion_dict["graphpeb"], 
                                            "control_rod_position": self.rod_insertion,
                                            "fima_discharge_threshold": self.fima_discharge_threshold,
                                            "fima_discharge_rel_std": self.fima_discharge_rel_std
                                            
                    }

                elif os.path.isfile(zone_file_name) and serpent_flag == 0:
                    print(f"{zone_file_name} file already exists. Skipping iteration.")
                    self.core.iteration += 1
                elif serpent_flag == 0:
                    self.core.save_zone_maps(f"zone_map{self.core.iteration}.json")
                    self.core.save_all_materials(f"core_materials_{self.core.iteration}.json.gz")
                    self.core.iteration += 1
                else:
                    input_name = self.core.generate_input(self.serpent_settings,
                                                          0,
                                                          self.burnup_time_step,
                                                          self.debug,
                                                          cr_position=self.rod_insertion)
                    operating_parameters = {"power": self.serpent_settings['power'],
                                            "burnup_step": self.burnup_time_step,
                                            "graphite_insertion_fraction": insertion_dict["graphpeb"], 
                                            "control_rod_position": self.rod_insertion,
                                            "fima_discharge_threshold": self.fima_discharge_threshold,
                                            "fima_discharge_rel_std": self.fima_discharge_rel_std
                                            
                    }
                    file_name = f"{self.core.simulation_name}_operating_{self.core.iteration}.json"
                    self.core.save_operating_parameters(file_name, operating_parameters)
                    


                    
                    self.run_serpent(input_name)
                    self.core.save_zone_maps(zone_file_name)
                    self.core.save_all_materials(f"core_materials_{self.core.iteration}.json.gz")
                    
                    self.core.update_from_bumat(self.debug, step=last_step)
                    self.days += self.burnup_time_step
                    self.core.iteration += 1
                
                operation_history_df = operation_history_df.iloc[1:]
                operation_history_new = extract_lstm_input_features(os.getcwd(), 
                                                                  self.core.iteration-1, 
                                                                  run_name=self.simulation_name, 
                                                                  merge=True).astype(float)
                operation_history_df.loc[self.lstm.window+1] = operation_history_new
                operation_history_df = operation_history_df.reset_index(drop=True)
                operation_history_df = add_computed_features(operation_history_df)
                
                    
                print(operation_history_df)
                ascending_control_value = operation_history_df[ascending_control].iloc[-1]



        if keyword == "steer_updated":
            model_path = "../" + line[1]
            print(f"Set LSTM model path to {model_path}.")
            
            constraint_path = line[2]
            print(f"Reading steering instructions from {constraint_path}.")
            
            serpent_flag = int(line[3]) # 0 for no serpent, 1 to run serpent, 2 to use serpent results that already exist
            print(f"Reading steering instructions from {constraint_path}.")
            
            control_feature_labels =["power","burnup_step","graphite_insertion_fraction", 
                         "fima_discharge_threshold", "control_rod_position", "fima_discharge_rel_std"]

           
            reactivity_tolerance = float(line[4])
            print(f"reactivity tolerance set to {reactivity_tolerance}.")

            with open(os.getcwd()+"/../"+constraint_path, 'r') as file:
                instruction_dict = json.load(file)
            

            steer_iteration = 0
            instruction_step = 0
            instruction_keys = list(instruction_dict["steps"].keys())
            num_instructions = len(instruction_keys)
            
            last_step = len(self.core.substep_schema)
            with open(model_path, "rb") as f:
                self.lstm = pickle.load(f)
                
            operation_history = []
            
            for step in range(self.core.iteration-self.lstm.window, self.core.iteration):
                
                operation_history += [extract_lstm_input_features(os.getcwd(), 
                                                                  step, 
                                                                  run_name=self.simulation_name, 
                                                                  merge=True)]
            operation_history_df = pd.DataFrame(operation_history).astype(float)
            operation_history_df = add_computed_features(operation_history_df)
            
            

            while instruction_step < num_instructions:
                instruction_key = instruction_keys[instruction_step]
                step_dict = instruction_dict["steps"][instruction_key]
                
                control_labels = instruction_dict["variable_order"]

                control_min_series = pd.Series(step_dict["min"], index = control_labels)
                control_max_series = pd.Series(step_dict["max"], index = control_labels)
                control_step_series = pd.Series(step_dict["step"], index = control_labels)
                
                endpoint_series = pd.Series(step_dict["endpoint"], index = control_labels)
                control_flex_series = pd.Series(step_dict["flexible"], index = control_labels)
                control_is_reversible_series = pd.Series(step_dict["is_reversible"], index = control_labels)
                
                min_adjustment_series = pd.Series(step_dict["min_adjustments"], index = control_labels)
                adjustments_made_series = pd.Series(0,index=control_labels)
                max_adjustment_series = pd.Series(step_dict["max_adjustments"], index = control_labels)
                
                endpoint_labels = endpoint_series.index[endpoint_series != -1]
                steering_file = open(f"ascending_{self.core.iteration}.log", "w")
                

                target_reactivity = step_dict["target_reactivity"]
                
                
                control_series = operation_history_df.iloc[-1].copy()
       
                steering_file.write("Initial value:\n")
                steering_file.write(str(control_series[endpoint_labels])+"\n")
       
                # Check if at endpoint and if we should proceed to next step
                if np.allclose(control_series[endpoint_labels], 
                               endpoint_series[endpoint_labels],
                               atol=0.0001):
                    instruction_step += 1
                    continue
                
                # Increase any controls with minimum perturbation listed
                for label in control_labels:
                    while adjustments_made_series[label] < min_adjustment_series[label]:
                        modified_value = control_series[label] + control_step_series[label]
                        adjustments_made_series[label] += 1
                        if control_min_series[label] <= modified_value <= control_max_series[label]:
                                control_series[label] = modified_value
                
                
                steering_file.write("Post Minimum Perturbation:\n")
                steering_file.write(str(control_series[endpoint_labels])+"\n")
                
                

                lstm_input_df = operation_history_df
                lstm_input_df[list(control_series.axes[0])].iloc[-1] = control_series
                predicted_reactivity = self.lstm.predict(lstm_input_df, ["reactivity"])["reactivity"].iloc[0]


                reactivity_margin = predicted_reactivity-target_reactivity
                flex_level = 1
                max_flex_level = control_flex_series.max()+1
                 
                print((np.abs(reactivity_margin) > reactivity_tolerance))
                print(all(adjustments_made_series <= max_adjustment_series))
                print(adjustments_made_series)
                print(max_adjustment_series)
                print((flex_level < (max_flex_level+1)))
                adjustment_step = 0
                
                while (np.abs(reactivity_margin) > reactivity_tolerance) and \
                    any(adjustments_made_series < max_adjustment_series) and \
                    (flex_level < max_flex_level+1):
                    adjustment_step += 1
                    steering_file.write(f"[Step {adjustment_step}] reactivity margin: {reactivity_margin}\n")
                    changed_parameter = False
                 
                    # Get variables that are considered flexible
                    perturb_feature_labels = control_flex_series.index[control_flex_series == flex_level]
                 
                    reactivity_impact_dict = get_reactivity_impacts_updated(self.lstm, predicted_reactivity, 
                        perturb_feature_labels, control_series, control_step_series,  control_min_series, control_max_series, lstm_input_df)
                    steering_file.write(f"Reactivity impacts: {reactivity_impact_dict}\n")
                
                 
                    if all(adjustments_made_series[perturb_feature_labels] == max_adjustment_series[perturb_feature_labels]) or \
                        all(value == 0 for value in reactivity_impact_dict.values()):
                        flex_level += 1
                        print("Flex level increased")
                        continue
                    
                    
                    
                    # If reactivity is predicted to be too high, increase a negative reactivity effect
                    if reactivity_margin > 0:
                        print("Introducing negative reactivity")
                        for feature_label in perturb_feature_labels:
                            if reactivity_impact_dict[feature_label] < 0:
                                modified_value = control_series[feature_label] + control_step_series[feature_label]
                                
                            elif (reactivity_impact_dict[feature_label] > 0) and (control_is_reversible_series[feature_label]):
                                modified_value = control_series[feature_label] - control_step_series[feature_label]
                            
                            else:
                                modified_value = control_series[feature_label]
                            
                            if (control_min_series[feature_label] <= modified_value <= control_max_series[feature_label]) and \
                                (adjustments_made_series[feature_label] < max_adjustment_series[feature_label]):
                                control_series[feature_label] = modified_value
                                print(f"Changed {feature_label} from {control_series[feature_label]} to {modified_value}")
                            adjustments_made_series[feature_label] += 1
                            
                    else:
                        print("Introducing positive reactivity")
                        for feature_label in perturb_feature_labels:
                            if reactivity_impact_dict[feature_label] > 0:
                                modified_value = control_series[feature_label] + control_step_series[feature_label]
                                
                            elif (reactivity_impact_dict[feature_label] < 0) and (control_is_reversible_series[feature_label]):
                                modified_value = control_series[feature_label] - control_step_series[feature_label]
                            
                            else:
                                modified_value = control_series[feature_label]
                            
                            if (control_min_series[feature_label] <= modified_value <= control_max_series[feature_label]) and  \
                                (adjustments_made_series[feature_label] < max_adjustment_series[feature_label]):
                                control_series[feature_label] = modified_value                                
                                print(f"Changed {feature_label} from {control_series[feature_label]} to {modified_value}")
                            adjustments_made_series[feature_label] += 1
                    

                    lstm_input_df.iloc[-1] = control_series
                    predicted_reactivity = self.lstm.predict(lstm_input_df, ["reactivity"])["reactivity"].iloc[0]
                    
                    steering_file.write("\n"+str(control_series)+"\n")
                    steering_file.write(f"Predicted reactivity: {predicted_reactivity} target reactivity: {target_reactivity}\n")

                    reactivity_margin = predicted_reactivity-target_reactivity
                steering_file.write("Done tuning input.")
                steering_file.close()
                self.power = control_series["power"]
                self.serpent_settings['power'] = control_series["power"]
                self.burnup_time_step = control_series["burnup_step"]
                self.fima_discharge_threshold = control_series["fima_discharge_threshold"]
                self.fima_discharge_rel_std = control_series["fima_discharge_rel_std"]
                self.rod_insertion = control_series["control_rod_position"]
                
                insertion_ratios = [("graphpeb", control_series["graphite_insertion_fraction"]),
                                    ("reinsert", 1.0),
                                    ("fuel20", 1.0)]
                
                insertion_dict = {"graphpeb": control_series["graphite_insertion_fraction"],
                                    "reinsert": 1.0,
                                    "fuel20": 1.0}
                                    
            
                self.core.insert(insertion_ratios, self.fima_discharge_threshold,
                    self.fima_discharge_rel_std, self.pebble_model, debug=self.debug)

                
                bumat_name = f"{self.simulation_name}_{self.core.iteration}.serpent.bumat{last_step}"
                zone_file_name = f"zone_map{self.core.iteration}.json"
                
                # If Serpent mode 2 is on, check to see if results for the step
                # already do, and skip it if so.
                if os.path.isfile(bumat_name) and serpent_flag == 2:
                    print(f"{bumat_name} file already exists. Skipping iteration.")
                    self.core.update_from_bumat(self.debug, step=last_step)
                    self.core.iteration += 1
                    operating_parameters = {"power": self.serpent_settings['power'],
                                            "burnup_step": self.burnup_time_step,
                                            "graphite_insertion_fraction": insertion_dict["graphpeb"], 
                                            "control_rod_position": self.rod_insertion,
                                            "fima_discharge_threshold": self.fima_discharge_threshold,
                                            "fima_discharge_rel_std": self.fima_discharge_rel_std
                                            
                    }

                elif os.path.isfile(zone_file_name) and serpent_flag == 0:
                    print(f"{zone_file_name} file already exists. Skipping iteration.")
                    self.core.iteration += 1
                elif serpent_flag == 0:
                    self.core.save_zone_maps(f"zone_map{self.core.iteration}.json")
                    self.core.save_all_materials(f"core_materials_{self.core.iteration}.json.gz")
                    self.core.iteration += 1
                else:
                    input_name = self.core.generate_input(self.serpent_settings,
                                                          0,
                                                          self.burnup_time_step,
                                                          self.debug,
                                                          cr_position=self.rod_insertion)
                    operating_parameters = {"power": self.serpent_settings['power'],
                                            "burnup_step": self.burnup_time_step,
                                            "graphite_insertion_fraction": insertion_dict["graphpeb"], 
                                            "control_rod_position": self.rod_insertion,
                                            "fima_discharge_threshold": self.fima_discharge_threshold,
                                            "fima_discharge_rel_std": self.fima_discharge_rel_std
                                            
                    }
                    file_name = f"{self.core.simulation_name}_operating_{self.core.iteration}.json"
                    self.core.save_operating_parameters(file_name, operating_parameters)
                    


                    
                    self.run_serpent(input_name)
                    self.core.save_zone_maps(zone_file_name)
                    self.core.save_all_materials(f"core_materials_{self.core.iteration}.json.gz")
                    
                    self.core.update_from_bumat(self.debug, step=last_step)
                    self.days += self.burnup_time_step
                    self.core.iteration += 1
                
                operation_history_df = operation_history_df.iloc[1:]
                operation_history_new = extract_lstm_input_features(os.getcwd(), 
                                                                  self.core.iteration-1, 
                                                                  run_name=self.simulation_name, 
                                                                  merge=True).astype(float)
                operation_history_df.loc[self.lstm.window+1] = operation_history_new
                operation_history_df = operation_history_df.reset_index(drop=True)
                operation_history_df = add_computed_features(operation_history_df)
                
                    
            


        if keyword == "set_substep_schema":
            values = line[1:]
            for i in range(len(values)):
                values[i] = float(values[i].replace("\n",""))
            print(f"Set substep schema to {values}")
            self.core.substep_schema = values


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
            flux_model_path = "../" + line[2]
            xs_model_path = "../" + line[3].replace("\n","")
            print(f"Loading current model from {current_model_path}")
            self.pebble_model.load_current_model(current_model_path)
            print(f"Loading burnup model from {flux_model_path}")
            self.pebble_model.load_flux_model(flux_model_path)
            print(f"Loading xs model from {xs_model_path}")
            self.pebble_model.load_xs_model(xs_model_path)

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
            insertion_ratios, insertion_dict = get_insertion_ratio_pairs(line[4:])
            last_step = len(self.core.substep_schema)
            for step in range(num_steps):
                self.core.insert(insertion_ratios, self.fima_discharge_threshold,
                    self.fima_discharge_rel_std, self.pebble_model, debug=self.debug)

                
                bumat_name = f"{self.simulation_name}_{self.core.iteration}.serpent.bumat{last_step}"
                zone_file_name = f"zone_map{self.core.iteration}.json"
                
                # If Serpent mode 2 is on, check to see if results for the step
                # already do, and skip it if so.
                if os.path.isfile(bumat_name) and serpent_flag == 2:
                    print(f"{bumat_name} file already exists. Skipping iteration.")
                    self.core.update_from_bumat(self.debug, step=last_step)
                    self.core.iteration += 1
                    operating_parameters = {"power": self.serpent_settings['power'],
                                            "burnup_step": self.burnup_time_step,
                                            "graphite_insertion_fraction": insertion_dict["graphpeb"], 
                                            "control_rod_position": self.rod_insertion,
                                            "fima_discharge_threshold": self.fima_discharge_threshold,
                                            "fima_discharge_rel_std": self.fima_discharge_rel_std
                                            
                    }
                    file_name = f"{self.core.simulation_name}_operating_{self.core.iteration}.json"
                    self.core.save_operating_parameters(file_name, operating_parameters)
                elif os.path.isfile(zone_file_name) and serpent_flag == 0:
                    print(f"{zone_file_name} file already exists. Skipping iteration.")
                    self.core.iteration += 1
                elif serpent_flag == 0:
                    self.core.save_zone_maps(f"zone_map{self.core.iteration}.json")
                    self.core.save_all_materials(f"core_materials_{self.core.iteration}.json.gz")
                    self.core.iteration += 1
                else:
                    input_name = self.core.generate_input(self.serpent_settings,
                                                          0,
                                                          self.burnup_time_step,
                                                          self.debug,
                                                          cr_position=self.rod_insertion)
                    operating_parameters = {"power": self.serpent_settings['power'],
                                            "burnup_step": self.burnup_time_step,
                                            "graphite_insertion_fraction": insertion_dict["graphpeb"], 
                                            "control_rod_position": self.rod_insertion,
                                            "fima_discharge_threshold": self.fima_discharge_threshold,
                                            "fima_discharge_rel_std": self.fima_discharge_rel_std
                                            
                    }
                    file_name = f"{self.core.simulation_name}_operating_{self.core.iteration}.json"
                    self.core.save_operating_parameters(file_name, operating_parameters)
                    


                    
                    self.run_serpent(input_name)
                    self.core.save_zone_maps(zone_file_name)
                    self.core.save_all_materials(f"core_materials_{self.core.iteration}.json.gz")
                    
                    self.core.update_from_bumat(self.debug, step=last_step)
                    self.days += self.burnup_time_step
                    self.core.iteration += 1


        if keyword == "set_averaging_mode":
            mode = line[1].replace("\n","")
            if mode == "pass":
                self.core.averaging_mode = "pass"
                print("Top zone averaging mode set to pass.")
            elif mode == "burnup":
                self.core.averaging_mode = "burnup"
                print("Top zone averaging mode set to burnup.")

        if keyword == "set_rod_depth":
            self.rod_insertion = float(line[1].replace("\n",""))
            print(f"Set insertion depth to {self.rod_insertion}.")

        if keyword == "set_discharge_threshold":
            self.fima_discharge_threshold = float(line[1].replace("\n",""))
            print(f"Set discharge fima to {self.fima_discharge_threshold}.")

        if keyword == "set_threshold_rel_std":
            self.fima_discharge_rel_std = float(line[1].replace("\n",""))
            print(f"Set discharge fima to {self.fima_discharge_rel_std}.")

        if keyword == "set_num_fuel_groups":
            self.core.fuel_groups = int(line[1].replace("\n",""))
            print(f"Set number of fuel groups for averaging to {self.core.fuel_groups}.")

        if keyword == "set_burnup_time_step":
            value = line[1].replace("\n","")
            self.burnup_time_step = float(value)
            print(f"Set burnup time step to {value} days.")
            
        if keyword in ["power", "set_power"]:
            value = line[1].replace("\n","")
            self.power = value
            self.serpent_settings['power'] = value
            print(f"Set power to {value}W.")
            

        if keyword == "set_max_passes":
            try:
                self.core.max_passes = int(line[1].replace("\n",""))
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
                
            for i in range(starting_step, starting_step+num_steps+1):
                print(f"Simulating pebbles (Step {i})")
                self.pebble_model.update_model(i, self, self.burnup_time_step,
                                               insertion_ratios, self.fima_discharge_threshold,
                                               self.debug)

        if keyword == "sample_pebbles_from_step":
            load_iteration = int(line[1])
            num_pebbles = int(line[2])
            for r in range(len(self.core.zones)):
                for zone in self.core.zones[r]:
                    zone.initialize({"graphpeb":1}, debug=self.debug)
            self.core.load_zone_maps(f"zone_map{load_iteration}.json")
            self.core.load_pebble_locations(f"pebble_positions_{load_iteration}.csv", debug=self.debug)
            self.core.update_from_bumat(self.debug, iteration=load_iteration, step=0)
            
            self.core.iteration = load_iteration
            self.load_from_step(load_iteration, 0)
            self.pebble_model.sample_pebbles(num_pebbles, self.core.pebble_locations, 
                self.core.materials, debug=self.debug)
            

        if keyword == "generate_training_data":
            step = int(line[1])
            self.core.iteration = step
            num_training_data = int(line[2])
            repeat_flag = get_boolean(line[3])
            
            result_file = f"gFHR_equilibrium_training_{step}.serpent_det0.m"
            if not repeat_flag and os.path.isfile(result_file):
                print(f"{result_file} already found. Skipping...")
            else:
                self.load_from_step(step, 0)
                print(f"Generating {num_training_data} points of training data from step {step}.")
                input_name = self.core.generate_input(self.serpent_settings,
                                                      num_training_data,
                                                      self.burnup_time_step,
                                                      self.debug,
                                                      cr_position=self.rod_insertion)
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
            last_step = len(self.core.substep_schema)
            if serpent_flag == 1:
                input_name = self.core.generate_input(self.serpent_settings,
                                                      0,
                                                      self.burnup_time_step,
                                                      self.debug,
                                                      cr_position=self.rod_insertion)
                operating_parameters = {"power": self.serpent_settings['power'],
                                        "burnup_step": self.burnup_time_step,
                                        "graphite_insertion_fraction": "static", 
                                        "control_rod_position": self.rod_insertion,
                                        "fima_discharge_threshold": self.fima_discharge_threshold,
                                        "fima_discharge_rel_std": self.fima_discharge_rel_std
                                        
                }
                file_name = f"{self.core.simulation_name}_operating_{self.core.iteration}.json"
                self.core.save_operating_parameters(file_name, operating_parameters)
                self.run_serpent(input_name)
                self.core.save_zone_maps(f"zone_map{self.core.iteration}.json")
                self.core.save_all_materials(f"core_materials_{self.core.iteration}.json.gz")
                self.core.update_from_bumat(self.debug, step=last_step)
                self.days += self.burnup_time_step
                self.core.iteration += 1
            elif serpent_flag == 2:
                self.core.update_from_bumat(self.debug, step=last_step)
                self.days += self.burnup_time_step
                self.core.iteration += 1
            elif serpent_flag == 3:
                self.core.update_from_bumat(self.debug, step=last_step)
                self.days += self.burnup_time_step
                input_name = self.core.generate_input(self.serpent_settings,
                                                      0,
                                                      self.burnup_time_step,
                                                      self.debug,
                                                      cr_position=self.rod_insertion)
                self.core.save_zone_maps(f"zone_map{self.core.iteration}.json")
                self.core.iteration += 1

        if keyword == "set":
            setting = line[1]
            value = raw_line.split(setting)[1].replace("\n","")
            if value == "clear":
                self.serpent_settings.pop(setting)
            else:
                self.serpent_settings[setting] = value
            if setting == "power":
                self.power = value

    def load_from_step(self, load_iteration, burnup_step, debug=0):
        for r in range(len(self.core.zones)):
            for zone in self.core.zones[r]:
                zone.initialize({"graphpeb":1}, debug=self.debug)
        self.core.iteration = load_iteration
        self.core.load_zone_maps(f"zone_map{load_iteration}.json")
        self.core.load_pebble_locations(f"pebble_positions_{load_iteration}.csv", debug=self.debug)
        self.core.update_from_bumat(self.debug, iteration=load_iteration, step=burnup_step)
        if debug > 1:
            print("Loaded materials:")
            print(self.core.materials.keys())
            print("Zone status:")
            for r in range(len(self.core.zones)):
                for zone in self.core.zones[r]:
                    zone.print_status()

    def run_serpent(self, input_name):
        if self.num_nodes > 1:
            print(f"Running with {self.num_nodes} nodes.")
            os.system(f"mpirun -np {self.num_nodes} --map-by ppr:1:node:pe={self.cpu_cores}"
                      f" {self.serpent_command} -omp {self.cpu_cores} {input_name}")
        else:
            print(f"Running with {self.cpu_cores} cores.")
            os.system(f"{self.serpent_command} {input_name} -omp {self.cpu_cores}")

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
    insertion_ratio_dict = {}
    for i in range(num_definitions):
        insertion_ratio_pairs += [(list_of_definitions[i * 2], float(list_of_definitions[i * 2 + 1]))]
        insertion_ratio_dict[list_of_definitions[i * 2]] = float(list_of_definitions[i * 2 + 1])
    return insertion_ratio_pairs, insertion_ratio_dict
