from pearlsim.results_processing import *
from pearlsim.ml_utilities import extract_from_bumat
from pearlsim.material import Material
from pearlsim.serpent_spectrum_tools import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import time
from numpy.random import poisson
import gzip


start_i = int(input("Starting step number?"))
end_i = int(input("Ending step number?"))
sequence_name = input("Directory/sequence name?")
average_flag = input("Grab averaged features?")
if average_flag in ["True","true","yes","Yes","y","Y"]:
    average_flag = True
elif average_flag in ["False","false","no","No","n","N"]:
    average_flag = False
else:
    print("Invalid flag input. Defaulting to yes.")
    average_flag = True

input_list = []
output_list = []
cumulative_time = [0.0]
simulation_data = {}

for i in range(start_i, end_i+1):
    step_inputs, step_averages, step_outputs, step_meshes, step_simulation_data = collect_results_from_step(sequence_name, i)
    cumulative_time += [cumulative_time[-1] + step_inputs["burnup_step"]]
    simulation_data[i] = step_simulation_data
    if average_flag:
        input_list += [pd.concat([step_inputs,step_averages])]
    else:
        input_list += [step_inputs]
    output_list += [step_outputs]

    

input_df = pd.DataFrame(input_list).astype(float)
input_df[['avg_FIMA']] = input_df[['avg_FIMA']].fillna(value=0)
input_df["cumulative_time"] = cumulative_time[1:]
output_df = pd.DataFrame(output_list).astype(float)

input_df.to_csv(f"sequence_data/{sequence_name}_inputs.csv.gz")

output_df.to_csv(f"sequence_data/{sequence_name}_outputs.csv.gz")

#with open(f"sequence_data/{sequence_name}_simulation_data.json", 'w') as fout:
#    json.dump(simulation_data, fout)

plt.figure(figsize=(8,4.5))
plt.plot(input_df["cumulative_time"], output_df["final_analog_keff"], 
         label="Keff")
plt.plot(input_df["cumulative_time"], input_df["power"]/max(input_df["power"]), 
         label="Power fraction")
plt.plot(input_df["cumulative_time"], (369.47-input_df["control_rod_position"])/(369.47-60.01), 
         label="Control Rod Insertion Fraction")
plt.plot(input_df["cumulative_time"], 1-input_df["graphite_insertion_fraction"], 
         label="Fuel insertion fraction")
plt.legend()
plt.ylabel("Normalized Value")
plt.xlabel("Simulation Time (days)")
plt.savefig(f"sequence_data/{sequence_name}-keff.png")


plt.figure(figsize=(8,4.5))
plt.plot(input_df["cumulative_time"], input_df["avg_FIMA"]/max(input_df["avg_FIMA"]), 
         label="Avg FIMA")
plt.plot(input_df["cumulative_time"], input_df["R1_avg_FIMA_last_pass"]/max(input_df["R1_avg_FIMA_last_pass"]), 
         label="R1 Avg FIMA (last pass)")
plt.plot(input_df["cumulative_time"], input_df["num_discarded"]/max(input_df["num_discarded"]), 
         label="Discarded Pebbles")
plt.ylabel("Normalized Value")
plt.xlabel("Simulation Time (days)")

plt.legend()
plt.savefig(f"sequence_data/{sequence_name}-dependent-inputs.png")


plt.figure(figsize=(8,4.5))
plt.plot(input_df["cumulative_time"], output_df["fluxR1Z10E1"]/max(output_df["fluxR1Z10E1"]), 
         label="Normalized flux in center of core")
plt.plot(input_df["cumulative_time"], input_df["power"]/max(input_df["power"]), 
         label="Power fraction")
plt.plot(input_df["cumulative_time"], 1-input_df["graphite_insertion_fraction"], 
         label="Fuel insertion fraction")
plt.ylabel("Normalized Value")
plt.xlabel("Simulation Time (days)")
plt.legend()
plt.savefig(f"sequence_data/{sequence_name}-flux.png")