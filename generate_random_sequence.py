import pandas as pd
import json
import random

job_number = 20
input_name = f"gFHR_random_{job_number}.inp"

template_path = "/global/scratch/users/ikolaja/PEARLSim_dev/PEARLSim/gFHR_random_template.inp"
with open(template_path, "r") as f:
    input_text = f.read()

sub_template_path = "/global/scratch/users/ikolaja/PEARLSim_dev/PEARLSim/gFHR_random_template.sub"

with open(sub_template_path, "r") as f:
    job_text = f.read()

job_text = job_text.replace("<number>", str(job_number))

with open(f"gFHR_random_{job_number}.sub", "w") as f:
    f.write(job_text)

initial_values = {
    "graphite_insertion_fraction": 0.87,
    "control_rod_position": 369.47,
    "fima_discharge_threshold": 18.63,
    "burnup_step": 6.525,
    "power": 14e6,
}

step_changes = {
    "graphite_insertion_fraction": 0.001,
    "control_rod_position": 1,
    "fima_discharge_threshold": 0.15,
    "burnup_step": 0.04,
    "power": 1.9e6,
}



step_text_template = "set_discharge_threshold <fima_discharge_threshold> \n"+\
    "set_burnup_time_step <burnup_step> \n"+\
    "set_power <power> \n"+\
    "set_rod_depth <control_rod_position> \n"+\
    "insertion <num_steps> off 2 graphpeb <graphite_insertion_fraction> reinsert 1.0 fuel20 1.0 \n\n"

change_probabilities = {'graphite_insertion_fraction': (0.75, 0.15), 
'control_rod_position': (0.85, 0.25), 
'fima_discharge_threshold': (0.25, 0.5), 
'burnup_step': (0.25, 0.5), 
'power': (0.75, 0.75)
} 
possible_condition_lengths = [1,2,3,4,5,7,10] 
possible_change_magnitudes = [1,5,10,20] 
total_steps = 400 

condition_text = f"change_probabilities = {change_probabilities} \n" +\
   f"possible_condition_lengths = {possible_condition_lengths} \n" +\
   f"possible_change_magnitudes = {possible_change_magnitudes} \n" +\
   f"total_steps = {total_steps} \n" 

with open(f"gFHR_random_{job_number}.inp.txt", "w") as f:
    f.write(condition_text)

generated_steps = 0

condition_list = []
conditions = initial_values.copy()

random.seed(job_number)

while generated_steps < total_steps:
    condition_length = random.choice(possible_condition_lengths)
    for control_key in change_probabilities.keys():
        is_changing_parameter = random.random() < change_probabilities[control_key][0]
        if is_changing_parameter:
            is_positive_change = random.random() < change_probabilities[control_key][1]
            changing_magnitude = random.choice(possible_change_magnitudes)
            if is_positive_change:
                sign = 1
            else:
                sign = -1
            parameter_difference = sign*changing_magnitude*step_changes[control_key]
            conditions[control_key] = round(conditions[control_key] + parameter_difference, 5)
    
    # bounds checks:
    if conditions["power"] < 0:
        conditions["power"] = 7e6
    if conditions["graphite_insertion_fraction"] < 0:
        conditions["graphite_insertion_fraction"] = 0
    if conditions["graphite_insertion_fraction"] > 1:
        conditions["graphite_insertion_fraction"] = 1
    if conditions["control_rod_position"] > 369.47:
        conditions["control_rod_position"] = 369.47
    
    # one time directionality switches
    if conditions["power"] > 280e6:
        change_probabilities["power"] = (0.4, 0.5)
        change_probabilities["fima_discharge_threshold"] = (0.5, 0.5)
    if conditions["graphite_insertion_fraction"] < 0.015:
        change_probabilities["graphite_insertion_fraction"] = (0.05, 0.5)
    
    for condition in range(condition_length):
        condition_list += [conditions.copy()]
    step_text = step_text_template.replace("<num_steps>", str(condition_length))
    step_text = step_text.replace("<power>", str(conditions["power"]))
    step_text = step_text.replace("<burnup_step>", str(conditions["burnup_step"]))
    step_text = step_text.replace("<control_rod_position>", str(conditions["control_rod_position"]))
    step_text = step_text.replace("<fima_discharge_threshold>", str(conditions["fima_discharge_threshold"]))
    step_text = step_text.replace("<graphite_insertion_fraction>", str(conditions["graphite_insertion_fraction"]))
    input_text += step_text
    generated_steps += condition_length

condition_df = pd.DataFrame(condition_list)

with open(input_name, "w") as f:
    f.write(input_text)
condition_df.to_csv(f"{input_name}.csv")

        