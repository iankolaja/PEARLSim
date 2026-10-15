import pandas as pd
import matplotlib.pyplot as plt
from pearlsim.results_processing import *
from pearlsim.ml_utilities import extract_from_bumat
from pearlsim.material import Material
from pearlsim.serpent_spectrum_tools import *

sequence_name = input("Sequence name?")
output = f"figures/status_plots/{sequence_name}"
start_i = 11


input_list = []
output_list = []
cumulative_time = [0.0]
simulation_data = {}
average_flag = True

for i in range(start_i, 1000):
    if os.path.exists(f"{sequence_name}/gFHR_equilibrium_{i}.serpent_res.m"):
        step_inputs, step_averages, step_outputs, step_meshes, step_simulation_data = collect_results_from_step(sequence_name, i)
        cumulative_time += [cumulative_time[-1] + step_inputs["burnup_step"]]
        simulation_data[i] = step_simulation_data
        if average_flag:
            input_list += [pd.concat([step_inputs,step_averages])]
        else:
            input_list += [step_inputs]
        output_list += [step_outputs]
    else:
        break


input_df = pd.DataFrame(input_list).astype(float)
input_df[['avg_FIMA']] = input_df[['avg_FIMA']].fillna(value=0)
input_df["cumulative_time"] = cumulative_time[1:]
output_df = pd.DataFrame(output_list).astype(float)


input_df = add_computed_features(input_df)
output_df = add_computed_targets(output_df)


# Preparation of variables
time = input_df["cumulative_time"] 
power = input_df["power"]/280e6
fuel_insertion_fraction = 1-input_df["graphite_insertion_fraction"]
discard_threshold = input_df["fima_discharge_threshold"]/18.63
circulation_rate = 6.525/input_df["burnup_step"]
control_rod_insertion = (369.47-input_df["control_rod_position"])/309.47

num_discarded = input_df["num_discarded"]/max(input_df["num_discarded"])

avg_power_per_pebble = input_df["average_power_per_pebble"]/max(input_df["average_power_per_pebble"])



reactivity = output_df["reactivity"]*1e5
discharge_burnup = input_df["avg_FIMA"]/max(input_df["avg_FIMA"])

thermal_flux_middle = output_df["fluxR1Z5E1"]/max(output_df["fluxR1Z5E1"])
power_middle = output_df["powerR1Z5"]/max(output_df["powerR1Z5"])
fast_flux_lower_corner = output_df["fluxR4Z1E3"]/max(output_df["fluxR4Z1E3"])


plt.rcParams.update({
    "figure.facecolor":  (1.0, 1.0, 1.0, 0.0),  # red   with alpha = 30%
    "axes.facecolor":    (1.0, 1.0, 1.0, 1),  # green with alpha = 50%
    "figure.autolayout": False
})


parameter_color_scheme = {
    "input": "blue",
    "dependent_input": "green",
    "output": "red"
}


key_input_variables_structure = {
    "Power": (power, "input"),
    "Fuel Insertion Fraction": (fuel_insertion_fraction, "input"),
    "Fuel Circulation Rate": (circulation_rate, "input"),
    "Control Rod Insertion": (control_rod_insertion, "input"),
    "Avg. Power Per Pebble": (avg_power_per_pebble, "dependent_input"),
    "Avg. Discharge Burnup": (discharge_burnup, "dependent_input"),
    "Reactivity (pcm)": (reactivity, "output"),
}



def make_good_spaghetti_plot_automation(input_plot_structure, sequence_name, b_show=False, 
                             b_do_time=True, unnormalize_reactivity=False, show_control_labels=True):
    
    axis_num = -1
    number_plots = len(input_plot_structure.keys())
    if show_control_labels:
        fig, ax = plt.subplots(nrows=number_plots, ncols=1, figsize=(8,8.25),sharex=True,gridspec_kw = {'wspace':0, 'hspace':0})
    else:
        fig, ax = plt.subplots(nrows=number_plots, ncols=1, figsize=(5.5,8.25),sharex=True,gridspec_kw = {'wspace':0, 'hspace':0})
    
    
    for primary_key in input_plot_structure.keys():
        axis_num += 1
        for secondary_key in input_plot_structure.keys():
            if primary_key == secondary_key:
                data = input_plot_structure[secondary_key][0]
                parameter_type = input_plot_structure[secondary_key][1]
                color_to_use = parameter_color_scheme[parameter_type]
                ax[axis_num].plot(time, data, color=color_to_use,zorder=20, linewidth=2.5)
            elif secondary_key != "Reactivity (pcm)":
                data = input_plot_structure[secondary_key][0]
                ax[axis_num].plot(time, data, color="gray")
        if axis_num != (number_plots-1):
            ax[axis_num].set_xticks([])
        if primary_key != "Reactivity (pcm)":
            ax[axis_num].set_yticks([0,1])
            ax[axis_num].set_ylim(-0.15, 1.4)
        if show_control_labels:
            ax[axis_num].set_ylabel(primary_key+" ", rotation=0, ha="right", va="center")
    if unnormalize_reactivity:
        max_reactivity = round(np.max(np.abs( input_plot_structure["Reactivity (pcm)"][0]) ),-2)
        ax[axis_num].set_yticks([-max_reactivity,-max_reactivity*0.5,0,max_reactivity*0.5,max_reactivity])
        ax[axis_num].set_ylim(-max_reactivity*1.15, max_reactivity*1.15)
    if b_do_time:
        ax[axis_num].set_xticks(np.linspace(0,round(max(time),-2), 7))
        ax[axis_num].set_xlim([0,round(max(time),-2)])
    fig.tight_layout()
    plt.savefig(f"figures/{sequence_name}.png")
    if b_show:
        plt.show()


make_good_spaghetti_plot_automation(key_input_variables_structure, "status_plots/"+sequence_name+"_runin", b_show=False, unnormalize_reactivity=True, 
                                        b_do_time=True, show_control_labels=True)
