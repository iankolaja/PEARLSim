import numpy as np
import pandas as pd
from pearlsim.material import get_cross_section_string

# Energy grid that is also defined in Serpent template files
ENERGY_BINS = np.array([1e-11, 5.8e-08, 1.4e-07, 2.8e-07, 6.25e-07, 9.72e-07, 1.15e-06,
                        1.855e-06, 4e-06, 9.877e-06, 1.5968e-05, 0.000148728, 0.00553,
                        0.009118, 0.111, 0.5, 0.821, 2.231, 10])
ENERGY_CENTERS = (ENERGY_BINS[1:] + ENERGY_BINS[:-1])/2


def read_det_file(file_name):
    reading=False
    energy_centers = ENERGY_CENTERS
    skip_names = ["E", "PHI", "Z", "R"]
    id_array = []
    x_array = []
    y_array = []
    z_array = []
    pebble_flux_matrix = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("DET","")
            if "map" in line or "peb" in line:
                if not any([x in line for x in skip_names]):
                    reading = True
                    data_array = []
                    unc_array = []
                    if "peb" in line:
                        header = line.split("_")
                        id_array += [int(header[1])]
                        x_array += [float(header[2])]
                        y_array += [float(header[3])]
                        z_array += [float(header[4])]
                        set_name = "peb"
                    else:
                        set_name = "core_map"
            elif "]" in line and reading:
                reading = False
                if set_name == "core_map":
                    core_flux = data_array
                elif set_name == "peb":
                    pebble_flux_matrix += [data_array]
            elif reading:
                data = float(line.split()[10])
                unc = float(line.split()[11])
                data_array += [data]
                unc_array += [unc]
    pebble_flux_matrix = np.array(pebble_flux_matrix)
    features = pd.DataFrame({"x": x_array, "y": y_array, "z": z_array })
    num_detectors = len(features)
    core_flux_headers = ["bin" + str(n) for n in range(1, 1 + len(core_flux))]
    features = pd.concat([features, pd.DataFrame([core_flux]*num_detectors,
                          columns=core_flux_headers)], axis=1)
    targets = pd.DataFrame(pebble_flux_matrix, columns=energy_centers)
    avg_uncertainty = np.mean(np.array(unc_array))
    return features, targets, id_array, avg_uncertainty


def extract_from_bumat(file_path):
    concentrations = []
    with open(file_path, 'r') as f:
        reading = False
        first_mat = True
        for line in f:
            line = line.split()
            if len(line) == 0:
                continue
            if line[0] == "mat":
                if not first_mat:
                    concentrations += [current_conc]
                current_mat_name = line[1].split("pp")[0]
                current_conc = {}
                reading = True
                first_mat = False
            elif reading:
                id = line[0].split(".")
                if len(id) > 1:
                    nuclide = id[0] + "<lib>"
                else:
                    nuclide = id[0]
                amount = float(line[1].replace("\n", ""))
                current_conc[nuclide] = amount
    concentrations += [current_conc]
    return concentrations

def generate_pebble_burnup_model(template_path, surface_current, power, concentrations, temp, time):
    lib_str = get_cross_section_string(temp)
    energy_bins = ENERGY_BINS
    with open(template_path, "r") as f:
        input_s = f.read()
    concentration_s = ""
    for key in concentrations.keys():
        concentration_s  += f"  {key}    {concentrations[key]}\n".replace("<lib>", lib_str)
    input_s = input_s.replace("<concentrations>", concentration_s)
    input_s = input_s.replace("<temperature>", str(temp))
    input_s = input_s.replace("<time>", str(time))
    input_s = input_s.replace("<power>", str(power))

    weights = [0]+list((surface_current/1e9).astype(int))
    source_str = f"sb {len(weights)} 1\n"
    for i in range(len(surface_current)+1):
        source_str += f"  {energy_bins[i]} {weights[i]}\n"
    input_s = input_s.replace("<current>", source_str)
    return input_s