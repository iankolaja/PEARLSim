import numpy as np
import pandas as pd


def read_det_file(file_name, energy_bins_lower=(0.0, 3e-08, 5.8e-08, 1.4e-07, 2.8e-07, 3.5e-07, 6.25e-07,
                                            4e-06, 4.8052e-05, 0.00553, 0.821, 2.231)):
    reading=False
    skip_names = ["E", "PHI", "Z", "R"]
    radius_array = []
    height_array = []
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
                        radius_array += [float(header[1])]
                        height_array += [float(header[2])]
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
    features = pd.DataFrame({"radius": radius_array, "height": height_array})
    num_detectors = len(features)
    core_flux_headers = ["bin" + str(n) for n in range(1, 1 + len(core_flux))]
    features = pd.concat([features, pd.DataFrame([core_flux]*num_detectors,
                          columns=core_flux_headers)], axis=1)
    targets = pd.DataFrame(pebble_flux_matrix, columns=energy_bins_lower)
    avg_uncertainty = np.mean(np.array(unc_array))
    return features, targets, avg_uncertainty