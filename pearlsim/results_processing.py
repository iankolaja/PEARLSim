import numpy as np
import pandas as pd
def read_core_flux(file_name):
    reading=False
    core_flux = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.replace("DET","")
            if "map" in line:
                reading = True
                data_array = []
                unc_array = []
            elif "]" in line and reading:
                core_flux = data_array
                avg_uncertainty = np.mean(np.array(unc_array))
                break
            elif reading:
                data = float(line.split()[10])
                unc = float(line.split()[11])
                data_array += [data]
                unc_array += [unc]
    core_flux_headers = ["bin" + str(n) for n in range(1, 1 + len(core_flux))]
    core_flux_map = pd.DataFrame(core_flux, columns=core_flux_headers)
    return core_flux_map, avg_uncertainty