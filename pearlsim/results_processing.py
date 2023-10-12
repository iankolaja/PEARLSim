import numpy as np
import pandas as pd
from .ml_utilities import ENERGY_BINS, RADIUS_BINS, HEIGHT_BINS

def read_core_flux(file_name, normalize_and_label=False):
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
    if normalize_and_label:
        core_flux_headers = []
        num_radius_bins = len(RADIUS_BINS)-1
        num_height_bins = len(HEIGHT_BINS)-1
        num_energy_bins = len(ENERGY_BINS)-1
        bin_e = 0
        bin_r = -1
        bin_z = 0
        for i in range(len(core_flux)):
            if bin_r == num_radius_bins-1:
                bin_r = 0
                if bin_z == num_height_bins-1:
                    bin_z = 0
                    bin_e += 1
                else:
                    bin_z += 1
            else:
                bin_r += 1
            # Calculate volume of a washer, noting that height bins are in descending
            # order while radius bins are increasing
            volume = np.pi*(RADIUS_BINS[bin_r+1]**2-RADIUS_BINS[bin_r]**2)*(HEIGHT_BINS[bin_z]-HEIGHT_BINS[bin_z+1])
            energy_width = ENERGY_BINS[bin_e+1]-ENERGY_BINS[bin_e+1]
            core_flux.iloc[i] = core_flux.iloc[i]/volume/energy_width
            core_flux_headers += [f"binR{bin_r+1}Z{bin_z+1}E{bin_e+1}"]
    else:
        core_flux_headers = ["bin" + str(n) for n in range(1, 1 + len(core_flux))]
    core_flux = pd.DataFrame([core_flux], columns=core_flux_headers)
    return core_flux, avg_uncertainty


def read_res_file(file_name, parameter, burnup_step=2, sub_index=0):
    with open(file_name, 'r') as f:
        lines = f.readlines()
        matches = 0
        matching_lines = []
        for i in range(len(lines)):
            if parameter in lines[i]:
                matches += 1
                if matches == burnup_step:
                    result = lines[i]
                    break
    # Sample : "ANA_KEFF                  (idx, [1:   6]) = [  1.29640E+00 0.00153  1.28804E+00 0.00150  8.70269E-03 0.02988 ];"
    result = result.split("=")[1]
    # Sample : " [  1.29640E+00 0.00153  1.28804E+00 0.00150  8.70269E-03 0.02988 ];"
    value = float(result.split()[1+sub_index*2])
    unc = float(result.split()[2+sub_index*2])
    return value, unc
        
        
        