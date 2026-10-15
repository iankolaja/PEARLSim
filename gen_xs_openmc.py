import openmc
import openmc.deplete
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython
import os
import time
import re
from sklearn.metrics import r2_score
import json
from io import StringIO
import gzip
plt.rcParams.update({'font.size': 12})
import pickle
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_validate
from itertools import product
import tensorflow as tf
from scipy.interpolate import PchipInterpolator

model_dir = "ml_models/xs"
openmc.deplete.chain.REACTIONS

features = pd.read_csv("training_data/xs_features.csv", index_col=0)
flux_targets = pd.read_csv("training_data/flux_targets.csv", index_col=0)
xs_targets = pd.read_csv("training_data/xs_targets.csv", index_col=0)

energy_bins = np.array([0.00000E+00, 5.00000E-09, 1.00000E-08, 1.50000E-08, 2.00000E-08, 2.50000E-08, 3.00000E-08, 3.50000E-08, 4.20000E-08, 5.00000E-08, 
5.80000E-08, 6.70000E-08, 8.00000E-08, 1.00000E-07, 1.40000E-07, 1.80000E-07, 2.20000E-07, 2.50000E-07, 2.80000E-07, 3.00000E-07, 3.20000E-07, 3.50000E-07, 
4.00000E-07, 5.00000E-07, 6.25000E-07, 7.80000E-07, 8.50000E-07, 9.10000E-07, 9.50000E-07, 9.72000E-07, 9.96000E-07, 1.02000E-06, 1.04500E-06, 1.07100E-06, 
1.09700E-06, 1.12300E-06, 1.15000E-06, 1.30000E-06, 1.50000E-06, 1.85500E-06, 2.10000E-06, 2.60000E-06, 3.30000E-06, 4.00000E-06, 9.87700E-06, 1.59680E-05, 
2.77000E-05, 4.80520E-05, 7.55014E-05, 1.48728E-04, 3.67262E-04, 9.06898E-04, 1.42510E-03, 2.23945E-03, 3.51910E-03, 5.50000E-03, 9.11800E-03, 1.50300E-02, 
2.47800E-02, 4.08500E-02, 6.74300E-02, 1.11000E-01, 1.83000E-01, 3.02500E-01, 5.00000E-01, 8.21000E-01, 1.35300E+00, 2.23100E+00, 3.67900E+00, 6.06550E+00, 
1.00000E+37])
energy_differences = np.diff(energy_bins)
energy_midpoints = (energy_bins[1:] + energy_bins[:-1]) / 2
fine_grid = np.logspace(-11,2,1000)


openmc.config['cross_sections'] = "/global/scratch/users/ikolaja/openmc_libs/endfb-vii.1-hdf5/cross_sections.xml"

kernel_volume = 0.36263376
flux_energies = list(flux_targets.columns)
flux_energies.pop(-1)
plot_flux = list(flux_targets.iloc[0])
plot_flux.pop(-1)
plot_flux = plot_flux/energy_differences/kernel_volume
plt.figure()
plt.stairs(plot_flux, energy_bins)
flux_interpolator = PchipInterpolator(np.log10(energy_midpoints), np.log10(plot_flux))
interpolated_flux = 10**flux_interpolator(np.log10(fine_grid))
interpolated_flux[fine_grid > energy_bins[-2]] = 0

plt.plot(fine_grid, interpolated_flux)
plt.yscale("log")
plt.xscale("log")
plt.xlim(1e-12,20)
plt.ylim(1e10,1e21)
plt.xlabel("Neutron Energy (MeV)")
plt.ylabel("Energy Normalized Fuel Kernel Flux")
plt.show()

chain_file = "/global/home/users/ikolaja/openmc_data/chain_endfb71_pwr.xml"
openmc.deplete.MicroXS.from_multigroup_flux(energy_bins, plot_flux, chain_file=chain_file, temperature=959, nuclides=["U235"])
