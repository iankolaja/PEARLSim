from pearlsim.serpent_spectrum_tools import *
import argparse
import pandas as pd
import numpy as np
import os
import math
import subprocess
from time import time
import matplotlib.pyplot as plt # not needed if you don't want to plot
import pickle
import warnings
import sys
import gzip
from scipy.interpolate import interp2d
from pearlsim.serpent_spectrum_tools import *
from pearlsim.results_processing import *

# constants
target_list = ['avg_r_dist', 'passes', 'residence_time',
       'burnup', 'integrated_flux_pebbles_thermal',
       'integrated_flux_pebbles_fast', 'pass_residence_time',
       'pass_avg_r_dist','pass_burnup',
       'pass_integrated_flux_pebbles_thermal',
       'pass_integrated_flux_pebbles_fast']

target_list = ['avg_r_dist','passes','residence_time',
               'burnup','fima','pass_residence_time',
               'pass_avg_r_dist','pass_fima','pass_burnup',
              'integrated_flux_pebbles_thermal',
              'integrated_flux_pebbles_fast',
              'integrated_power_pebbles', 
              'pass_integrated_flux_pebbles_thermal',
              'pass_integrated_flux_pebbles_fast',
              'pass_integrated_power_pebbles', 
              ]

unc_list = ['integrated_flux_pebbles_thermal_rel_unc',
            'integrated_flux_pebbles_fast_rel_unc',
            'integrated_power_pebbles_rel_unc', 
            'pass_integrated_flux_pebbles_thermal_rel_unc',
            'pass_integrated_flux_pebbles_fast_rel_unc',
            'pass_integrated_power_pebbles_rel_unc']

# argument handling
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--isotopes')
parser.add_argument('-h', '--histories')
parser.add_argument('-d', '--datadir')
parser.add_argument('-w', '--workingdir')
parser.add_argument('-c', '--cores')
args = parser.parse_args()

working_directory = "pebble_decay"
data_directory = ""
isotopic_path = ""
history_path = ""
triso_path = "../gFHR_triso.dat"
cores = 20

args = parser.parse_args()
 
if args.isotopes:
    print("Reading concentration data from file: ", args.isotopes)
    isotopic_path = args.isotopes
if args.histories:
    print("Reading pebble history data from file: ", args.histories)
    history_path = args.histories
if args.workingdir:
    print("Working in directory", args.workingdir)
    working_directory = args.workingdir
if args.cores:
    print("Setting number of cores", args.cores)
    simulation.cpu_cores = int(args.cores)

    
np.random.seed(42)


isotopic_data = pd.read_csv(data_directory+isotopic_path, index_col=0).reset_index(drop=True)

history_data = pd.read_csv(data_directory+history_path, index_col=0).reset_index(drop=True)

unc_data = history_data[unc_list]
history_data = history_data[target_list]

num_data = len(history_data)
sample_size = int(num_data*fraction_of_pebbles_to_use)
sample_indices = np.random.choice(num_data, sample_size, replace=False)

isotopic_data = isotopic_data.iloc[sample_indices].reset_index(drop=True)
history_data = history_data.iloc[sample_indices].reset_index(drop=True)

# Remove isotope columns that are globally 0.
isotopic_data = isotopic_data.loc[:, (isotopic_data > 0).any(axis=0)]

# Ensure burnups match up correctly.
#assert np.all(tol > np.abs(isotopic_data['burnup'] - history_data['burnup']))

# Convert carbon to not natural
isotopic_data['60120'] = isotopic_data['60120'] + isotopic_data['60000']*0.9889
isotopic_data['60130'] = isotopic_data['60130'] + isotopic_data['60000']*(1-0.9889)

isotopic_data = isotopic_data.drop(columns=['-1', '60000'])

# Convert to total nuclide number
isotopic_data = isotopic_data*1e-24

real_isotopic_dataset[step] = isotopic_data
history_dataset[step] = history_data
unc_dataset[step] = unc_data

