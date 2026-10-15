import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, clone_model
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from pearlsim.ml_utilities import *

FLUX_GROUP_LABELS = [ '2.50000E-09', '7.50000E-09', '1.25000E-08', '1.75000E-08', 
'2.25000E-08', '2.75000E-08', '3.25000E-08', '3.85000E-08', '4.60000E-08', 
'5.40000E-08', '6.25000E-08', '7.35000E-08', '9.00000E-08', '1.20000E-07', 
'1.60000E-07', '2.00000E-07', '2.35000E-07', '2.65000E-07', '2.90000E-07', 
'3.10000E-07', '3.35000E-07', '3.75000E-07', '4.50000E-07', '5.62500E-07', 
'7.02500E-07', '8.15000E-07', '8.80000E-07', '9.30000E-07', '9.61000E-07', 
'9.84000E-07', '1.00800E-06', '1.03250E-06', '1.05800E-06', '1.08400E-06', 
'1.11000E-06', '1.13650E-06', '1.22500E-06', '1.40000E-06', '1.67750E-06', 
'1.97750E-06', '2.35000E-06', '2.95000E-06', '3.65000E-06', '6.93850E-06', 
'1.29225E-05', '2.18340E-05', '3.78760E-05', '6.17767E-05', '1.12115E-04', 
'2.57995E-04', '6.37080E-04', '1.16600E-03', '1.83227E-03', '2.87927E-03', 
'4.50955E-03', '7.30900E-03', '1.20740E-02', '1.99050E-02', '3.28150E-02', 
'5.41400E-02', '8.92150E-02', '1.47000E-01', '2.42750E-01', '4.01250E-01', 
'6.60500E-01', '1.08700E+00', '1.79200E+00', '2.95500E+00', '4.87225E+00', 
'5.00000E+36']

class SingleModelWrapper():
    def __init__(self, model_object, feature_labels, data_mean, data_std, target_mean, target_std, 
                 target_label="target", log_features = []):
        self.model = model_object # Object for model
        self.input_column_labels = feature_labels # List of strings corresponding to features to use
        self.log_scale_labels = log_features # List of strings corresponding to log features to use
        self.data_mean = data_mean
        self.data_std = data_std
        self.target_mean = target_mean.iloc[0]
        self.target_std = target_std.iloc[0]
        self.target_label = target_label

    def log_standardize_data(self, data_raw, axis=0):
        data_log = log_scale_data(data_raw, self.log_scale_labels)
        # Align data to model's expected order if not
        data_log = data_log[self.input_column_labels]
        data_std_log = (data_log - self.data_mean[self.input_column_labels]) / self.data_std[self.input_column_labels]
        return data_std_log

    def unstandardize_target(self, target_std_log):
        target_raw = 10**((target_std_log*self.target_std)+self.target_mean)
        return target_raw

    def predict(self, features_raw, debug=0):
        results = {}
        results_std = {}
        features_std_log = self.log_standardize_data(features_raw).reset_index(drop=True)
        target_std_log = self.model.predict(features_std_log)
        target = self.unstandardize_target(target_std_log)

        return pd.Series(target, name=self.target_label)


class XSModelWrapper():
    def __init__(self, model_object_dict, energy_group_labels, reaction_keys, 
                 data_mean, data_std, target_mean_dict, target_std_dict):
        self.model_dict = model_object_dict # Object for model
        self.input_column_labels = energy_group_labels # List of strings corresponding to features to use
        self.reaction_keys = reaction_keys
        self.data_mean = data_mean
        self.data_std = data_std
        self.target_mean_dict = target_mean_dict
        self.target_std_dict = target_std_dict

    def log_standardize_data(self, data_raw, axis=0):
        data_log = log_scale_data(data_raw, "all")
        # Align data to model's expected order if not
        data_log = data_log[self.input_column_labels]
        data_std_log = (data_log - self.data_mean[self.input_column_labels]) / self.data_std[self.input_column_labels]
        return data_std_log

    def unstandardize_xs(self, xs_std_log, reaction_key):
        xs_raw = 10**((xs_std_log*self.target_std_dict[reaction_key].iloc[0])+self.target_mean_dict[reaction_key].iloc[0])
        return xs_raw

    def predict(self, features_raw, reaction_keys_to_predict="all", debug=0):
        xs_predictions = {}

        if reaction_keys_to_predict == "all":
            reaction_keys_to_predict = self.reaction_keys
        elif type(reaction_keys_to_predict) == str:
            reaction_keys_to_predict = [reaction_keys_to_predict]
        
        features_std_log = self.log_standardize_data(features_raw).reset_index(drop=True)
        for reaction_key in reaction_keys_to_predict:
            xs_std_log = self.model_dict[reaction_key].predict(features_std_log)
            xs_predictions[reaction_key] = self.unstandardize_xs(xs_std_log, reaction_key)
        return pd.DataFrame(xs_predictions)