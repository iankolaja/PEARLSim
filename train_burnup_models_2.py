import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import IPython
import os
import time
import re
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import json
from io import StringIO
import gzip
plt.rcParams.update({'font.size': 16})
import pickle
from copy import deepcopy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, cross_validate
from itertools import product
import tensorflow as tf


features = pd.read_csv("training_data/burnup_features.csv", index_col=0)
secondary_targets = pd.read_csv("training_data/burnup_secondary_target.csv", index_col=0)
features = features.join(secondary_targets)
targets = pd.read_csv("training_data/burnup_target.csv", index_col=0)
print("Data loaded")


train_split = 0.8
np.random.seed(42)

def standardize(raw_data, mean=None, std=None, axis=0):
    if mean is None:
        mean = np.mean(raw_data, axis = axis)
    if std is None:
        std = np.std(raw_data, axis = axis)
        std[ std==0 ] = 0.1
    result = (raw_data - mean) / std
    return result, mean, std

def unstandardize(standardized_data, mean, std):
    raw_data = (standardized_data*std)+mean
    return raw_data


num_data = len(features)
training_size = int(num_data*train_split)
testing_size = num_data - training_size
data_indices = np.arange(num_data)
training_indices = np.random.choice(num_data, training_size, replace=False)
testing_indices = data_indices[np.in1d(data_indices, training_indices, invert=True)]

training_data, data_mean, data_std = standardize(features.iloc[training_indices])
training_target, target_mean, target_std = standardize(targets.iloc[training_indices])
testing_data, _, _  = standardize(features.iloc[testing_indices], mean=data_mean, std=data_std)
testing_target, _, _  = standardize(targets.iloc[testing_indices], mean=target_mean, std=target_std)

data_mean.to_csv("ml_models/burnup_nn/data_mean.csv")
data_std.to_csv("ml_models/burnup_nn/data_std.csv")
target_mean.to_csv("ml_models/burnup_nn/target_mean.csv")
target_std.to_csv("ml_models/burnup_nn/target_std.csv")

print("Data split")
print(np.shape(training_data))
print(np.shape(training_target))
print(np.shape(testing_data))
print(np.shape(testing_target))


np.random.seed(2)
tf.random.set_seed(2)

important_features_by_nuclide = {}
rfr_scores = {}
rfr_scores_important = {}
conc_scores = {}
conc_best_models = {}
conc_best_networks_sizes = {}
epoch_counts = [300, 400,  500, 600]
network_sizes = (
    (8,2),
    (16,2),
    (32,4),
    (64,4),
    (64,8),
    (64,16),
    (128,4),
    (128,8),
    (128,16),
    (128,32),
    (128),
    (64),
    (32),
    (16),
    (2)
)

rfr_params = {'max_depth': 20, 
               'n_estimators': 1000,
               'n_jobs': 32} 
forced_features = ['current_E1', 'current_E2','current_E3','current_E4','current_E5','current_E6',
            'current_E7', 'current_E8', 'current_E9', 'current_E10',
            'current_E11', 'current_E12', 'current_E13', 'current_E14',
            'current_E15', 'current_E16','current_E17','current_E18',
            'power', '92235<lib>', '92238<lib>', 'depletion_time']


nuclides_to_model = training_target.columns

result_array = []

for isotope in nuclides_to_model[600:]:
    print(f"\n\n{isotope}")
    file_name = f"ml_models/burnup_nn/{isotope}.pkl".replace("<","_").replace(">","_")
    file_name2 = f"ml_models/burnup_nn/{isotope}_model.keras".replace("<","_").replace(">","_")
    config_file_name = f"ml_models/burnup_nn/{isotope}_results.txt".replace("<","_").replace(">","")
    print(file_name)
    print(config_file_name)
    if os.path.exists(file_name) or os.path.exists(file_name2):
        print("Model for nuclide already exists. Skipping.")
        continue

    if os.path.isfile(config_file_name):
        print("Model has been previously optimized. Retraining...")
        b_already_optimized = True
        with open(config_file_name, 'r') as f:
            optimal_features = json.load(f)
    else:
        b_already_optimized = False
    
    # Determine most important features
    rfr_model = RandomForestRegressor(random_state=0)
    rfr_model.set_params(**rfr_params)
    rfr_model.fit(training_data, training_target[isotope])
    rfr_model_test_score = rfr_model.score(testing_data, testing_target[isotope])
    print(f"{isotope} R^2 score no importance: {rfr_model_test_score}")
    rfr_scores[isotope] = rfr_model_test_score
    feature_labels = training_data.columns
    importances = rfr_model.feature_importances_
    
    nuclides_to_use = sum( importances > np.mean(importances)/4 )
    indices = np.argsort(importances)[-nuclides_to_use:]
    important_features_by_nuclide[isotope] = indices
    top_importance_vals = importances[indices]
    top_features = feature_labels[indices]
    top_features = list(set( list(top_features) + forced_features + [isotope]))

    # Some isotopes globally do not seem exist before the burnup step, likely super short lived
    if isotope not in training_data.columns:
        top_features.remove(isotope)
    
    important_features_by_nuclide[isotope] = top_features
    rfr_model.fit(training_data[top_features], training_target[isotope])
    rfr_model_test_score = rfr_model.score(testing_data[top_features], testing_target[isotope])
    rfr_scores_important[isotope] = rfr_model_test_score
    
    predicted = rfr_model.predict(testing_data[top_features])
    expected = testing_target[isotope].to_numpy()
    mse = np.mean( (expected-predicted)**2 )
    print(f"{isotope} R^2 score: {rfr_model_test_score} MSE {mse}")

    importances = rfr_model.feature_importances_
    indices = np.argsort(importances)
    top_importance_vals = importances[indices]
    sorted_top_features = testing_data[top_features].columns[indices]
    
    print(sorted_top_features)
    print("\n\n")
    
    # Tune network
    network_fits = []
    best_epoch_amounts = []
    models = []


    train_len = training_data[top_features].shape[0]
    test_len = testing_data.shape[0]
    num_outputs = 1
    num_inputs = training_data[top_features].shape[1]
    
    X_train = np.array(training_data[top_features]).reshape(train_len, -1, 1)
    X_test = np.array(testing_data[top_features]).reshape(test_len, -1, 1)
    y_train = np.array(training_target[isotope]).reshape(train_len, -1, 1)
    y_test = np.array(testing_target[isotope]).reshape(test_len, -1, 1)
        
    if not b_already_optimized:
        
        for size in network_sizes:
            print(f"\n{isotope} model - Network Size {size}")
            
    
    
            if type(size) == int:
                nn_model = Sequential()
                nn_model.add(Dense(size, input_dim=num_inputs, activation='relu'))
                nn_model.add(Dense(num_outputs, activation='linear'))
            else:
                nn_model = Sequential()
                nn_model.add(Dense(size[0], input_dim=num_inputs, activation='relu'))
                nn_model.add(Dense(size[1], activation='relu')) 
                nn_model.add(Dense(num_outputs, activation='linear'))  
            
            models_by_epochs = []
            fit_by_epochs = []
            
            for epochs in epoch_counts:
                print(f"{epochs} epochs")
                batch_size = 256
                learning_rate = 0.001
    
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
                nn_model.compile(loss='mse', optimizer=optimizer)
                history = nn_model.fit(X_train, y_train, 
                                        epochs=epochs, batch_size=batch_size, verbose=0, 
                                        validation_data=(X_test, y_test))
    
    
                y_predict = nn_model.predict(X_test)
                models_by_epochs += [nn_model]
                column_predict = y_predict[:,0]
                column_truth = testing_target[isotope]
                r2_score_val = r2_score(column_truth, column_predict)
                fit_by_epochs += [r2_score_val]
                print(r2_score_val)
                print(f"Training loss: {history.history['loss'][-1]}")
                print(f"Validation loss: {history.history['val_loss'][-1]}")
            models += [models_by_epochs[np.argmax(fit_by_epochs)] ]
            network_fits += [np.max(fit_by_epochs)]
            best_epoch_amounts += [epoch_counts[np.argmax(fit_by_epochs)] ]
            print(f"Optimal epoch number determined {epoch_counts[np.argmax(fit_by_epochs)]} with r^2 {np.max(fit_by_epochs)}")
    
        conc_scores[isotope] = max(network_fits)
        conc_best_networks_sizes[isotope] = network_sizes[np.argmax(network_fits)]
        conc_best_models[isotope] = deepcopy(models[np.argmax(network_fits)])
        for model in models:
            del model
        print("\n\n")
        file_name = f"ml_models/burnup_nn/{isotope}_model.keras".replace("<","_").replace(">","_")
        conc_best_models[isotope].save(file_name)
        file_name = f"ml_models/burnup_nn/{isotope}_results.txt".replace("<","_").replace(">","")
        results = {}
        results["best_network_size"] = network_sizes[np.argmax(network_fits)]
        results["best_epochs"] = best_epoch_amounts[np.argmax(network_fits)]
        results["r2_score"] = max(network_fits)
        results["important_features"] = top_features
        
        
        with open(file_name, 'w') as f:
            json.dump(results, f)
    
    else:
        size = optimal_features['best_network_size']
        epochs = optimal_features["best_epochs"]


        if type(size) == int:
            nn_model = Sequential()
            nn_model.add(Dense(size, input_dim=num_inputs, activation='relu'))
            nn_model.add(Dense(num_outputs, activation='linear'))
        else:
            nn_model = Sequential()
            nn_model.add(Dense(size[0], input_dim=num_inputs, activation='relu'))
            nn_model.add(Dense(size[1], activation='relu')) 
            nn_model.add(Dense(num_outputs, activation='linear'))  
            
        
        
    
        batch_size = 512
        learning_rate = 0.001

        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        nn_model.compile(loss='mse', optimizer=optimizer)
        history = nn_model.fit(X_train, y_train, 
                                epochs=epochs, batch_size=batch_size, verbose=0, 
                                validation_data=(X_test, y_test))


        y_predict = nn_model.predict(X_test)
        column_predict = y_predict[:,0]
        column_truth = testing_target[isotope]
        r2_score_val = r2_score(column_truth, column_predict)
        mape_score_val = mean_absolute_percentage_error(column_truth, column_predict)*100
        print(f"score: R^2={r2_score_val}   MAPE={mape_score_val}")
        print(f"Training loss: {history.history['loss'][-1]}")
        print(f"Validation loss: {history.history['val_loss'][-1]}")
    

        file_name = f"ml_models/burnup_nn/{isotope}_model.keras".replace("<","_").replace(">","_")
        nn_model.save(file_name)
        
        file_name = f"ml_models/burnup_nn/{isotope}_results.txt".replace("<","_").replace(">","_")
        results = {}
        results["best_network_size"] = size
        results["best_epochs"] = epochs
        results["nn_r2_score"] = r2_score_val
        results["rfr_r2_score"] = rfr_model_test_score
        results["mape_score"] = mape_score_val
        results["important_features"] = top_features
        file_name = f"ml_models/burnup_nn/{isotope}_results.txt".replace("<","_").replace(">","")
        with open(file_name, 'w') as f:
            json.dump(results, f)
