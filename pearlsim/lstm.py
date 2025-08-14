from pearlsim.ml_utilities import *
import tensorflow as tf
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense, LSTM, GRU
import keras
import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import r2_score, mean_absolute_error

class LSTM_predictor():
    def __init__(self, feature_mean, feature_std, target_mean, target_std):
        self.models = {}
        self.input_shape = (0)
        self.window = 1
        self.feature_labels = []
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_mean = target_mean
        self.target_std = target_std
        self.models_trained = False
        self.history_dict = {}
        self.r2_scores_train = {}
        self.r2_scores_test = {}
        self.r2_scores_val = {}

    def standardize(self, training_data_matrix, training_target_df=None, 
                    testing_data_matrix=None, testing_target_df=None,
                    validation_data_matrix=None, validation_target_df=None):

        # Standardize features
        training_data_matrix_std, _, _ = standardize(training_data_matrix, 
                                                     mean=self.feature_mean, 
                                                     std=self.feature_std, 
                                                     axis=2)
        if training_target_df is not None:
            testing_data_matrix_std, _, _ = standardize(testing_data_matrix, 
                                                         mean=self.feature_mean, 
                                                         std=self.feature_std, 
                                                         axis=2)
        # Escape early if just features provided
        else:
            return training_data_matrix_std


        
        if validation_target_df is not None:
            validation_data_matrix_std, _, _ = standardize(validation_data_matrix, 
                                                         mean=self.feature_mean, 
                                                         std=self.feature_std, 
                                                         axis=2)


        # Standardize targets
        if testing_data_matrix is not None and testing_target_df is not None and validation_data_matrix is None:
            training_target_df_std, _, _ = standardize(training_target_df, 
                                                       mean=self.target_mean, 
                                                       std=self.target_std,
                                                       axis=0)
            testing_target_df_std, _, _ = standardize(testing_target_df, 
                                                       mean=self.target_mean, 
                                                       std=self.target_std, 
                                                       axis=0)
            return (training_data_matrix_std, 
                    testing_data_matrix_std, 
                    training_target_df_std, 
                    testing_target_df_std)
 
        if testing_data_matrix is not None and testing_target_df is not None and validation_data_matrix is not None and validation_target_df is not None:
            training_target_df_std, _, _ = standardize(training_target_df, 
                                                       mean=self.target_mean, 
                                                       std=self.target_std,
                                                       axis=0)
            testing_target_df_std, _, _ = standardize(testing_target_df, 
                                                       mean=self.target_mean, 
                                                       std=self.target_std, 
                                                       axis=0)
            validation_target_df_std, _, _ = standardize(validation_target_df, 
                                                       mean=self.target_mean, 
                                                       std=self.target_std,
                                                       axis=0)
            return (training_data_matrix_std, 
                    testing_data_matrix_std,  
                    validation_data_matrix_std, 
                    training_target_df_std, 
                    testing_target_df_std, 
                    validation_target_df_std)
        else:
            return (training_data_matrix_std, 
                    testing_data_matrix_std)   
            

    
    def train(self, training_data_matrix, training_target_df, testing_data_matrix, testing_target_df,
              validation_data_matrix, validation_target_df, 
              target_labels, feature_labels, epochs_key={}, epochs_default=500, batch_size=1024, learning_rate=0.001, 
              default_network_size=[64], network_size_key={}, predict_labels = [], 
              input_dropout=0, recurrent_dropout=0, recurrent_regularizer=None):
        self.feature_labels = feature_labels
        X_train, X_test, X_val, Y_train_df, Y_test_df, Y_val_df = self.standardize(training_data_matrix, 
                                                             training_target_df, 
                                                             testing_data_matrix=testing_data_matrix, 
                                                             testing_target_df=testing_target_df, 
                                                             validation_data_matrix=validation_data_matrix, 
                                                             validation_target_df=validation_target_df)
        train_len, dim1, dim2 = training_data_matrix.shape
        test_len, _, _ = testing_data_matrix.shape
        val_len, _, _ = validation_data_matrix.shape
        self.input_shape = (dim1, dim2)
        self.window = dim1

        self.history_dict = {}
        
        for label in target_labels:
            # Define LSTM model
            if label in epochs_key.keys():
                label_epochs = epochs_key[label]
            else:
                label_epochs = epochs_default

            if label in network_size_key.keys():
                network_size = network_size_key[label]
            else:
                network_size = default_network_size
            early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=25,
                                                          start_from_epoch=10,
                                                          restore_best_weights=True)

            

            
            if len(network_size) == 2:
                model = tf.keras.Sequential([
                    tf.keras.Input((dim1, dim2)),
                    tf.keras.layers.LSTM(network_size[0], 
                                         dropout=input_dropout, 
                                         recurrent_dropout=recurrent_dropout,
                                         return_sequences=True,
                                         recurrent_regularizer=recurrent_regularizer),
                    tf.keras.layers.LSTM(network_size[1]),
                    tf.keras.layers.Dense(1, activation="linear")
                ])

            else:
                model = tf.keras.Sequential([
                    tf.keras.Input((dim1, dim2)),
                    tf.keras.layers.LSTM(network_size[0], 
                                         dropout=input_dropout, 
                                         recurrent_dropout=recurrent_dropout,
                                         recurrent_regularizer=recurrent_regularizer),
                    tf.keras.layers.Dense(1, activation="linear")
                ]) 

            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=learning_rate,
                decay_steps=10000,
                decay_rate=0.9)
            optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)#, clipnorm=0.8)
            
            # Compile model
            model.compile(loss='mse', optimizer=optimizer)
            
            # Train model
            y_train = Y_train_df[label].to_numpy().reshape([train_len, 1])
            y_val = Y_val_df[label].to_numpy().reshape([val_len, 1])
            y_test = Y_test_df[label].to_numpy().reshape([test_len, 1])
            
            self.history_dict[label] = model.fit(X_train, y_train, 
                           epochs=label_epochs, 
                           batch_size=batch_size, 
                           verbose=1, 
                           validation_data=(X_val, y_val),
                           callbacks=[early_stopper])
            self.models[label] = model

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            self.r2_scores_train[label] = r2_score(y_train, y_train_pred)
            self.r2_scores_test[label] = r2_score(y_test, y_test_pred)
            self.r2_scores_val[label] = r2_score(y_val, y_val_pred)
        self.models_trained = True
        return self.history_dict

    def forecast(self, training_data_sequence, num_steps_ahead, dependent_input_labels, predict_labels):
        X_train = self.standardize(training_data_sequence)
        X_train = X_train.reshape([1,self.input_shape[0],self.input_shape[1]])
        predicted_inputs_list = []
        predicted_outputs_list = []
        for step in range(num_steps_ahead):
            predicted_outputs_dict = {}
            
            for label in predict_labels:    
                predicted_outputs_dict[label] = self.models[label].predict(X_train)[0][0]
            predicted_outputs_list += [predicted_outputs_dict]
                

            next_values = X_train[0,-1,:]
            next_inputs = dict()
            next_inputs.update(zip(self.feature_labels, next_values))
            for label in dependent_input_labels:
                next_inputs[label] = self.models[label].predict(X_train)[0][0]
            next_inputs = np.array(list(next_inputs.values()))
            next_inputs = next_inputs.reshape([1,1,self.input_shape[1]])
            X_train = X_train[0,1:,:].reshape([1,self.input_shape[0]-1,self.input_shape[1]])
            X_train = np.concatenate([X_train, next_inputs], axis=1)
        return pd.DataFrame(predicted_outputs_list)

    def predict(self, features, predict_labels):

        length, num_features = features[self.feature_labels].shape
        num_data = length-1
        num_samples = num_data - self.window
        feature_list = []
        target_list = []
        if len(features) == self.window:
            input_matrix = features[self.feature_labels].to_numpy().reshape(1,self.window, num_features)
        else:
            for index in np.arange(self.window, num_data):
                feature_list += [features[self.feature_labels].iloc[index-self.window:index].to_numpy().reshape(1, self.window, num_features)]
            input_matrix = np.concatenate(feature_list, axis=0)
        X_train = self.standardize(input_matrix)
        predicted_outputs_dict = {}
        
        for label in predict_labels:    
            predicted_values_std = self.models[label].predict(X_train).T[0]
            predicted_outputs_dict[label] = (predicted_values_std*self.target_std[label])+self.target_mean[label]
        return pd.DataFrame(predicted_outputs_dict)
