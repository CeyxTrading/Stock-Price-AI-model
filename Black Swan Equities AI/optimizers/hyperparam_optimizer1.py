import os
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, RandomSearch
from utils.file_utils import *
from utils.log_utils import *
from utils.training_utils import *


class StockHyperParamModel(HyperModel):
    def __init__(self, input_shape, config):
        self.input_shape = input_shape
        self.config = config

    def build(self, hp):
        model = keras.models.Sequential()

        # Define number of layers
        num_layers = hp.Int('num_layers', self.config['min_layers'], self.config['max_layers'])
        for i in range(num_layers):
            #  Add LSTM layer
            num_units = hp.Int('units_' + str(i), min_value=self.config['min_units'],
                                  max_value=self.config['max_units'], step=self.config['units_step'])
            model.add(layers.LSTM(units=num_units,
                                  return_sequences=True if i < num_layers - 1 else False,
                                  input_shape=self.input_shape))

            #  Add Dropout layers
            dropout_rate = hp.Float('dropout_' + str(i),
                                     min_value=self.config['dropout_rate_min'],
                                     max_value=self.config['dropout_rate_max'],
                                     step=self.config['dropout_rate_step'],
                                     default=self.config['dropout_rate_default'])
            model.add(layers.Dropout(rate=dropout_rate))

        # Add Output Layer
        num_units_output = self.config['forecast_steps']
        model.add(layers.Dense(units=num_units_output, activation='relu'))

        #  Compile the model with Adam optimizer
        learning_rate = self.config['learning_rate_values']
        model.compile(optimizer=Adam(hp.Choice('learning_rate', values=learning_rate)), loss='mean_squared_error',
                      metrics=[keras.metrics.MeanAbsoluteError()])

        return model


class HyperparamOptimizer1():
    def __init__(self):
        self._name = "HyperparamOptimizer1"
        setup_logger(self._name + '.txt')

    def train(self, model, X_train, y_train, X_test, y_test, patience, epochs, batch_size):
        #  Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)

        history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=epochs,
                            batch_size=batch_size, callbacks=[early_stopping])

        return history

    def perform_hyperparam_tuning(self, X_train, y_train, X_test, y_test, config):
        hypermodel = StockHyperParamModel(input_shape=(X_train.shape[1], X_train.shape[2]), config=config)

        tuner = RandomSearch(hypermodel, objective='val_loss', max_trials=config['max_trials'],
                             executions_per_trial=config['executions_per_trial'],
                             directory='random_search', project_name='LSTM_hyper')

        tuner.search(X_train, y_train, epochs=config['epochs'], validation_data=(X_test, y_test),
                     batch_size=config['batch_size'], callbacks=[EarlyStopping('val_loss', patience=config['patience'])])

        # Get the optimal hyperparameters
        best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]

        #  Build the model with the best hyperparams
        model = tuner.hypermodel.build(best_hyperparams)

        return model, best_hyperparams

    def optimize_model(self, symbol, df, config):
        #  Remove the 'random_search' caching dir from a previous run
        path = os.path.join(ROOT_DIR, 'random_search')
        delete_dir(path)

        #  Create Xs and Yx
        X, y = create_xs_ys(df)

        #  Scale data
        X_scaled, y_scaled, scaler_y = scale_data(X, y)

        #  Perform train-test split
        X_train, X_test, y_train, y_test, X_future, y_future = perform_train_test_split(X_scaled,
                                                                                        y_scaled,
                                                                                        config['time_steps'],
                                                                                        config['forecast_steps'],
                                                                                        config['train_ratio'],
                                                                                        config['test_ratio'])
        #  Perform hyperparam tuning
        model, best_hyperparams = self.perform_hyperparam_tuning(X_train, y_train, X_test, y_test, config)
        logi(best_hyperparams.values)
        best_hyperparam_values = best_hyperparams.values
        best_hyperparam_values['symbol'] = symbol

        #  Create dataframe form hyperparam results
        hyperparam_results_df = pd.DataFrame([best_hyperparam_values])

        #  Perform training with the best hyperparams
        history = train(model, X_train, y_train, X_test, y_test,
                        config['patience'], config['epochs'], config['batch_size'])

        #  Create dataframe with training results
        training_results_df = pd.DataFrame(
            {'symbol': symbol, 'loss': [history.history['loss'][-1]], 'val_loss': [history.history['val_loss'][-1]],
             'mean_absolute_error': [history.history['mean_absolute_error'][-1]],
             'val_mean_absolute_error': [history.history['val_mean_absolute_error'][-1]]})

        #  Perform memory cleanup
        keras.backend.clear_session()

        return hyperparam_results_df, training_results_df

    def optimize_all(self):
        logi("Hyper param optimization")

        #  Configure model
        config = {
            "epochs": 200,  # epochs should be chosen carefully
            "batch_size": 64,  # batch_size is used in tuner.search, not in model.fit
            "patience": 24,  # patience should be chosen carefully
            "learning_rate": 0.001,
            "time_steps": 5,
            "forecast_steps": 1,
            "train_ratio": 0.6,
            "test_ratio": 0.2,
            "forecast_ratio": 0.2,
            "max_trials": 40,
            "executions_per_trial": 2,
            "min_layers": 2,
            "max_layers": 6,
            "layers_step": 1,
            "min_units": 32,
            "max_units": 256,
            "units_step": 32,
            "dropout_rate_min": 0.0,
            "dropout_rate_max": 0.5,
            "dropout_rate_step": 0.05,
            "dropout_rate_default": 0.25,
            "learning_rate_values": [1e-2, 1e-3, 1e-4]
        }

        #  Set the features we want to work with
        best_features = ['OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE',
                         'QUICK RATIO', 'THREE_WHITE_SOLDIERS',
                         'THREE_BLACK_CROWS', 'IDENTICAL_THREE_CROWS',
                         'EVENING_STAR', 'THREE_OUTSIDE_DOWN',
                         'PCT_CHANGE_1D', 'PCT_CHANGE_1M', 'XLRE_VOLUME']

        #  Get list of files
        all_training_results_df = pd.DataFrame()
        all_hyperparam_results_df = pd.DataFrame()
        file_list = get_file_list(STOCK_DATA_DIR)
        for file_name in file_list:
            if "combined" not in file_name:
                continue
            logi(f"Processing {file_name}")

            #  Get symbol from file name
            symbol = file_name.split("_")[0]

            #  Load the file
            df = load_combined_data_file(file_name)

            #  Make sure we have some data for training
            if df is None or len(df) < 260:
                logi(f"No data for symbol {symbol}, file: {file_name}")
                return

            #  Select best features
            df = df[best_features]

            #  Perform optimization for a single stock model
            hyperparam_results_df, training_results_df = self.optimize_model(symbol, df, config)
            all_hyperparam_results_df = pd.concat([all_hyperparam_results_df, hyperparam_results_df], axis=0, ignore_index=True)
            all_training_results_df = pd.concat([all_training_results_df, training_results_df], axis=0, ignore_index=True)

        #  Print config
        print_config_values(config)

        #  Print stats for hyper params for all models
        print_hyperparam_stats(all_hyperparam_results_df, config['max_layers'])

        #  Store all hyper param settings
        file_name = 'all_hyperparam_results_df.csv'
        path = os.path.join(RESULTS_HYPERPARAMS_DIR, file_name)
        all_hyperparam_results_df.to_csv(path)

        #  Print stats for training for all models
        print_training_stats(all_training_results_df)

        logi('Done optimizing models')
