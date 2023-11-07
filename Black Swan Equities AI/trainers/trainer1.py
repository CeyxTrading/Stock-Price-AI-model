import os
import pandas as pd
from utils.file_utils import *
from utils.log_utils import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


class Trainer1():
    """
    Trains the AI model
    """
    def __init__(self):
        self._name = "Trainer1"

    def load_data_file(self, file_name):
        #  Read the file
        path = os.path.join(STOCK_DATA_DIR, file_name)
        df = read_csv_file(path)
        if df is None:
            return None
        logi(f"Data columns: {len(df.columns)}, rows: {len(df)}")

        # Set Date as index
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)

        #  Perform additional cleaning -> todo: move to data loader
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
        return df[indices_to_keep].astype(np.float64)

    def build_model(self, X_train, forecast_steps, learning_rate):
        # Build the LSTM model
        model = Sequential()

        # Layer 1
        model.add(LSTM(units=65, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))

        # Layer 2
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dropout(0.2))

        # Output Layer
        model.add(Dense(units=forecast_steps, activation='relu'))

        #  Initialize Adam optimizer
        optimizer = Adam(learning_rate=learning_rate)

        # Compile the model
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
        return model

    def create_Xs_ys(self, df):
        #  Specify feature and target columns
        target_column = 'CLOSE'
        feature_columns = df.columns.tolist()
        feature_columns.remove(target_column)

        #  Create Xs and Yx
        X = df[feature_columns].values
        y = df[target_column].values

        return X, y

    def scale_data(self, X, y):
        # Scale the features and target
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)

        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
        return X_scaled, y_scaled, scaler_y

    def create_datasets(self, X, y, time_steps=1, forecast_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps - forecast_steps + 1):
            Xs.append(X[i:(i + time_steps)])
            ys.append(y[i + time_steps:(i + time_steps + forecast_steps)])

        return np.array(Xs), np.array(ys)

    def perform_train_test_split(self, X, y, time_steps, forecast_steps, train_ratio, test_ratio):
        # Create dataset considering lookback steps and forecast days
        X, y = self.create_datasets(X, y, time_steps, forecast_steps)

        # Calculate split points
        train_size = int(len(X) * train_ratio)
        test_size = int(len(X) * test_ratio)

        # Split data into training, test and future prediction sets
        X_train, X_test, X_future = X[0:train_size], X[train_size:train_size + test_size], X[train_size + test_size:]
        y_train, y_test, y_future = y[0:train_size], y[train_size:train_size + test_size], y[train_size + test_size:]

        return X_train, X_test, y_train, y_test, X_future, y_future

    def train(self, model, X_train, y_train, X_test, y_test, patience, epochs, batch_size):
        #  Define early stopping
        early_stopping = EarlyStopping(monitor='val_loss',
                                       patience=patience,
                                       mode='min',
                                       verbose=1)

        history = model.fit(x=X_train,
                            y=y_train,
                            validation_data=(X_test, y_test),
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[early_stopping])
        return history

    def plot_performance_metrics(self, symbol, history):
        # Plot training & validation loss values
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')

        #  Plot mean_absolute_error
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('Model Mean Absolute Error')
        plt.ylabel('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.legend(['Train MAE', 'Test MAE'], loc='upper right')

        plt.tight_layout()

        # Save the plot
        file_name = f"{symbol}_metrics_plot.png"
        path = os.path.join(RESULTS_METRICS_DIR, file_name)
        plt.savefig(path)

    def plot_test_validation_close_price(self, symbol, y_test, y_test_predicted, y_validation, y_validation_predicted):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.plot(y_test[-200:], color='blue', label='Real close price')
        plt.plot(y_test_predicted[-200:], color='red', label='Predicted close price')
        plt.title('Close Price Prediction on Training Set')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(y_validation[:200], color='blue', label='Real close price')
        plt.plot(y_validation_predicted[:200], color='red', label='Predicted close price')
        plt.title('Close Price Prediction on Test Set')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.legend()

        plt.tight_layout()

        # Save the plot
        file_name = f"{symbol}_training_validation_plot.png"
        path = os.path.join(RESULTS_TRAINING_PLOTS_DIR, file_name)
        plt.savefig(path)

    def perform_training(self, symbol, df, config):
        #  Perform training for a single stock

        #  Create Xs and Yx
        X, y = self.create_Xs_ys(df)

        #  Scale data
        X_scaled, y_scaled, scaler_y = self.scale_data(X, y)

        #  Perform train-test split
        X_train, X_test, y_train, y_test, X_future, y_future = self.perform_train_test_split(X_scaled,
                                                                                             y_scaled,
                                                                                             config['time_steps'],
                                                                                             config['forecast_steps'],
                                                                                             config['train_ratio'],
                                                                                             config['test_ratio'])

        #  Build the model
        model = self.build_model(X_train, config['forecast_steps'], config['learning_rate'])

        # Train the model
        history = self.train(model,
                             X_train,
                             y_train,
                             X_test,
                             y_test,
                             config['patience'],
                             config['epochs'],
                             config['batch_size'])

        #  Visualize performance metrics
        self.plot_performance_metrics(symbol, history)

        #  Perform predictions for the training and test data
        y_train_predicted = model.predict(X_train)
        y_test_predicted = model.predict(X_test)

        # Flatten y and predicted y before inverse transformation
        y_train_flattened = y_train.reshape(-1, 1)
        y_test_flattened = y_test.reshape(-1, 1)
        y_train_predicted_flattened = y_train_predicted.reshape(-1, 1)
        y_test_predicted_flattened = y_test_predicted.reshape(-1, 1)

        #  Inverse-transform the predictions before plotting
        y_train_scaled = scaler_y.inverse_transform(y_train_flattened)
        y_test_scaled = scaler_y.inverse_transform(y_test_flattened)
        y_train_predicted_scaled = scaler_y.inverse_transform(y_train_predicted_flattened)
        y_test_predicted_scaled = scaler_y.inverse_transform(y_test_predicted_flattened)

        self.plot_test_validation_close_price(symbol, y_train_scaled, y_train_predicted_scaled, y_test_scaled, y_test_predicted_scaled)

        return history

    def print_training_stats(self, config, training_results_df):
        logi("Training stats")
        for key in config:
            logi(f"{key}: {config[key]}")

        logi(f"Average Training Loss: {training_results_df['loss'].mean()}")
        logi(f"Average Validation Loss: {training_results_df['val_loss'].mean()}")
        logi(f"Average Training MAE: {training_results_df['mean_absolute_error'].mean()}")
        logi(f"Average Validation MAE: {training_results_df['val_mean_absolute_error'].mean()}")

    def train_all(self):
        logi("Training models")

        #  Configure model
        config = {
            "epochs": 200,
            "batch_size": 64,
            "patience": 24,
            "learning_rate": 0.001,
            "time_steps": 20,
            "forecast_steps": 5,
            "train_ratio": 0.6,
            "test_ratio": 0.2,
            "forecast_ratio": 0.2
        }

        #  Get list of files
        training_results_df = pd.DataFrame()
        file_list = get_file_list(STOCK_DATA_DIR)
        for file_name in file_list:
            if "combined" not in file_name:
                continue
            logi(f"Processing {file_name}")

            #  Get symbol from file name
            symbol = file_name.split("_")[0]

            #  Load the file
            df = self.load_data_file(file_name)

            #  Make sure we have some data for training
            if df is None or len(df) < 260:
                logi(f"No data for symbol {symbol}, file: {file_name}")
                continue

            #  Train the model
            history = self.perform_training(symbol, df, config)

            #  Store results
            result_df = pd.DataFrame({'loss': [history.history['loss'][-1]],
                                      'val_loss': [history.history['val_loss'][-1]],
                                      'mean_absolute_error': [history.history['mean_absolute_error'][-1]],
                                      'val_mean_absolute_error': [history.history['val_mean_absolute_error'][-1]]})
            training_results_df = pd.concat([training_results_df, result_df], axis=0, ignore_index=True)

        #  Print stats for training for all models
        self.print_training_stats(config, training_results_df)

        logi('Done training models')

trainer1 = Trainer1()
trainer1.train_all()