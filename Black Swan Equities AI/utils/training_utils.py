from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from utils.log_utils import *
import os


def build_general_model(X_train, forecast_steps, learning_rate):
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


def create_xs_ys(df):
    #  Specify feature and target columns
    target_column = 'CLOSE'
    feature_columns = df.columns.tolist()
    feature_columns.remove(target_column)

    #  Create Xs and Yx
    X = df[feature_columns].values
    y = df[target_column].values

    return X, y


def scale_data(X, y):
    # Scale the features and target
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    return X_scaled, y_scaled, scaler_y


def create_datasets(X, y, time_steps=1, forecast_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps - forecast_steps + 1):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps:(i + time_steps + forecast_steps)])

    return np.array(Xs), np.array(ys)


def perform_train_test_split(X, y, time_steps, forecast_steps, train_ratio, test_ratio):
    # Create dataset considering lookback steps and forecast days
    X, y = create_datasets(X, y, time_steps, forecast_steps)

    # Calculate split points
    train_size = int(len(X) * train_ratio)
    test_size = int(len(X) * test_ratio)

    # Split data into training, test and future prediction sets
    X_train, X_test, X_future = X[0:train_size], X[train_size:train_size + test_size], X[train_size + test_size:]
    y_train, y_test, y_future = y[0:train_size], y[train_size:train_size + test_size], y[train_size + test_size:]

    return X_train, X_test, y_train, y_test, X_future, y_future


def train(model, X_train, y_train, X_test, y_test, patience, epochs, batch_size):
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
                        callbacks=[early_stopping],
                        verbose=0)
    return history


def plot_performance_metrics(symbol, history):
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


def plot_test_validation_close_price(symbol, y_test, y_test_predicted, y_validation, y_validation_predicted):
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


def perform_training(symbol, df, config):
    #  Perform training for a single stock

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

    #  Build the model
    model = build_general_model(X_train, config['forecast_steps'], config['learning_rate'])

    # Train the model
    history = train(model,
                     X_train,
                     y_train,
                     X_test,
                     y_test,
                     config['patience'],
                     config['epochs'],
                     config['batch_size'])

    """ 
    #  Visualize performance metrics
    plot_performance_metrics(symbol, history)

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

    #  Visualize close actual versus predicted during training/validation
    plot_test_validation_close_price(symbol, y_train_scaled, y_train_predicted_scaled, y_test_scaled, y_test_predicted_scaled)
    """
    return history


def build_model_from_hyperparams(X_train, forecast_steps, hyperparams):
    # Build the LSTM model
    model = Sequential()

    num_layers = int(hyperparams['num_layers'])
    for i in range(num_layers):
        # Get the number of units and dropout for the layer
        units = int(hyperparams[f'units_{i}'])
        dropout = hyperparams[f'dropout_{i}']
        # Add the layer to the model
        model.add(LSTM(units=units, return_sequences=(i != num_layers - 1), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout))

    # Output Layer
    model.add(Dense(units=forecast_steps, activation='relu'))

    #  Initialize Adam optimizer
    optimizer = Adam(learning_rate=hyperparams['learning_rate'])

    # Compile the model
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return model


def perform_training_with_hyperparams(symbol, df, config, hyperparams, plot_predictions):
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

    #  Build the model from hyperparams
    model = build_model_from_hyperparams(X_train, config['forecast_steps'], hyperparams)

    # Assume you have a model called "model"
    print(model.summary())

    # Train the model
    history = train(model,
                     X_train,
                     y_train,
                     X_test,
                     y_test,
                     config['patience'],
                     config['epochs'],
                     config['batch_size'])

    #  Save the model
    file_name = f"{symbol}_model.h5"
    save_model_file(file_name, model)

    if plot_predictions:
        #  Visualize performance metrics
        plot_performance_metrics(symbol, history)

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

        #  Visualize close actual versus predicted during training/validation
        plot_test_validation_close_price(symbol, y_train_scaled, y_train_predicted_scaled, y_test_scaled, y_test_predicted_scaled)

    return history


def perform_training_and_predictions(symbol, df, config, hyperparams):
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

    #  Build the model from hyperparams
    model = build_model_from_hyperparams(X_train, config['forecast_steps'], hyperparams)

    # Train the model
    history = train(model,
                     X_train,
                     y_train,
                     X_test,
                     y_test,
                     config['patience'],
                     config['epochs'],
                     config['batch_size'])

    #  Save the model
    file_name = f"{symbol}_model.h5"
    save_model_file(file_name, model)

    #  Generate predictions for future data
    y_future_predicted = model.predict(X_future)

    #  Reshape the future data and predictions
    y_future_flattened = y_future.reshape(-1, 1)
    y_future_predicted_flattened = y_future_predicted.reshape(-1, 1)

    #  Inverse scale the data
    y_future_scaled = scaler_y.inverse_transform(y_future_flattened)
    y_future_predicted_scaled = scaler_y.inverse_transform(y_future_predicted_flattened)

    #  Create dataframe
    future_predictions_df = pd.DataFrame()
    future_predictions_df['CLOSE'] = y_future_scaled.flatten()
    future_predictions_df['PREDICTED_CLOSE'] = y_future_predicted_scaled.flatten()

    return history, future_predictions_df


def print_training_stats(training_results_df):
    logi("\nTraining stats:\n===============================")
    logi(f"Average Training Loss: {training_results_df['loss'].mean()}")
    logi(f"Average Validation Loss: {training_results_df['val_loss'].mean()}")
    logi(f"Average Training MAE: {training_results_df['mean_absolute_error'].mean()}")
    logi(f"Average Validation MAE: {training_results_df['val_mean_absolute_error'].mean()}")


def print_config_values(config):
    logi("\nConfig:\n===============================")
    for key in config:
        logi(f"{key}: {config[key]}")


def print_hyperparam_stats(all_hyperparam_results_df, max_layers):
    logi("\nHyper param stats:\n===============================")
    logi(f"Average number of layers: {round(all_hyperparam_results_df['num_layers'].mean(),2)}")
    logi(f"Average learning rate: {round(all_hyperparam_results_df['learning_rate'].mean(),2)}")

    for i in range(0, max_layers):
        units_column = f"units_{i}"
        if units_column in all_hyperparam_results_df.columns:
            logi(f"Average units {i}: {round(all_hyperparam_results_df[units_column].mean(), 2)}")
        dropout_column = f"dropout_{i}"
        if dropout_column in all_hyperparam_results_df.columns:
            logi(f"Average dropout rate {i}: {round(all_hyperparam_results_df[dropout_column].mean(), 2)}")


def save_model_file(file_name, model):
    try:
        if not os.path.exists(RESULTS_MODELS_DIR):
            os.makedirs(RESULTS_MODELS_DIR)
        path = os.path.join(RESULTS_MODELS_DIR, file_name)
        model.save(path)
    except Exception as ex:
        loge(f"Failed to save model:")
        loge(ex)


def load_model_file(file_name):
    try:
        path = os.path.join(RESULTS_MODELS_DIR, file_name)
        if os.path.exists(path):
            model = load_model(path)
            logi(f"Loaded model for {file_name}")
            return model
    except Exception as ex:
        loge(f"Failed to load model: {file_name}")
        loge(ex)

