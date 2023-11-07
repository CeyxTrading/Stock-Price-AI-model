import os
import pandas as pd
from utils.log_utils import *
from utils.training_utils import *
from utils.file_utils import *


class Trainer2():
    """
    Trains the AI model by loading the best hyperparameters and building the model using them
    """
    def __init__(self):
        self._name = "Trainer2"
        setup_logger(self._name + '.txt')

    def train_all(self):
        logi("Training models")

        #  Configure model
        config = {
            "epochs": 200,
            "batch_size": 64,
            "patience": 24,
            "learning_rate": 0.001,
            "time_steps": 5,
            "forecast_steps": 1,
            "train_ratio": 0.6,
            "test_ratio": 0.2,
            "forecast_ratio": 0.2
        }

        #  Set the features we want to work with
        best_features = ['OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE',
                         'QUICK RATIO', 'THREE_WHITE_SOLDIERS',
                         'THREE_BLACK_CROWS', 'IDENTICAL_THREE_CROWS',
                         'EVENING_STAR', 'THREE_OUTSIDE_DOWN',
                         'PCT_CHANGE_1D', 'PCT_CHANGE_1M', 'XLRE_VOLUME']

        #  Load the best hyperparams
        hyperparam_dict = load_hyperparameters_file()

        #  Get list of files
        all_training_results_df = pd.DataFrame()
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

            #  Train one model
            hyperparams = hyperparam_dict[symbol]

            #  Perform training with best hyperparams
            history = perform_training_with_hyperparams(symbol, df, config, hyperparams, plot_predictions=True)

            #  Store results
            training_results_df = pd.DataFrame({'loss': [history.history['loss'][-1]],
                                      'val_loss': [history.history['val_loss'][-1]],
                                      'mean_absolute_error': [history.history['mean_absolute_error'][-1]],
                                      'val_mean_absolute_error': [history.history['val_mean_absolute_error'][-1]]})
            all_training_results_df = pd.concat([all_training_results_df, training_results_df], axis=0, ignore_index=True)

            #  todo
            break

        #  Print config
        print_config_values(config)

        #  Print stats for training for all models
        print_training_stats(all_training_results_df)

        logi('Done training models')

trainer2 = Trainer2()
trainer2.train_all()