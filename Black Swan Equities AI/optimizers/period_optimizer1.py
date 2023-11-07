from utils.file_utils import *
from utils.log_utils import *
from utils.training_utils import *


class PeriodOptimizer1:
    """
        Optimizes the lookback and forecast periods
    """
    def __init__(self):
        self._name = "PeriodOptimizer1"
        setup_logger(self._name + '.txt')

    def perform_training(self, test_features, config):
        #  Performs training of all selected stocks for one set of features
        training_results_df = pd.DataFrame()
        max_items = config['max_items']
        item_counter = 1
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
                continue

            #  Only keep the candidate features we want to test
            temp_df = df[test_features].copy()

            #  Train the model
            history = perform_training(symbol, temp_df, config)

            #  Store results
            result_df = pd.DataFrame({'loss': [history.history['loss'][-1]],
                                      'validation_loss': [history.history['val_loss'][-1]],
                                      'mean_absolute_error': [history.history['mean_absolute_error'][-1]],
                                      'validation_mean_absolute_error': [history.history['val_mean_absolute_error'][-1]]})
            training_results_df = pd.concat([training_results_df, result_df], axis=0, ignore_index=True)

            #  Only process files up to max items
            item_counter += 1
            if item_counter > max_items:
                break
        return training_results_df

    def print_stats(self, config, best_training_loss, best_validation_loss, best_training_mae, best_validation_mae,
                    best_time_step, best_forecast_step):
        logi("\nStats\n===============================")
        for key in config:
            logi(f"{key}: {config[key]}")

        round_digits = 8
        logi(f"Best Training Loss: {round(best_training_loss, round_digits)}")
        logi(f"Best Validation Loss: {round(best_validation_loss, round_digits)}")
        logi(f"Best Training MAE: {round(best_training_mae, round_digits)}")
        logi(f"Best Validation MAE: {round(best_validation_mae, round_digits)}")
        logi(f"Best Time step: {best_time_step}")
        logi(f"Best Forecast step: {best_forecast_step}")

    def optimize(self, max_items):
        logi("Training models")
        
        #  Set Time Periods
        time_step_list = [5, 10, 15, 20, 30, 60]
        forecast_step_list = [1, 2, 3, 4, 5, 10]

        #  Configure model
        config = {
            "epochs": 200,
            "batch_size": 64,
            "patience": 24,
            "learning_rate": 0.001,
            "time_steps": 1,
            "forecast_steps": 5,
            "train_ratio": 0.6,
            "test_ratio": 0.2,
            "forecast_ratio": 0.2,
            "max_items": max_items
        }

        #  Set the features we want to work with
        best_features = ['OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE',
                         'QUICK RATIO', 'THREE_WHITE_SOLDIERS',
                         'THREE_BLACK_CROWS', 'IDENTICAL_THREE_CROWS',
                         'EVENING_STAR', 'THREE_OUTSIDE_DOWN',
                         'PCT_CHANGE_1D', 'PCT_CHANGE_1M', 'XLRE_VOLUME']

        best_training_loss = float('inf')
        best_validation_loss = float('inf')
        best_training_mae = float('inf')
        best_validation_mae = float('inf')
        best_time_step = 1
        best_forecast_step = 5
        #  Loop through forecast periods
        for forecast_step in forecast_step_list:
            logi(f"Testing forecast step {forecast_step}")
            config['forecast_steps'] = forecast_step
            #  Loop through lookback periods
            for time_step in time_step_list:
                logi(f"Testing time step {time_step}")
                if time_step < forecast_step:
                    continue

                config['time_steps'] = time_step

                #  Perform training
                training_results_df = self.perform_training(best_features, config)

                #  Check if adding the feature improved validation MAE
                avg_training_loss = training_results_df['loss'].mean()
                avg_validation_loss = training_results_df['validation_loss'].mean()
                avg_training_mae = training_results_df['mean_absolute_error'].mean()
                avg_validation_mae = training_results_df['validation_mean_absolute_error'].mean()
                if avg_validation_mae < best_validation_mae:
                    best_training_loss = avg_training_loss
                    best_validation_loss = avg_validation_loss
                    best_training_mae = avg_training_mae
                    best_validation_mae = avg_validation_mae
                    best_time_step = time_step
                    best_forecast_step = forecast_step
                    logi(f"Found a better period combination: \ntime_step: {time_step}, feature_step: {forecast_step}")
                    self.print_stats(config,
                                     best_training_loss,
                                     best_validation_loss,
                                     best_training_mae,
                                     best_validation_mae,
                                     best_time_step,
                                     best_forecast_step)

        #  Print final stats
        self.print_stats(config,
                         best_training_loss,
                         best_validation_loss,
                         best_training_mae,
                         best_validation_mae,
                         best_time_step,
                         best_forecast_step)

        logi('Done training models')
