from utils.file_utils import *
from utils.log_utils import *
from utils.training_utils import *


class FeatureOptimizer1:
    """
    Selectively adds features to the model in order to try to enhance the MAE
    """

    def __init__(self):
        self._name = "FeatureOptimizer1"
        setup_logger(self._name + '.txt')

    def perform_test_for_feature_set(self, test_features, config):
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
                                      'validation_mean_absolute_error': [
                                          history.history['val_mean_absolute_error'][-1]]})
            training_results_df = pd.concat([training_results_df, result_df], axis=0, ignore_index=True)

            #  Only process files up to max items
            item_counter += 1
            if item_counter > max_items:
                break

        return training_results_df

    def print_stats(self, config, best_training_loss, best_validation_loss, best_training_mae, best_validation_mae,
                    best_features):
        logi("\n\nStats\n===============================")
        for key in config:
            logi(f"{key}: {config[key]}")

        round_digits = 8
        logi(f"Best Training Loss: {round(best_training_loss, round_digits)}")
        logi(f"Best Validation Loss: {round(best_validation_loss, round_digits)}")
        logi(f"Best Training MAE: {round(best_training_mae, round_digits)}")
        logi(f"Best Validation MAE: {round(best_validation_mae, round_digits)}")

        logi(f"Best Features:\n{best_features}")

    def optimize(self, max_items):
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
            "forecast_ratio": 0.2,
            "max_items": max_items
        }

        #  Start with the core features
        best_features = ['OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE']
        all_features = get_all_features()

        #  Remove the core features from all feature set
        candidate_features = []
        for feature in all_features:
            if feature not in best_features:
                candidate_features.append(feature)

        #  Run first batch with only the core features
        training_results_df = self.perform_test_for_feature_set(best_features, config)

        #  Set the best values based on training with core features
        best_training_loss = training_results_df['loss'].mean()
        best_validation_loss = training_results_df['validation_loss'].mean()
        best_training_mae = training_results_df['mean_absolute_error'].mean()
        best_validation_mae = training_results_df['validation_mean_absolute_error'].mean()

        #  Print initial stats
        self.print_stats(config, best_training_loss, best_validation_loss, best_training_mae, best_validation_mae,
                         best_features)

        #  Loop through all candidate features and collect metrics
        for feature in candidate_features:
            logi(f"\n\nTesting feature '{feature}'")

            #  Create set of test features
            test_features = best_features + [feature]

            # Perform training with test features
            training_results_df = self.perform_test_for_feature_set(test_features, config)

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
                best_features = test_features
                logi(f"Adding feature {feature} to best features")
                self.print_stats(config, best_training_loss, best_validation_loss, best_training_mae,
                                 best_validation_mae, best_features)

        #  Print final stats
        self.print_stats(config, best_training_loss, best_validation_loss, best_training_mae, best_validation_mae,
                         best_features)

        logi('Done training models')
