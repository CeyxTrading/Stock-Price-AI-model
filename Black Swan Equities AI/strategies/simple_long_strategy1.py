import os
import pandas as pd
import pandas_ta as ta
from utils.log_utils import *
from utils.file_utils import *
from utils.training_utils import *
from utils.strategy_utils import *

'''
STRATEGY RULES:
Long Entry:
- RSI < RSI low threshold (to identify oversold positions)
- Current day price prediction greater or equal percentage of price

Long Exit
- Take profit target of percentage of the price prediction price (to account for AI inaccuracy)
- Stop loss as percentage of the price
'''


class SimpleLongStrategy1():
    """
    Implements the strategy rules above
    """
    def __init__(self):
        self._name = "SimpleLongStrategy1"
        setup_logger(self._name + '.txt')

    def train_and_predict(self, config):
        #  Set the features
        best_features = ['OPEN', 'HIGH', 'LOW', 'VOLUME', 'CLOSE',
                         'QUICK RATIO', 'THREE_WHITE_SOLDIERS',
                         'THREE_BLACK_CROWS', 'IDENTICAL_THREE_CROWS',
                         'EVENING_STAR', 'THREE_OUTSIDE_DOWN',
                         'PCT_CHANGE_1D', 'PCT_CHANGE_1M', 'XLRE_VOLUME']

        #  Load the best hyperparams
        hyperparam_dict = load_hyperparameters_file()

        #  Get list of files
        future_predictions_dict = {}
        file_list = get_file_list(STOCK_DATA_DIR)
        for file_name in file_list:
            if "combined" not in file_name:
                continue
            logi(f"Processing {file_name}")

            #  Get symbol from file name
            symbol = file_name.split("_")[0]

            #  Skip symbols with bad training results
            if symbol in ['AAMC', 'AAOI']:
                continue

            #  Load the file
            df = load_combined_data_file(file_name)

            #  Make sure we have some data for training
            if df is None or len(df) < 260:
                logi(f"No data for symbol {symbol}, file: {file_name}")
                continue

            #  Select the best features
            df = df[best_features]

            #  Get the best hyperparams for this stock
            hyperparams = hyperparam_dict[symbol]

            #  Train and save the model
            history, future_predictions_df = perform_training_and_predictions(symbol, df, config, hyperparams)

            #  Visualize predictions
            plot_future_predictions(symbol, future_predictions_df)

            #  Add indicator
            future_predictions_df['RSI'] = ta.rsi(future_predictions_df['CLOSE'], window=14)

            #  Add symbol
            future_predictions_dict[symbol] = future_predictions_df
        return future_predictions_dict

    def run_strategy(self, predictions_df, predicted_change, rsi, tp, sl):
        # Initialize required variables
        long_position = False
        buy_price = None
        sell_price = None
        take_profit_target = None
        stop_loss_target = None
        peak_price = 0

        results = []
        for i in range(0, len(predictions_df)):
            close = predictions_df.iloc[i]["CLOSE"]
            predicted_close = predictions_df.iloc[i]["PREDICTED_CLOSE"]
            rsi_value = predictions_df.iloc[i]["RSI"]
            predicted_change_value = ((predicted_close - close) / close) * 100

            #  Enter long
            if not long_position:
                if rsi_value < rsi and predicted_change_value >= predicted_change:
                    buy_price = close
                    take_profit_target = close * (1 + tp/100)
                    stop_loss_target = buy_price * (1 - sl/100)
                    long_position = True

            #  In case we have a position
            if long_position:
                #  Track the peak price
                if close > peak_price:
                    peak_price = close
                #  Exit long
                exit_long = False
                #  Stop loss
                if close < stop_loss_target:
                    exit_long = True
                #  Check take profit target
                if close >= take_profit_target:
                    exit_long = True

                #  Close the trade
                if exit_long is True:
                    sell_price = close
                    results.append({"Index": i, "Buy": buy_price, "Sell": sell_price, "Profit": sell_price - buy_price})
                    buy_price = None
                    sell_price = None
                    take_profit_target = None
                    stop_loss_target = None
                    peak_price = 0
                    long_position = False
        #  If there is still an open long position after all data is processed, we close it automatically
        if long_position is True:
            close = predictions_df.iloc[-1]["CLOSE"]
            sell_price = close
            results.append({"Index": i, "Buy": buy_price, "Sell": sell_price, "Profit": sell_price - buy_price})

        #  Convert results array to dataframe
        results_df = pd.DataFrame(results)

        trades = 0
        profit = 0
        if len(results_df) > 0:
            #  Calculate totals
            trades = len(results_df)
            profit = results_df['Profit'].sum()

        return trades, profit

    def optimize_strategy(self, future_predictions_dict, config):
        all_combination_results_df = pd.DataFrame()
        for predicted_change in range(config['predicted_change_low'], config['predicted_change_high'], config['predicted_change_step']):
            for rsi in range(config['rsi_low'], config['rsi_high'], config['rsi_step']):
                for tp in range(config['tp_pc_low'], config['tp_pc_high'], config['tp_pc_step']):
                    for sl in range(config['sl_pc_low'], config['sl_pc_high'], config['sl_pc_step']):
                        logi(f"Testing combination: Predicted Change: {predicted_change}, RSI {rsi}, Take Profit: {tp}, Stop Loss: {sl}")

                        #  Iterate through each symbol and run the strategy
                        symbols = future_predictions_dict.keys()
                        combination_results_df = pd.DataFrame()
                        for symbol in symbols:
                            #  Get predictions
                            predictions_df = future_predictions_dict[symbol]

                            # Run strategy
                            trades, profit = self.run_strategy(predictions_df, predicted_change, rsi, tp, sl)
                            strategy_result_df = pd.DataFrame({'Symbol': [symbol], 'Trades': [trades], 'Profit': [profit]})
                            combination_results_df = pd.concat([combination_results_df, strategy_result_df], axis=0, ignore_index=True)

                        #  Get totals for all stocks for this strategy combination
                        all_symbol_results_df = pd.DataFrame({'Predicted Change': [predicted_change],
                                                                          'RSI': [rsi],
                                                                          'TP': [tp],
                                                                          'SL': [sl],
                                                                          'Trades': [combination_results_df['Trades'].sum()],
                                                                          'Profit': [combination_results_df['Profit'].sum()]})
                        all_combination_results_df = pd.concat([all_combination_results_df, all_symbol_results_df],
                                                               axis=0, ignore_index=True)

        return all_combination_results_df

    def run(self):
        #  Configure the model. Use the same values as you used for training!!!
        config = {
            "epochs": 200,
            "batch_size": 64,
            "patience": 24,
            "learning_rate": 0.001,
            "time_steps": 5,
            "forecast_steps": 1,
            "train_ratio": 0.6,
            "test_ratio": 0.2,
            "forecast_ratio": 0.2,
            "predicted_change_low": 1,
            "predicted_change_high": 10,
            "predicted_change_step": 1,
            "rsi_low": 10,
            "rsi_high": 40,
            "rsi_step": 5,
            "tp_pc_low": 1,
            "tp_pc_high": 10,
            "tp_pc_step": 1,
            "sl_pc_low": 1,
            "sl_pc_high": 10,
            "sl_pc_step": 1
        }

        #  Train and generate predictions
        future_predictions_dict = self.train_and_predict(config)

        #  Optimize strategy
        all_combination_results_df = self.optimize_strategy(future_predictions_dict, config)

        #  Sort by profit
        all_combination_results_df = all_combination_results_df.sort_values(by=['Profit'], ascending=False)

        #  Store result
        path = os.path.join(RESULTS_STRATEGIES_DIR, 'all_combination_results_df.csv')
        all_combination_results_df.to_csv(path)

        #  Print best result
        print_best_combination_result(all_combination_results_df)

