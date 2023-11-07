from utils.log_utils import *
from matplotlib import pyplot as plt


def plot_future_predictions(symbol, future_df):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(future_df['CLOSE'], color='blue', label='Real close price')
    plt.plot(future_df['PREDICTED_CLOSE'], color='red', label='Predicted close price')
    plt.title('Close Price Prediction on Future Data Set')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()

    # Save the plot
    file_name = f"{symbol}_predictions_plot.png"
    path = os.path.join(RESULTS_STRATEGIES_DIR, file_name)
    plt.savefig(path)


def print_strategy_results(results_df):
    logi("\nStrategy Results:\n===============================")
    logi(f"Number of trades: {len(results_df)}")
    logi(f"Total Profit: {round(results_df['Profit'].sum(),2)}")


def print_best_combination_result(results_df):
    logi("\nBest combination Resuls:\n===============================")
    logi(f"RSI: {results_df['RSI'].iloc[0]}, Predicted Change: {results_df['Predicted Change'].iloc[0]} \
            Take Profit: {results_df['TP'].iloc[0]}, Stop loss: {results_df['SL'].iloc[0]}, Profit: {results_df['Profit'].iloc[0]}")