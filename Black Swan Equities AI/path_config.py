import os

#  Contains path configuration for your environment
ROOT_DIR = "C:\\dev\\trading\\aiscreener1"

#  Define directories
LOG_DIR = os.path.join(ROOT_DIR, "logs")
KEY_DIR = os.path.join(ROOT_DIR, "keys")
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_CACHE_DIR = os.path.join(ROOT_DIR, "cache")
STOCK_DATA_DIR = os.path.join(DATA_DIR, "stock_data")
MARKET_DATA_DIR = os.path.join(DATA_DIR, "market_data")
ECONOMIC_DATA_DIR = os.path.join(DATA_DIR, "economic_data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
RESULTS_METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
RESULTS_TRAINING_PLOTS_DIR = os.path.join(RESULTS_DIR, "training_plots")
RESULTS_HYPERPARAMS_DIR = os.path.join(RESULTS_DIR, "hyperparams")
RESULTS_MODELS_DIR = os.path.join(RESULTS_DIR, "models")
RESULTS_STRATEGIES_DIR = os.path.join(RESULTS_DIR, "strategies")