from datetime import timedelta
from utils.log_utils import *
from path_config import *
from loaders.fred_loader import FredEconomicDataLoader
from loaders.simfin_loader import SimfinDataLoader
from loaders.tiingo_loader import TiingoDataLoader
from assemblers.data_assembler import DataAssembler
from optimizers.feature_optimizer1 import FeatureOptimizer1
from optimizers.period_optimizer1 import PeriodOptimizer1
from optimizers.hyperparam_optimizer1 import HyperparamOptimizer1
from strategies.simple_long_strategy1 import SimpleLongStrategy1
from trainers.trainer1 import Trainer1


def setup_paths():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(KEY_DIR):
        os.makedirs(KEY_DIR)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(DATA_CACHE_DIR):
        os.makedirs(DATA_CACHE_DIR)
    if not os.path.exists(STOCK_DATA_DIR):
        os.makedirs(STOCK_DATA_DIR)
    if not os.path.exists(MARKET_DATA_DIR):
        os.makedirs(MARKET_DATA_DIR)
    if not os.path.exists(ECONOMIC_DATA_DIR):
        os.makedirs(ECONOMIC_DATA_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if not os.path.exists(RESULTS_METRICS_DIR):
        os.makedirs(RESULTS_METRICS_DIR)
    if not os.path.exists(RESULTS_TRAINING_PLOTS_DIR):
        os.makedirs(RESULTS_TRAINING_PLOTS_DIR)
    if not os.path.exists(RESULTS_HYPERPARAMS_DIR):
        os.makedirs(RESULTS_HYPERPARAMS_DIR)
    if not os.path.exists(RESULTS_MODELS_DIR):
        os.makedirs(RESULTS_MODELS_DIR)
    if not os.path.exists(RESULTS_STRATEGIES_DIR):
        os.makedires(RESULTS_STRATEGIES_DIR)

def perform_data_loads(start_date_str, end_date_str, max_items):
    logi("Starting data loads")

    #  Load SimFin data
    simfin_loader = SimfinDataLoader()
    simfin_loader.load(max_items)

    #  Load FRED data - automatically limited to 10 years
    fred_loader = FredEconomicDataLoader()
    fred_loader.load(start_date_str)

    #  Load Tiingo data
    tiingo_loader = TiingoDataLoader()
    tiingo_loader.load(start_date_str, end_date_str)

    logi("Data loads complete")


#  MAIN
setup_paths()

setup_logger("aiscreener1-log.txt")
logi("Start aiscreener1")

#  Configuration
max_items = 10

#  Set start and end dates for data loads - 10 years of data
start_date = datetime.now() - timedelta(days=365 * 10)
start_date_str = start_date.strftime('%Y-%m-%d')
end_date = datetime.now()
end_date_str = end_date.strftime('%Y-%m-%d')

#  Start data loads
#perform_data_loads(start_date_str, end_date_str, max_items)

#  Assemble data
#assembler = DataAssembler()
#assembler.assemble()

#  Train models
#trainer1 = Trainer1()
#trainer1.train_all()

#  Optimize features
#feature_optimizer1 = FeatureOptimizer1()
#feature_optimizer1.optimize(max_items)

#  Optimize periods
#period_optimizer1 = PeriodOptimizer1()
#period_optimizer1.optimize(max_items)

#  Optimize hyperparameters
#hyperparam_optimizer1 = HyperparamOptimizer1()
#hyperparam_optimizer1.optimize_all()

#  Optimize strategy
strategy = SimpleLongStrategy1()
strategy.run()

logi("aiscreener1 complete.")

