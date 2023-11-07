import simfin as sf
import pandas as pd
from utils.file_utils import read_api_key_file
from path_config import *
from utils.log_utils import *
# Suppress slice warning
pd.options.mode.chained_assignment = None


class SimfinDataLoader():
    """
    Fetches data using SimFin Data API
    """

    def __init__(self):
        self._name = "SimfinDataLoader"
        self.configure_simfin()

    def configure_simfin(self):
        sf.set_data_dir(DATA_CACHE_DIR)
        sf_api_key = read_api_key_file('simfin_api_key.txt')
        sf.set_api_key(sf_api_key)

    def load(self, max_items=None):
        logi("Starting SimFin data load")

        # Data for US Market
        market = 'us'

        # Daily Share-Prices.
        logi("Downloading daily prices")
        df_prices = sf.load_shareprices(variant='daily', market=market)

        # TTM Income Statements - quarterly
        logi("Downloading income statements")
        df_income_ttm = sf.load_income(variant='ttm', market=market)

        # Sort the index (required for resampling)
        df_income_ttm.sort_index(inplace=True)

        # Resample to daily frequency for each 'Ticker'.
        # Note that we need to drop the 'Ticker' index to be able to perform the resampling
        df_income_daily = df_income_ttm.groupby(level=0).apply(
            lambda x: x.reset_index(level=0, drop=True).resample('D').asfreq())

        # Forward fill the missing values for each 'Ticker'
        df_income_daily = df_income_daily.groupby('Ticker').fillna(method='ffill')

        # TTM Balance Sheets  - quarterly
        logi("Downloading balance sheets")
        df_balance_ttm = sf.load_balance(variant='ttm', market=market)

        # TTM Cash-Flow Statements - quarterly
        logi("Downloading cashflow statements")
        df_cashflow_ttm = sf.load_cashflow(variant='ttm', market=market)

        #  Calculate value signals - ffill daily data
        logi("Calculating value signals")
        df_value_signals = sf.val_signals(df_prices=df_prices,
                                          df_income_ttm=df_income_ttm,
                                          df_balance_ttm=df_balance_ttm,
                                          df_cashflow_ttm=df_cashflow_ttm,
                                          fill_method='ffill')

        #  Calculate financial signals - ffill daily data
        logi("Calculating financial signals")
        df_financial_signals = sf.fin_signals(df_prices=df_prices,
                                              df_income_ttm=df_income_ttm,
                                              df_balance_ttm=df_balance_ttm,
                                              df_cashflow_ttm=df_cashflow_ttm,
                                              fill_method='ffill')

        #  Select the columns we need
        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividend']
        prices_df = df_prices[price_columns]

        income_columns = ['Revenue', 'Net Income']
        df_income_daily = df_income_daily[income_columns]

        value_columns = ['P/E', 'P/Sales']
        df_value_signals = df_value_signals[value_columns]

        financial_columns = ['Current Ratio',
                             'Debt Ratio',
                             'Quick Ratio',
                             'Gross Profit Margin',
                             'Net Profit Margin',
                             'Return on Assets',
                             'Return on Equity']
        df_financial_signals = df_financial_signals[financial_columns]

        #  Process each ticker separately
        symbol_list = prices_df.reset_index()['Ticker'].unique()
        counter = 0
        for symbol in symbol_list:
            if '_old' in symbol or '_delist' in symbol:
                continue

            if max_items is not None and counter > max_items:
                break
            counter += 1

            #  Get dfs for each symbol
            temp_df = prices_df.reset_index()
            symbol_prices_df = temp_df[temp_df['Ticker'] == symbol]
            symbol_prices_df.set_index('Date', inplace=True)
            symbol_prices_df.drop(columns=['Ticker'], inplace=True)

            #  Check minimum data length
            if len(symbol_prices_df) < 500:
                logi(f"{symbol} does not have enough data")
                continue

            temp_df = df_income_daily.reset_index()
            symbol_income_daily_df = temp_df[temp_df['Ticker'] == symbol]
            symbol_income_daily_df.set_index('Report Date', inplace=True)
            symbol_income_daily_df.drop(columns=['Ticker'], inplace=True)

            temp_df = df_value_signals.reset_index()
            symbol_value_signals_df = temp_df[temp_df['Ticker'] == symbol]
            symbol_value_signals_df.set_index('Date', inplace=True)
            symbol_value_signals_df.drop(columns=['Ticker'], inplace=True)

            temp_df = df_financial_signals.reset_index()
            symbol_financial_signals_df = temp_df[temp_df['Ticker'] == symbol]
            symbol_financial_signals_df.set_index('Date', inplace=True)
            symbol_financial_signals_df.drop(columns=['Ticker'], inplace=True)

            #  Merge all dataframes
            logi(f"Merging SimFin dataframes for symbol {symbol}")
            merged_df = pd.merge(symbol_prices_df, symbol_income_daily_df, left_index=True, right_index=True, how='left')
            merged_df = pd.merge(merged_df, symbol_value_signals_df, left_index=True, right_index=True, how='left')
            merged_df = pd.merge(merged_df, symbol_financial_signals_df, left_index=True, right_index=True, how='left')

            #  Fill remaining nan values with zeros
            merged_df.fillna(0, inplace=True)

            # Convert all column names to uppercase
            merged_df.columns = merged_df.columns.str.upper()

            file_name = f"{symbol}_price_fundamentals_df.csv"
            path = os.path.join(STOCK_DATA_DIR, file_name)
            merged_df.to_csv(path)

