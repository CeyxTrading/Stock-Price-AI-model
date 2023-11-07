from utils.file_utils import *
from utils.log_utils import *
import pandas_ta as ta
import talib


class DataAssembler():
    """
    Assembles the different pieces of data, calculates technical indicators and candle patterns
    """
    def __init__(self):
        self._name = "DataAssembler"

    def load_market_data(self):
        market_data_list = []
        file_list = get_file_list(MARKET_DATA_DIR)
        for file_name in file_list:
            path = os.path.join(MARKET_DATA_DIR, file_name)
            df = read_csv_file(path)
            if df is not None:
                #  Set date index
                df['DATE'] = pd.to_datetime(df['DATE'])
                df.set_index('DATE', inplace=True)
                market_data_list.append(df)
        return market_data_list

    def load_economic_data(self):
        economic_data_list = []
        file_list = get_file_list(ECONOMIC_DATA_DIR)
        for file_name in file_list:
            path = os.path.join(ECONOMIC_DATA_DIR, file_name)
            df = read_csv_file(path)
            if df is not None:
                #  Set date index
                df['DATE'] = pd.to_datetime(df['DATE'])
                df.set_index('DATE', inplace=True)
                df.index = df.index.tz_localize('UTC')  # Convert to timezone aware

                economic_data_list.append(df)
        return economic_data_list

    def add_technical_indicators(self, df):
        #  MAs
        df['MA_200'] = talib.SMA(df['CLOSE'], timeperiod=200)
        df['MA_50'] = talib.SMA(df['CLOSE'], timeperiod=50)
        df['MA_20'] = talib.SMA(df['CLOSE'], timeperiod=20)

        # EMAs
        df['EMA_12'] = talib.EMA(df['CLOSE'], timeperiod=12)
        df['EMA_7'] = talib.EMA(df['CLOSE'], timeperiod=7)
        df['EMA_5'] = talib.EMA(df['CLOSE'], timeperiod=5)
        df['EMA_3'] = talib.EMA(df['CLOSE'], timeperiod=3)

        #  Percentage changes
        df['PCT_CHANGE_1D'] = df['CLOSE'].pct_change(periods=1)
        df['PCT_CHANGE_2D'] = df['CLOSE'].pct_change(periods=2)
        df['PCT_CHANGE_3D'] = df['CLOSE'].pct_change(periods=3)
        df['PCT_CHANGE_5D'] = df['CLOSE'].pct_change(periods=5)
        df['PCT_CHANGE_1M'] = df['CLOSE'].pct_change(periods=20)
        df['PCT_CHANGE_3M'] = df['CLOSE'].pct_change(periods=60)
        df['PCT_CHANGE_6M'] = df['CLOSE'].pct_change(periods=130)
        df['PCT_CHANGE_12M'] = df['CLOSE'].pct_change(periods=260)

        #  Add lags
        df['LAG_P1'] = df['CLOSE'].shift(1)
        df['LAG_P2'] = df['CLOSE'].shift(2)
        df['LAG_P3'] = df['CLOSE'].shift(3)
        df['LAG_P4'] = df['CLOSE'].shift(4)
        df['LAG_P5'] = df['CLOSE'].shift(5)
        df['LAG_P10'] = df['CLOSE'].shift(10)
        df['LAG_P15'] = df['CLOSE'].shift(15)
        df['LAG_P21'] = df['CLOSE'].shift(21)

        # Add +DMI and -DMI
        temp_df = ta.adx(df['HIGH'], df['LOW'], df['CLOSE'], length=14)
        df['DMI_POS'] = temp_df['DMP_14']
        df['DMI_NEG'] = temp_df['DMN_14']

        # Add Fisher Transform
        temp_df = ta.fisher(df['HIGH'], df['LOW'], length=9)
        df['FISHER_TRANSFORM'] = temp_df['FISHERT_9_1']
        df['FISHER_TRANSFORM_SIGNAL'] = temp_df['FISHERTs_9_1']

        # Add MACD
        temp_df = ta.macd(df['CLOSE'], fast=12, slow=26, signal=9)
        df['MACD'] = temp_df["MACD_12_26_9"]
        df['MACD_SIGNAL'] = temp_df["MACDs_12_26_9"]

        # Add RSI
        df['RSI'] = ta.rsi(df['CLOSE'], length=14)

        # Add upper and lower Bollinger Bands
        temp_df = ta.bbands(close=df['CLOSE'], length=20, std=2.0)
        df['BB_UPPER'] = temp_df["BBU_20_2.0"]
        df['BB_LOWER'] = temp_df["BBL_20_2.0"]

        #  Add ATR
        df['ATR'] = ta.atr(high=df['HIGH'], low=df['LOW'], close=df['CLOSE'], length=14)

        #  Add OBV
        df['OVB'] = ta.obv(close=df['CLOSE'], volume=df['VOLUME'])

        #  Discard the first 200 periods which were needed for MA calculation
        df = df.iloc[199:]

        return df

    def add_candlestick_patterns(self, df):
        # Ensure columns have a numeric format
        df['OPEN'] = pd.to_numeric(df['OPEN'], errors='coerce')
        df['HIGH'] = pd.to_numeric(df['HIGH'], errors='coerce')
        df['LOW'] = pd.to_numeric(df['LOW'], errors='coerce')
        df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')

        # Calculating candlestick patterns
        df['THREE_WHITE_SOLDIERS'] = talib.CDL3WHITESOLDIERS(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        df['THREE_BLACK_CROWS'] = talib.CDL3BLACKCROWS(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        df['IDENTICAL_THREE_CROWS'] = talib.CDLIDENTICAL3CROWS(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        df['THREE_LINE_STRIKE'] = talib.CDL3LINESTRIKE(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        df['MORNING_STAR'] = talib.CDLMORNINGSTAR(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'], penetration=0)
        df['EVENING_STAR'] = talib.CDLEVENINGSTAR(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'], penetration=0)
        df['THREE_OUTSIDE_UP'] = talib.CDL3OUTSIDE(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        df['THREE_OUTSIDE_DOWN'] = talib.CDL3OUTSIDE(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        df['ENGULFING'] = talib.CDLENGULFING(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        df['BELT_HOLD'] = talib.CDLBELTHOLD(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        df['ABANDONED_BABY'] = talib.CDLABANDONEDBABY(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'], penetration=0)
        df['SEPARATING_LINES'] = talib.CDLSEPARATINGLINES(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])
        df['DOJI_STAR'] = talib.CDLDOJISTAR(df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'])

        return df

    def merge_stock_data_market_data(self, df, market_data_list):
        for market_df in market_data_list:
            #  Perform a left join on date
            df = pd.merge(df, market_df, how='left', left_index=True, right_index=True)
        return df

    def merge_stock_data_economic_data(self, df, economic_data_list):
        for economic_df in economic_data_list:
            #  Perform a left join on date
            df = pd.merge(df, economic_df, how='left', left_index=True, right_index=True)

            #  Forward-fill any missing economic data
            df.fillna(method='ffill', inplace=True)
        return df

    def process_stock_data_file(self, df, market_data_list, economic_data_list):
        #  Set Date index
        df.rename(columns={'Date': 'DATE'}, inplace=True)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
        df.index = df.index.tz_localize('UTC')  # Convert to timezone aware

        #  Add candlestick patterns
        df = self.add_candlestick_patterns(df)

        #  Add technical indicators
        df = self.add_technical_indicators(df)

        #  Merge with market data
        df = self.merge_stock_data_market_data(df, market_data_list)

        #  Merge with economic data
        df = self.merge_stock_data_economic_data(df, economic_data_list)

        #  Fill remaining nan values
        df.fillna(0, inplace=True)

        return df

    def process_stock_data_files(self, market_data_list, economic_data_list):
        #  Get list of files
        file_list = get_file_list(STOCK_DATA_DIR)
        for file_name in file_list:
            if "price_fundamentals" not in file_name:
                continue
            logi(f"Processing {file_name}")
            symbol = file_name.split("_")[0]

            #  Read the file
            path = os.path.join(STOCK_DATA_DIR, file_name)
            df = read_csv_file(path)
            if df is None:
                continue

            #  Add indicators, candlestick patterns and merge with market- and economic data
            df = self.process_stock_data_file(df, market_data_list, economic_data_list)

            #  Store combined dataframe as CSV
            file_name = f"{symbol}_combined.csv"
            path = os.path.join(STOCK_DATA_DIR, file_name)
            df.to_csv(path)

    def assemble(self):
        logi("Assembling data files")

        #  Load market data
        market_data_list = self.load_market_data()

        #  Load economic data
        economic_data_list = self.load_economic_data()

        #  Process stock data
        self.process_stock_data_files(market_data_list, economic_data_list)

        logi('Done assembling files')
