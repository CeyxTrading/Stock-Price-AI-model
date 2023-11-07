import requests
from utils.log_utils import *
import pandas as pd
from utils.file_utils import read_api_key_file

# List of all SPDR Sector ETFs symbols
spdr_sector_etfs = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLRE', 'XLK', 'XLU']

#  List of market indexes
market_index_symbols = ['SPY', 'IWD']


class TiingoDataLoader():
    """
    Fetches data using TIINGO Data API
    """

    def __init__(self):
        self._name = "TiingoDataLoader"
        self._tiingo_api_key = read_api_key_file('tiingo_api_key.txt')

    def build_headers(self):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Token ' + self._tiingo_api_key
        }
        return headers

    def fetch_tiingo_data(self, symbol, start_date_str, end_date_str):
        headers = self.build_headers()
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date_str}&endDate={end_date_str}"

        #  Fetch data
        response = requests.get(url, headers=headers)

        # Create a DataFrame
        data = response.json()
        df = pd.DataFrame(data)

        #  Fill missing data
        df.fillna(method='ffill', inplace=True)
        df = df.fillna(0)

        #  drop unneeded columns
        df.drop(columns=['open', 'high', 'low', 'close', 'volume', 'divCash', 'splitFactor'], inplace=True)

        #  Rename columns with symbol
        df.rename(columns={"date": f"DATE", \
                           "adjOpen": f"{symbol}_OPEN", \
                           "adjHigh": f"{symbol}_HIGH", \
                           "adjLow": f"{symbol}_LOW", \
                           "adjClose": f"{symbol}_CLOSE", \
                           "adjVolume": f"{symbol}_VOLUME"}, inplace=True)

        return df

    def fetch_multiple_symbols(self, symbol_list, start_date_str, end_date_str):
        for symbol in symbol_list:
            logi(f"Fetching data for {symbol}")

            #  Fetch prices
            df = self.fetch_tiingo_data(symbol, start_date_str, end_date_str)

            #  Store data
            path = os.path.join(MARKET_DATA_DIR, f"{symbol}_data.csv")
            df.to_csv(path, index=False)

    def load(self, start_date_str, end_date_str):
        #  Fetch Sector data
        logi("Fetching sector data")
        self.fetch_multiple_symbols(spdr_sector_etfs, start_date_str, end_date_str)

        #  Fetch market index data
        logi("Fetching sector data")
        self.fetch_multiple_symbols(market_index_symbols, start_date_str, end_date_str)
        logi("Tiingo data load complete")
