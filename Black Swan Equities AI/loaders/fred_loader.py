import pandas_datareader as pdr
from utils.log_utils import *
import pandas as pd


#  Define constants for FRED economic data
data_source = 'fred'
ten_year_treasury_yield_code = 'DGS10'  # 10-year Treasury Rate
two_year_treasury_yield_code = 'DGS2'  # 2-year Treasury Rate
unemployment_rate_code = 'UNRATE'
gdp_code = 'GDPC1'
cpi_series = 'CPIAUCSL'
consumer_opinion_survey_code = 'CSCICP03USM665S'
non_farm_payroll_code = 'PAYEMS'
retail_food_sales_code = 'MRTSSM44X72USS'
producer_price_index_code = 'PPIACO'
industrial_production_index_code = 'INDPRO'


class FredEconomicDataLoader():
    """
    Fetches data using FRED Data API
    """

    def __init__(self):
        self._name = "FredDataLoader"

    def fetch_fred_data(self, code_list, start_date_str):
        try:
            df = pdr.DataReader(code_list, data_source, start_date_str)
            df.reset_index(inplace=True)

            #  Cleanup
            df['DATE'] = pd.to_datetime(df['DATE'], '%Y-%m-%d %H:%M:%S')

            #  Fill missing data
            df.fillna(method='ffill', inplace=True)
            df = df.fillna(0)
            return df
        except Exception as ex:
            loge(f"Failed in {self._name}.fetch_fred_data(): {ex}")
            return None

    def resample_to_daily(self, df):
        df.set_index('DATE', inplace=True)

        # Resample the dataframe to fill missing days with the last observed value
        df_resampled = df.resample('D').ffill()

        df_resampled.reset_index(inplace=True)
        return df_resampled

    def load_fred_data(self, code, start_date_str, file_name, do_resample=False):
        logi(f"Fetching FRED data for {code}")
        df = self.fetch_fred_data([code], start_date_str)

        #  Resample to daily data
        if do_resample:
            df = self.resample_to_daily(df)

        #  Store data as CSV
        path = os.path.join(ECONOMIC_DATA_DIR, file_name)
        df.to_csv(path)

    def load(self, start_date_str):
        logi("Starting FRED data loads...")
        self.load_fred_data(two_year_treasury_yield_code, start_date_str, "treasury_yield_2y.csv", False)
        self.load_fred_data(ten_year_treasury_yield_code, start_date_str, "treasury_yield_10y.csv", False)
        self.load_fred_data(gdp_code, start_date_str, "gdp_data.csv", True)
        self.load_fred_data(non_farm_payroll_code, start_date_str, "non_farm_payroll_data.csv", True)
        self.load_fred_data(unemployment_rate_code, start_date_str, "unemployment_data.csv", True)
        self.load_fred_data(consumer_opinion_survey_code, start_date_str, "consumer_opinion_survey_data.csv", True)
        self.load_fred_data(cpi_series, start_date_str, "cpi_series_data.csv", True)
        self.load_fred_data(retail_food_sales_code, start_date_str, "retail_food_sales_data.csv", True)
        self.load_fred_data(producer_price_index_code, start_date_str, "producer_price_index_data.csv", True)
        self.load_fred_data(industrial_production_index_code, start_date_str, "industrial_production_index_data.csv", True)
        logi("FRED data load complete")


