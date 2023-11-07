import os
from path_config import *
import pandas as pd
import numpy as np
import shutil


def read_api_key_file(file_name):
    path = os.path.join(KEY_DIR, file_name)
    if os.path.exists(path):
        file = open(path, "r")
        lines = file.readlines()
        api_key = lines[0].replace('\'', '')
        file.close()
        return api_key


def read_csv_file(path):
    if os.path.exists(path):
        df = pd.read_csv(path)
        #  Remove 'Unnamed columns'
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        return df
    return None


def get_file_list(path):
    if os.path.exists(path):
        file_list = os.listdir(path)
        return file_list
    return []


def load_combined_data_file(file_name):
    #  Read the file
    path = os.path.join(STOCK_DATA_DIR, file_name)
    df = read_csv_file(path)
    if df is None:
        return None

    # Set Date as index
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)

    #  Perform additional cleaning
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)
    return df[indices_to_keep].astype(np.float64)


def get_all_features():
    #  Find the first combined data file and get list of columns
    file_list = get_file_list(STOCK_DATA_DIR)
    all_features = []
    for file_name in file_list:
        if "combined" not in file_name:
            continue

        #  Read data file
        df = load_combined_data_file(file_name)
        if df is not None:
            all_features = df.columns.values.tolist()
        break
    return all_features


def delete_dir(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))


def load_hyperparameters_file():
    file_name = "all_hyperparam_results_df.csv"
    path = os.path.join(RESULTS_HYPERPARAMS_DIR, file_name)
    # Load the file
    hyperparams_df = pd.read_csv(path)

    # Convert the dataframe into a dictionary with the symbol as lookup key
    hyperparams_dict = hyperparams_df.set_index('symbol').T.to_dict()

    return hyperparams_dict

