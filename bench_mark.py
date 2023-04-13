import utils
from config import config as cf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import train
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from dataset import TimeSeriesDataset, Classification_TimeSeriesDataset
import infer
from plot import to_plot
import pandas_ta as ta
import bench_mark_model
def  train_arima(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = cf["model"]["assemble_1"]["window_size"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)

    # date modified 
    train_date = train_date[ int(len(train_date) - len(train_df)) :]
    valid_date = valid_date[ int(len(valid_date) - len(valid_df)) :]

    test_date = test_date[ int(len(test_date) - len(test_df)) :]
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    train_n_row = len(train_close_df) - window_size
    valid_n_row = len(valid_close_df) - window_size
    test_n_row = len(test_close_df) - window_size

    # calculate y
    y_train = utils.prepare_timeseries_data_y(train_n_row, train_close_df.to_numpy(), window_size= window_size,output_size=1)
    y_valid = utils.prepare_timeseries_data_y(valid_n_row, valid_close_df.to_numpy(), window_size= window_size,output_size=1)
    y_test = utils.prepare_timeseries_data_y(test_n_row, test_close_df.to_numpy(), window_size= window_size, output_size=1)

    # date modified 
    train_date = train_date[ int(len(train_date) - len(y_train)) :]
    valid_date = valid_date[ int(len(valid_date) - len(y_valid)) :]
    test_date = test_date[ int(len(test_date) - len(y_test)) :]
    # close_df and dataset_df should be the same
    X_train = utils.prepare_timeseries_data_x(train_df.to_numpy(), window_size = window_size)
    X_valid = utils.prepare_timeseries_data_x(valid_df.to_numpy(), window_size = window_size)
    X_test = utils.prepare_timeseries_data_x(test_df.to_numpy(), window_size = window_size)

   
if __name__ == "__main__":
    data_df, num_data_points, data_dates = utils.download_data_api()
    data_df.set_index('date', inplace=True)
    train_df, valid_df, test_df, train_date, valid_date, test_date = utils.split_train_valid_test_dataframe(data_df, num_data_points, data_dates)