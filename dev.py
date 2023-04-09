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

def train_random_tree_classifier_14(data_df, num_data_points, data_date):
    # data_df = utils.get_new_df(data_df, '2018-01-01')

    sma = utils.SMA(data_df['4. close'].values, cf['data']['window_size'])
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], cf['data']['window_size'])
    rsi = utils.RSI(data_df, cf['data']['window_size'])
    vwap = utils.VWAP(data_df, cf['data']['window_size'])
    hma = utils.HMA(data_df['4. close'], cf['data']['window_size'])
    upward = utils.upward(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'],'upward': upward, 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
    dataset_df = dataset_df[15:]
    X = dataset_df.to_numpy()

    window_size = cf["data"]["window_size"]
    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_trend_14 = utils.prepare_tree_data_y_trend(n_row, close, 14)
    X = X[:-14]
    split_index = int(y_trend_14.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X[:split_index]
    X_test = X[split_index:]
    y_train_first = y_trend_14[:split_index]
    y_test = y_trend_14[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]
    # random_tree_classifier = train.train_random_forest_classfier(X_train, y_train, X_val, y_val, X_test, y_test)
    svm_classifier = train.train_svm_classfier(X_train, y_train, X_val, y_val, X_test, y_test)
    
def train_assemble(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):

    window_size = cf["model"]["assemble_1"]["window_size"]
    # data loss due to window_size for indicators
    dataset_df = utils.prepare_dataset_and_indicators(data_df, window_size)

    # close_df and dataset_df should be the same
    close_df = pd.DataFrame({'close': dataset_df['close']})
    features = dataset_df.columns.values
    close = close_df.to_numpy()
    n_row = len(dataset_df) - window_size
    # y_real_1 should be less than close_df = window size
    y_real_1 = utils.prepare_timeseries_data_y(
        num_rows=n_row, 
        data=close, 
        window_size=window_size, 
        output_size=1)
    # X should be equal to dataset_df lenght
    X = dataset_df.to_numpy()
    # X_set should be less than X = window size and equal to y_real
    X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
    loss_data_count = num_data_points - len(X_set)

    # new dates update from the losing datapoints
    dates = data_date[loss_data_count:]

    split_index = int(y_real_1.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_real_1[:split_index]
    y_test = y_real_1[split_index:]

    # data_date lenght must equal X_set Lenght, dates lenght must equal X_train_first lenght

    train_dates_first = dates[:split_index]
    test_dates = dates[split_index:]

    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    
    train_dates = train_dates_first[:split_index]
    val_dates = train_dates_first[split_index:]

    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]
    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_val, y_val)
    dataset_test = TimeSeriesDataset(X_test, y_test)

    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_val, y_val)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_assemble_model_1(dataset_train, dataset_val, features)
    infer.evalute_assembly_regression(dataset_val=dataset_val , features = features)
    infer.evalute_assembly_regression(dataset_val=dataset_test, features = features)
    to_plot(dataset_test, dataset_val, y_test, y_val, num_data_points, dates, test_dates, val_dates)
def train_diff_1(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):
    window_size = cf["model"]["diff_1"]["window_size"]
    max_features = cf["model"]["diff_1"]["max_features"]
    thresh_hold = cf["training"]["diff_1"]["corr_thresh_hold"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    train_close = train_close_df.to_numpy()
    valid_close = valid_close_df.to_numpy()
    test_close = test_close_df.to_numpy()

    train_n_row = len(train_close) - window_size
    valid_n_row = len(valid_close) - window_size
    test_n_row = len(test_close) - window_size

    # calculate y
    y_train = utils.prepare_timeseries_data_y_diff(train_n_row, train_close, output_size = 1)
    y_valid = utils.prepare_timeseries_data_y_diff(valid_n_row, valid_close, output_size = 1)
    y_test = utils.prepare_timeseries_data_y_diff(test_n_row, test_close, output_size = 1)

    # Re allocated df with lossing data points
    train_df = train_df[window_size:]
    valid_df = valid_df[window_size:]
    test_df = test_df[window_size:]
    train_close_df = train_close_df[window_size:]
    valid_close_df = valid_close_df[window_size:]
    test_close_df = test_close_df[window_size:]

    # copy dataframe
    temp_df = train_df.copy()
    # temp_df["target_trend_down"] = y_trend_percentage_3[:, :1]
    temp_df["target"] = y_train
    train_df, features, mask = utils.correlation_filter(dataframe=temp_df, 
                                                    main_columns=["target"], 
                                                    max_columns = max_features,
                                                    threshold=thresh_hold, 
                                                    show_heat_map = show_heat_map)
    X_train = train_df.to_numpy()
    X_valid = valid_df.to_numpy()
    X_test = test_df.to_numpy()

    X_train = utils.prepare_timeseries_data_x(X_train, window_size = window_size)
    X_valid = utils.prepare_timeseries_data_x(X_valid, window_size = window_size)
    X_test = utils.prepare_timeseries_data_x(X_test, window_size = window_size)

    y_train = y_train[window_size:]
    y_valid = y_valid[window_size:]
    y_test = y_test[window_size:]

    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_valid, y_valid)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_LSTM_regression_1(dataset_train, dataset_val, features = features, mask = mask)
    infer.evalute_diff_1(dataset_val=dataset_val, features=features)
    infer.evalute_diff_1(dataset_val=dataset_test, features=features)

def train_movement_3(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):
    
    window_size = cf["model"]["movement_3"]["window_size"]
    max_features = cf["model"]["movement_3"]["max_features"]
    thresh_hold = cf["training"]["movement_3"]["corr_thresh_hold"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    train_close = train_close_df.to_numpy()
    valid_close = valid_close_df.to_numpy()
    test_close = test_close_df.to_numpy()

    train_n_row = len(train_close) - window_size
    valid_n_row = len(valid_close) - window_size
    test_n_row = len(test_close) - window_size

    # calculate y
    y_train = utils.prepare_timeseries_data_y_trend_percentage(train_n_row, train_close, output_size = 3)
    y_valid = utils.prepare_timeseries_data_y_trend_percentage(valid_n_row, valid_close, output_size = 3)
    y_test = utils.prepare_timeseries_data_y_trend_percentage(test_n_row, test_close, output_size = 3)

    # Re allocated df with lossing data points
    train_df = train_df[window_size:]
    valid_df = valid_df[window_size:]
    test_df = test_df[window_size:]
    train_close_df = train_close_df[window_size:]
    valid_close_df = valid_close_df[window_size:]
    test_close_df = test_close_df[window_size:]

    # copy dataframe
    temp_df = train_df.copy()
    # temp_df["target_trend_down"] = y_trend_percentage_3[:, :1]
    temp_df["target_increasing"] = y_train[:, 1:2]
    temp_df["target_percentage"] = y_train[:, 2:]
    train_df, features, mask = utils.correlation_filter(dataframe=temp_df, 
                                                    main_columns=["target_increasing","target_percentage"], 
                                                    max_columns = max_features,
                                                    threshold=thresh_hold, 
                                                    show_heat_map = show_heat_map)
    X_train = train_df.to_numpy()
    X_valid = valid_df.to_numpy()
    X_test = test_df.to_numpy()

    X_train = utils.prepare_timeseries_data_x(X_train, window_size = window_size)
    X_valid = utils.prepare_timeseries_data_x(X_valid, window_size = window_size)
    X_test = utils.prepare_timeseries_data_x(X_test, window_size = window_size)

    y_train = y_train[window_size:]
    y_valid = y_valid[window_size:]
    y_test = y_test[window_size:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_valid, y_valid)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)

    if is_train:
        train.train_Movement_3(dataset_train_trend, dataset_val_trend, features, mask)
    infer.evalute_Movement_3(dataset_val=dataset_val_trend, features = features)
    infer.evalute_Movement_3(dataset_val=dataset_test_trend, features = features)

def train_movement_7(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):
    
    window_size = cf["model"]["movement_7"]["window_size"]
    max_features = cf["model"]["movement_7"]["max_features"]
    thresh_hold = cf["training"]["movement_7"]["corr_thresh_hold"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    train_close = train_close_df.to_numpy()
    valid_close = valid_close_df.to_numpy()
    test_close = test_close_df.to_numpy()

    train_n_row = len(train_close) - window_size
    valid_n_row = len(valid_close) - window_size
    test_n_row = len(test_close) - window_size

    # calculate y
    y_train = utils.prepare_timeseries_data_y_trend_percentage(train_n_row, train_close, output_size = 7)
    y_valid = utils.prepare_timeseries_data_y_trend_percentage(valid_n_row, valid_close, output_size = 7)
    y_test = utils.prepare_timeseries_data_y_trend_percentage(test_n_row, test_close, output_size = 7)

    # Re allocated df with lossing data points
    train_df = train_df[window_size:]
    valid_df = valid_df[window_size:]
    test_df = test_df[window_size:]
    train_close_df = train_close_df[window_size:]
    valid_close_df = valid_close_df[window_size:]
    test_close_df = test_close_df[window_size:]

    # copy dataframe
    temp_df = train_df.copy()
    # temp_df["target_trend_down"] = y_trend_percentage_3[:, :1]
    temp_df["target_increasing"] = y_train[:, 1:2]
    temp_df["target_percentage"] = y_train[:, 2:]
    train_df, features, mask = utils.correlation_filter(dataframe=temp_df, 
                                                    main_columns=["target_increasing","target_percentage"], 
                                                    max_columns = max_features,
                                                    threshold=thresh_hold, 
                                                    show_heat_map = show_heat_map)
    X_train = train_df.to_numpy()
    X_valid = valid_df.to_numpy()
    X_test = test_df.to_numpy()

    X_train = utils.prepare_timeseries_data_x(X_train, window_size = window_size)
    X_valid = utils.prepare_timeseries_data_x(X_valid, window_size = window_size)
    X_test = utils.prepare_timeseries_data_x(X_test, window_size = window_size)

    y_train = y_train[window_size:]
    y_valid = y_valid[window_size:]
    y_test = y_test[window_size:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_valid, y_valid)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)

    if is_train:
        train.train_Movement_7(dataset_train_trend, dataset_val_trend, features, mask)
    infer.evalute_Movement_7(dataset_val=dataset_val_trend, features = features)
    infer.evalute_Movement_7(dataset_val=dataset_test_trend, features = features)


def train_movement_14(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = False):
    
    window_size = cf["model"]["movement_14"]["window_size"]
    max_features = cf["model"]["movement_14"]["max_features"]
    thresh_hold = cf["training"]["movement_14"]["corr_thresh_hold"]

    train_df = utils.prepare_dataset_and_indicators(train_df, window_size)
    valid_df = utils.prepare_dataset_and_indicators(valid_df, window_size)
    test_df = utils.prepare_dataset_and_indicators(test_df, window_size)
    # prepare y df
    train_close_df = pd.DataFrame({'close': train_df['close']})
    valid_close_df = pd.DataFrame({'close': valid_df['close']})
    test_close_df = pd.DataFrame({'close': test_df['close']})

    train_close = train_close_df.to_numpy()
    valid_close = valid_close_df.to_numpy()
    test_close = test_close_df.to_numpy()

    train_n_row = len(train_close) - window_size
    valid_n_row = len(valid_close) - window_size
    test_n_row = len(test_close) - window_size

    # calculate y
    y_train = utils.prepare_timeseries_data_y_trend_percentage(train_n_row, train_close, output_size = 14)
    y_valid = utils.prepare_timeseries_data_y_trend_percentage(valid_n_row, valid_close, output_size = 14)
    y_test = utils.prepare_timeseries_data_y_trend_percentage(test_n_row, test_close, output_size = 14)

    # Re allocated df with lossing data points
    train_df = train_df[window_size:]
    valid_df = valid_df[window_size:]
    test_df = test_df[window_size:]
    train_close_df = train_close_df[window_size:]
    valid_close_df = valid_close_df[window_size:]
    test_close_df = test_close_df[window_size:]

    # copy dataframe
    temp_df = train_df.copy()
    # temp_df["target_trend_down"] = y_trend_percentage_3[:, :1]
    temp_df["target_increasing"] = y_train[:, 1:2]
    temp_df["target_percentage"] = y_train[:, 2:]
    train_df, features, mask = utils.correlation_filter(dataframe=temp_df, 
                                                    main_columns=["target_increasing","target_percentage"], 
                                                    max_columns = max_features,
                                                    threshold=thresh_hold, 
                                                    show_heat_map = show_heat_map)
    X_train = train_df.to_numpy()
    X_valid = valid_df.to_numpy()
    X_test = test_df.to_numpy()

    X_train = utils.prepare_timeseries_data_x(X_train, window_size = window_size)
    X_valid = utils.prepare_timeseries_data_x(X_valid, window_size = window_size)
    X_test = utils.prepare_timeseries_data_x(X_test, window_size = window_size)

    y_train = y_train[window_size:]
    y_valid = y_valid[window_size:]
    y_test = y_test[window_size:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_valid, y_valid)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)

    if is_train:
        train.train_Movement_14(dataset_train_trend, dataset_val_trend, features, mask)
    infer.evalute_Movement_14(dataset_val=dataset_val_trend, features = features)
    infer.evalute_Movement_14(dataset_val=dataset_test_trend, features = features)
    
if __name__ == "__main__":
    data_df, num_data_points, data_dates = utils.download_data_api()
    # data_df, num_data_points, data_dates = utils.get_new_df(data_df, '2018-01-01')
    data_df.set_index('date', inplace=True)
    train_df, valid_df, test_df, train_date, valid_date, test_date = utils.split_train_valid_test_dataframe(data_df, num_data_points, data_dates)
    # data_df = utils.get_new_df(data_df, '2018-01-01')
    # train_random_tree_classifier_14(data_df, num_data_points, data_date)

    train_movement_3(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = True)
    train_movement_7(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = True)
    train_movement_14(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = True)
    train_diff_1(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = True)
    train_assemble(data_df, 
                    num_data_points,
                    train_df, valid_df,
                    test_df, train_date,valid_date, test_date,
                    data_dates, show_heat_map = False, is_train = True)