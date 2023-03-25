import utils
from config import config as cf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import train
from sklearn.metrics import f1_score
import model
from sklearn.metrics import mean_squared_error
from dataset import TimeSeriesDataset, Classification_TimeSeriesDataset
import infer
# data_df, num_data_points, data_date = utils.download_data_api()

# # data_df = utils.get_new_df(data_df, '2018-01-01')

# sma = utils.SMA(data_df['4. close'].values, cf['data']['window_size'])
# ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], cf['data']['window_size'])
# rsi = utils.RSI(data_df, cf['data']['window_size'])
# vwap = utils.VWAP(data_df, cf['data']['window_size'])
# hma = utils.HMA(data_df['4. close'], cf['data']['window_size'])
# bullish = utils.bullish(data_df['1. open'], data_df['4. close'])

# dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'bullish': bullish, 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
# y_df = pd.DataFrame({'close': data_df['4. close']})
# # dataset_df = pd.DataFrame({'close': data_df['4. close'], 'bullish': bullish})
# dataset_df = dataset_df.drop(dataset_df.index[:15])

# X = dataset_df.values
# # y = np.array(dataset_df['close'])
# y = np.array(y_df['close'])
# window_size = cf["data"]["window_size"]
# n_row = X.shape[0] - window_size
# y_diff = utils.prepare_timeseries_data_y_diff(n_row, y, window_size)
# # y_trend = utils.prepare_timeseries_data_y_trend(n_row, y, 1)
# # y_trend_14 = utils.prepare_timeseries_data_y_trend(n_row, y, 14)
# y_trend = utils.prepare_timeseries_data_y_trend_percentage(n_row, y, 1)
# y_trend_14 = utils.prepare_timeseries_data_y_trend_percentage(n_row, y, 14)
# y_real = y [14:]
# X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
# X_test = X[n_row:]
# # split dataset
# split_index = int(y_diff.shape[0]*cf["data"]["train_split_size"])

# tree_data_X_train = X[:split_index]
# tree_data_X_val = X[split_index:]
# # from 0 - 80%
# data_x_train = X_set[:split_index]
# # from 80% - 100%
# data_x_val = X_set[split_index:]
# # from 0 - 80%
# data_y_train_diff = y_diff[:split_index]
# # from 80% - 100%
# data_y_val_diff = y_diff[split_index:]
# # from 0 - 80%
# data_y_train_trend = y_trend[:split_index]
# # from 80% - 100%
# data_y_val_trend = y_trend[split_index:]
# # from 0 - 80%
# data_y_train_trend_14 = y_trend_14[:split_index]
# # from 80% - 100%
# data_y_val_trend_14 = y_trend[split_index:]
# # from 0 - 80%
# data_y_train_trend_real = y_real[:split_index]
# data_y_train_trend_real =  np.expand_dims(data_y_train_trend_real, axis=1)
# # from 80% - 100%
# data_y_val_trend_real = y_real[split_index:]
# data_y_val_trend_real =  np.expand_dims(data_y_val_trend_real, axis=1)

# dataset_train_diff = TimeSeriesDataset(data_x_train, data_y_train_diff)
# dataset_val_diff = TimeSeriesDataset(data_x_val, data_y_val_diff)
# dataset_train_trend = Classification_TimeSeriesDataset(data_x_train, data_y_train_trend)
# dataset_val_trend = Classification_TimeSeriesDataset(data_x_val, data_y_val_trend)
# dataset_train_trend14 = Classification_TimeSeriesDataset(data_x_train, data_y_train_trend_14)
# dataset_val_trend14 = Classification_TimeSeriesDataset(data_x_val, data_y_val_trend_14)
# dataset_train_real = TimeSeriesDataset(data_x_train, data_y_train_trend_real)
# dataset_val_real = TimeSeriesDataset(data_x_val, data_y_val_trend_real)

# lstm_regression = train.train_LSTM_regression(dataset_train_diff, dataset_val_diff)
# lstm_binary = train.train_LSTM_binary_1(dataset_train_trend, dataset_val_trend)
# lstm_binary_14 = train.train_LSTM_binary_14(dataset_train_trend14, dataset_val_trend14)
# assembly_regression = train.train_assembly_model(dataset_train_real, dataset_val_real)

# infer.evalute_regression(dataset_val=dataset_val_diff)
# infer.evalute_binary1(dataset_val=dataset_val_trend)
# infer.evalute_binary14(dataset_val=dataset_val_trend14)
# infer.evalute_regression(dataset_val=dataset_val_real)
# y_pred = assembly_regression.predict(X_val[:-1])
# # Evaluate the model's performance using mean squared error
# mape = utils.mean_absolute_percentage_error(y_val, y_pred)
# mse = mean_squared_error(y_val, y_pred)
# print("Mean Absolute Percentage Error: {:.2f}".format(mape))
# print("Mean Squared Error: {:.2f}".format(mse))

def train_random_tree_classifier_14(data_df, num_data_points, data_date):
    # data_df = utils.get_new_df(data_df, '2018-01-01')

    sma = utils.SMA(data_df['4. close'].values, cf['data']['window_size'])
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], cf['data']['window_size'])
    rsi = utils.RSI(data_df, cf['data']['window_size'])
    vwap = utils.VWAP(data_df, cf['data']['window_size'])
    hma = utils.HMA(data_df['4. close'], cf['data']['window_size'])
    bullish = utils.bullish(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'],'bullish': bullish, 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
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
    
def train_lstm_classifier_14(data_df, num_data_points, data_date, is_train):
    window_size = cf["data"]["window_size"]
    sma = utils.SMA(data_df['4. close'].values, window_size)
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], window_size)
    rsi = utils.RSI(data_df, window_size)
    vwap = utils.VWAP(data_df, window_size)
    hma = utils.HMA(data_df['4. close'], window_size)
    bullish = utils.bullish(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'bullish': bullish, 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
    dataset_df = dataset_df[15:]
    X = dataset_df.to_numpy()

    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_trend_14 = utils.prepare_timeseries_data_y_trend(n_row, close, 14)
    X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
    split_index = int(y_trend_14.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_trend_14[:split_index]
    y_test = y_trend_14[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_val, y_val)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_LSTM_classifier_14(dataset_train_trend, dataset_val_trend)
    infer.evalute_classifier_14(dataset_val=dataset_val_trend)
    infer.evalute_classifier_14(dataset_val=dataset_test_trend)

def train_lstm_classifier_1(data_df, num_data_points, data_date, is_train):
    window_size = cf["data"]["window_size"]
    sma = utils.SMA(data_df['4. close'].values, window_size)
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], window_size)
    rsi = utils.RSI(data_df, window_size)
    vwap = utils.VWAP(data_df, window_size)
    hma = utils.HMA(data_df['4. close'], window_size)
    bullish = utils.bullish(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'bullish': bullish, 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
    dataset_df = dataset_df[15:]
    X = dataset_df.to_numpy()

    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_trend_1 = utils.prepare_timeseries_data_y_trend(n_row, close, 1)
    X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
    split_index = int(y_trend_1.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_trend_1[:split_index]
    y_test = y_trend_1[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]

    dataset_train_trend = Classification_TimeSeriesDataset(X_train, y_train)
    dataset_val_trend = Classification_TimeSeriesDataset(X_val, y_val)
    dataset_test_trend = Classification_TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_LSTM_classifier_1(dataset_train_trend, dataset_val_trend)
    infer.evalute_classifier_1(dataset_val=dataset_val_trend)
    infer.evalute_classifier_1(dataset_val=dataset_test_trend)

def train_assemble(data_df, num_data_points, data_date, is_train):
    window_size = cf["data"]["window_size"]
    sma = utils.SMA(data_df['4. close'].values, window_size)
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], window_size)
    rsi = utils.RSI(data_df, window_size)
    vwap = utils.VWAP(data_df, window_size)
    hma = utils.HMA(data_df['4. close'], window_size)
    bullish = utils.bullish(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'bullish': bullish, 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
    dataset_df = dataset_df[15:]
    X = dataset_df.to_numpy()

    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_real_1 = utils.prepare_timeseries_data_y(n_row, close, window_size, 1)
    X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
    split_index = int(y_real_1.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_real_1[:split_index]
    y_test = y_real_1[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]

    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_val, y_val)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_assemble_model(dataset_train, dataset_val)
    infer.evalute_regression(dataset_val=dataset_val)
    infer.evalute_regression(dataset_val=dataset_test)


def train_lstm_regressor_1(data_df, num_data_points, data_date, is_train):
    window_size = cf["data"]["window_size"]
    sma = utils.SMA(data_df['4. close'].values, window_size)
    ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], window_size)
    rsi = utils.RSI(data_df, window_size)
    vwap = utils.VWAP(data_df, window_size)
    hma = utils.HMA(data_df['4. close'], window_size)
    bullish = utils.bullish(data_df['1. open'], data_df['4. close'])

    dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'bullish': bullish, 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
    dataset_df = dataset_df[15:]
    X = dataset_df.to_numpy()

    n_row = X.shape[0] - window_size

    close_df = pd.DataFrame({'close': dataset_df['close']})
    close = close_df.to_numpy()
    y_diff_1 = utils.prepare_timeseries_data_y_diff(n_row, close, window_size)
    X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
    split_index = int(y_diff_1.shape[0]*cf["data"]["train_split_size"])

    X_train_first = X_set[:split_index]
    X_test = X_set[split_index:]
    y_train_first = y_diff_1[:split_index]
    y_test = y_diff_1[split_index:]
    split_index = int(y_train_first.shape[0]*cf["data"]["train_split_size"])
    X_train = X_train_first[:split_index]
    X_val = X_train_first[split_index:]
    y_train = y_train_first[:split_index]
    y_val = y_train_first[split_index:]

    dataset_train = TimeSeriesDataset(X_train, y_train)
    dataset_val = TimeSeriesDataset(X_val, y_val)
    dataset_test = TimeSeriesDataset(X_test, y_test)
    if is_train:
        train.train_LSTM_regression(dataset_train, dataset_val)
    infer.evalute_regression(dataset_val=dataset_val)
    infer.evalute_regression(dataset_val=dataset_test)

if __name__ == "__main__":
    data_df, num_data_points, data_date = utils.download_data_api()
    # train_random_tree_classifier_14(data_df, num_data_points, data_date)
    train_lstm_classifier_14(data_df, num_data_points, data_date, is_train = False)
    train_lstm_classifier_1(data_df, num_data_points, data_date, is_train = False)
    train_lstm_regressor_1(data_df, num_data_points, data_date, is_train = False)
    train_assemble(data_df, num_data_points, data_date, is_train = False)