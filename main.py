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
data_df, num_data_points, data_date = utils.download_data_api()

data_df = utils.get_new_df(data_df, '2018-01-01')

sma = utils.SMA(data_df['4. close'].values, cf['data']['window_size'])
ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], cf['data']['window_size'])
rsi = utils.RSI(data_df, cf['data']['window_size'])
vwap = utils.VWAP(data_df, cf['data']['window_size'])
hma = utils.HMA(data_df['4. close'], cf['data']['window_size'])
bullish = utils.bullish(data_df['1. open'], data_df['4. close'])

dataset_df = pd.DataFrame({ 'close': data_df['4. close'], 'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'bullish': bullish, 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
dataset_df = dataset_df.drop(dataset_df.index[:15])

X = dataset_df.values
y = np.array(dataset_df['close'])

window_size = cf["data"]["window_size"]
n_row = X.shape[0] - window_size
y_diff = utils.prepare_timeseries_data_y_diff(n_row, y, window_size)
y_trend = utils.prepare_timeseries_data_y_trend(n_row, y, 1)
y_trend_14 = utils.prepare_timeseries_data_y_trend(n_row, y, 14)
y_real = y [14:]
X_set = utils.prepare_timeseries_data_x(X, window_size=window_size)
X_test = X[n_row:]
# split dataset
split_index = int(y_diff.shape[0]*cf["data"]["train_split_size"])
# from 0 - 80%
data_x_train = X_set[:split_index]
# from 80% - 100%
data_x_val = X_set[split_index:]
# from 0 - 80%
data_y_train_diff = y_diff[:split_index]
# from 80% - 100%
data_y_val_diff = y_diff[split_index:]
# from 0 - 80%
data_y_train_trend = y_trend[:split_index]
# from 80% - 100%
data_y_val_trend = y_trend[split_index:]
# from 0 - 80%
data_y_train_trend_14 = y_trend_14[:split_index]
# from 80% - 100%
data_y_val_trend_14 = y_trend[split_index:]
# from 0 - 80%
data_y_train_trend_real = y_real[:split_index]
data_y_train_trend_real =  np.expand_dims(data_y_train_trend_real, axis=1)
# from 80% - 100%
data_y_val_trend_real = y_real[split_index:]
data_y_val_trend_real =  np.expand_dims(data_y_val_trend_real, axis=1)

dataset_train_diff = TimeSeriesDataset(data_x_train, data_y_train_diff)
dataset_val_diff = TimeSeriesDataset(data_x_val, data_y_val_diff)
dataset_train_trend = Classification_TimeSeriesDataset(data_x_train, data_y_train_trend)
dataset_val_trend = Classification_TimeSeriesDataset(data_x_val, data_y_val_trend)
dataset_train_trend14 = Classification_TimeSeriesDataset(data_x_train, data_y_train_trend_14)
dataset_val_trend14 = Classification_TimeSeriesDataset(data_x_val, data_y_val_trend_14)
dataset_train_real = TimeSeriesDataset(data_x_train, data_y_train_trend_real)
dataset_val_real = TimeSeriesDataset(data_x_val, data_y_val_trend_real)

# lstm_regression = train.train_LSTM_regression(dataset_train_diff, dataset_val_diff)
# lstm_binary = train.train_LSTM_binary_1(dataset_train_trend, dataset_val_trend)
# lstm_binary_14 = train.train_LSTM_binary_14(dataset_train_trend14, dataset_val_trend14)
# assembly_regression = train.train_assembly_model(dataset_train_real, dataset_val_real)
infer.evalute_regression(dataset_val=dataset_val_diff)
infer.evalute_binary1(dataset_val=dataset_val_trend)
infer.evalute_binary14(dataset_val=dataset_val_trend14)
infer.evalute_regression(dataset_val=dataset_val_real)
# y_pred = assembly_regression.predict(X_val[:-1])
# # Evaluate the model's performance using mean squared error
# mape = utils.mean_absolute_percentage_error(y_val, y_pred)
# mse = mean_squared_error(y_val, y_pred)
# print("Mean Absolute Percentage Error: {:.2f}".format(mape))
# print("Mean Squared Error: {:.2f}".format(mse))