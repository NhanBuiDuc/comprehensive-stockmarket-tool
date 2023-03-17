import utils
from config import config as cf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import train
from sklearn.metrics import f1_score
import infer
import model
from sklearn.metrics import mean_squared_error
data_df, num_data_points, data_date = utils.download_data_api()

data_df = utils.get_new_df(data_df, '2018-01-01')

sma = utils.SMA(data_df['4. close'].values, cf['data']['window_size'])
ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], cf['data']['window_size'])
rsi = utils.RSI(data_df, cf['data']['window_size'])
vwap = utils.VWAP(data_df, cf['data']['window_size'])
hma = utils.HMA(data_df['4. close'], cf['data']['window_size'])
bullish = utils.bullish(data_df['1. open'], data_df['4. close'])

dataset_df = pd.DataFrame({'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'],  'close': data_df['4. close'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'bullish': bullish, 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
dataset_df = dataset_df.drop(dataset_df.index[:15])

X = dataset_df.values.tolist()
y = np.array(dataset_df['close'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

random_forest_classification_model = train.train_random_forest_classfier(X_train, X_test, X_val, y_train, y_test, y_val)
random_forest_regression_model = train.train_random_forest_regressior(X_train, y_train)
y_test = np.array(y_test)
y_val = np.array(y_val)
y_test = (utils.diff(y_test))
y_val = (utils.diff(y_val))

assembly_regression = model.assembly_regression(random_forest_regression_model, random_forest_classification_model)
y_pred = assembly_regression.predict(X_val[:-1])
# Evaluate the model's performance using mean squared error
mape = utils.mean_absolute_percentage_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
print("Mean Absolute Percentage Error: {:.2f}".format(mape))
print("Mean Squared Error: {:.2f}".format(mse))