import utils
from config import config as cf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import train

data_df, num_data_points, data_date = utils.download_data_api()

data_df = utils.get_new_df(data_df, '2018-01-01')

sma = utils.SMA(data_df['4. close'].values, cf['data']['window_size'])
ema = utils.EMA(np.array(data_df['4. close']), cf['data']['smoothing'], cf['data']['window_size'])
rsi = utils.RSI(data_df, cf['data']['window_size'])
vwap = utils.VWAP(data_df, cf['data']['window_size'])
hma = utils.HMA(data_df['4. close'], cf['data']['window_size'])

dataset_df = pd.DataFrame({'open': data_df['1. open'], 'high': data_df['2. high'], 'low': data_df['3. low'],  'close': data_df['4. close'], 'adjusted close': data_df['5. adjusted close'], 'volume': data_df['6. volume'], 'sma' : sma, 'ema' : ema, 'rsi' : rsi, 'vwap' : vwap, 'hma' : hma})
dataset_df = dataset_df.drop(dataset_df.index[:15])
day_steps = cf["model"]["rdfc"]["output_dates"]
y = (dataset_df['close'] > dataset_df['close'].shift(day_steps)).astype(int).fillna(0)
y = y.values.tolist()
X = dataset_df.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest_model, random_forest_acc = train.train_random_forest_classfier(X_train, y_train, X_test, y_test)
print("Accuracy:", random_forest_acc)
print("Weights:", random_forest_model.feature_importances_)