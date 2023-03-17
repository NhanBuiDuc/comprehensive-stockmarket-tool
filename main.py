import utils
from config import config as cf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import train
from sklearn.metrics import f1_score
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

rfc_day_steps = cf["model"]["rdfc"]["output_dates"]
rfc_y = (dataset_df['close'] > dataset_df['close'].shift(1)).astype(int).fillna(0)
rfc_y = rfc_y.values.tolist()
rfc_X = dataset_df.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(rfc_X, rfc_y, test_size=0.2, random_state=42)

# Split the training set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

random_forest_model, random_forest_acc, f1 = train.train_random_forest_classfier(X_train, y_train, X_val, y_val)
print("Accuracy:", random_forest_acc)
# Print the F1 score
print("F1 score: {:.2f}".format(f1))
print("Weights:", random_forest_model.feature_importances_)
