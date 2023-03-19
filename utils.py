import datetime
from alpha_vantage.timeseries import TimeSeries 
from config import config as cf
import pandas as pd
import numpy as np


"""
return df: pandas dataframe, num_data_points: int, data_date: list
"""
def download_data_api():
    ts = TimeSeries(key=cf["alpha_vantage"]["key"])
    data, meta_data = ts.get_daily_adjusted(cf["alpha_vantage"]["symbol"], outputsize=cf["alpha_vantage"]["outputsize"])

    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.astype(float)

    # Set the name of the first column (date data)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "date"}, inplace=True)

    # Reset the index of the DataFrame
    df.reset_index(drop=True, inplace=True)
    #df['date'] = df['date'].apply(str_to_datetime)
    #df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.iloc[::-1].reset_index(drop=True)
    data_date = [date for date in df["date"]]
    num_data_points = len(data_date)
    return df, num_data_points, data_date

def get_new_df(df, new_date):
    df['date'] = pd.to_datetime(df['date'])
    new_date = pd.to_datetime(new_date)
    df = df.loc[df['date'] >= new_date]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.drop(columns = ['7. dividend amount', '8. split coefficient'])
    return df

def str_to_datetime(s):
  split = s.split('-')
  year, month, day = int(split[0]), int(split[1]), int(split[2])
  return datetime.datetime(year=year, month=month, day=day)

def prepare_new_shape(x, window_size):
    n_row = x.shape[0] - window_size + 1
    unseen_row = x.shape[0] - window_size - cf['model']['output_dates'] + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row,window_size), strides = (x.strides[0], x.strides[0])) 
    return output[:unseen_row],  output[unseen_row:]

def SMA(x, window_size):
    #output = [sum(row) / len(row) for row in x]
    i = 0
    sma = []
    while i < (len(x) - window_size + 1):
        window = x[i:i+window_size]
        window_avg = np.sum(window)/window_size
        sma.append(window_avg)
        i += 1   
    sma = [float('nan')]*(window_size-1) + sma
    return sma


def EMA(x, smoothing, window_size):
    k = smoothing/(window_size + 1)
    ema = []
    ema.append(x[0])
    i = 1
    while i < (len(x) - window_size + 1):
        window_avg = x[i]*k + ema[i-1]*(1-k)
        ema.append(window_avg)
        i += 1
    ema = [float('nan')]*(window_size-1) + ema
    return ema

def RSI(df, window_size, ema=True):
    delta_close = df['4. close'].diff()
    up = delta_close.clip(lower=0)
    down = -1 * delta_close.clip(upper=0)
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = window_size - 1, adjust=True, min_periods = window_size).mean()
        ma_down = down.ewm(com = window_size - 1, adjust=True, min_periods = window_size).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = window_size, adjust=False).mean()
        ma_down = down.rolling(window = window_size, adjust=False).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi.to_numpy().tolist()

def VWAP(df, window_size):
    close = np.array(df['4. close'])
    vol = np.array(df['6. volume'])
    cum_price_volume = np.cumsum(close * vol)
    cum_volume = np.cumsum(vol)
    vwap = cum_price_volume[window_size-1:] / cum_volume[window_size-1:]
    vwap = vwap.tolist()
    vwap = [float('nan')]*(window_size-1) + vwap
    return vwap

'''
https://oxfordstrat.com/trading-strategies/hull-moving-average/
'''
def WMA(s, window_size):
    wma = s.rolling(window_size).apply(lambda x: ((np.arange(window_size)+1)*x).sum()/(np.arange(window_size)+1).sum(), raw=True)
    return wma

def HMA(s, window_size):
    wma1 = WMA(s, window_size//2)
    wma2 = WMA(s, window_size)
    hma = WMA(wma1.multiply(2).sub(wma2), int(np.sqrt(window_size)))
    return hma.tolist()
  
    
def bullish(open, close):
    # Create a new empty array to hold the results
    comparison_array = []
    open = np.array(open)
    close = np.array(close)
    # Loop through each element in the arrays and compare them
    for i in range(len(open)):
        if open[i] > close[i]:
            comparison_array.append(0)
        else:
            comparison_array.append(1)
    return comparison_array

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the mean absolute percentage error (MAPE) between two arrays.
    
    Parameters:
    y_true (array): array of actual values
    y_pred (array): array of predicted values
    
    Returns:
    mape (float): MAPE between the two arrays
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def diff(np_array):
    diff = np.zeros(len(np_array) - 1)
    for i in range(len(diff)):
        diff[i] = np_array[i] - np_array[i+1]
    return diff
def prepare_timeseries_data_x(x, window_size):
    '''
    x: 1D arr, window_size: int
    Note: len(x) > window_size
    window_size: the size of the sliding window
    n_row: the number of rows in the windowed data. Can take it by func below.
    output return view of x with the shape is (n_row,window_size) and the strides equal to (x.strides[0],x.strides[0])  
    which ensures that the rows of the output are contiguous in memory.
    
    return:
    tuple of 2 array
    output[:-1]: has shape (n_row, window_size)
    output[-1]: has shape (window_size,) and contains the last window of x.
    '''
    x = np.array(x)
    num_features = x.shape[-1]
    n_row = x.shape[0] - window_size
    # Example: window_size = 20, x with shape = (100, 20) -> n_row = 100 - 20 + 1 = 81
    # output shape = (81, 20), strides = (8, 8)
    # -> output will move up one by one until the 100th element(last element) from the original x,
    # each row of output will have 20 elements
    #   x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # output = array([[1, 2, 3, 4],
    #                 [2, 3, 4, 5],
    #                 [3, 4, 5, 6],
    #                 [4, 5, 6, 7],
    #                 [5, 6, 7, 8],
    #                 [6, 7, 8, 9],
    #                 [7, 8, 9, 10]])
    output = np.zeros((n_row, window_size, num_features))
    for i in range(n_row):
        for j in range(window_size):
            output[i][j] = x[i + j]

    #return (all the element but the last one, return the last element)
    return output

def prepare_timeseries_data_y(num_rows, data, window_size):
    # output = x[window_size:]
    output_size = cf["model"]["output_dates"]
    # X has 10 datapoints, y is the label start from the windowsize 3 with output dates of 3
    # Then x will have 6 rows, 4 usable row
    # x: 0, 1, 2 || 1, 2, 3 || 2, 3, 4 || 3, 4, 5 || 4, 5, 6 || 6, 7, 8 || 7, 8, 9  
    # y: 3, 4, 5 || 4, 5, 6 || 5, 6, 7 || 6, 7, 8 || 7, 8, 9

    # X has 10 datapoints, y is the label start from the windowsize 4 with output dates of 3
    # Then x will have 6 rows, 4 usable row
    # x: 0, 1, 2, 3 || 1, 2, 3, 4 || 2, 3, 4, 5 || 3, 4, 5, 6 || 4, 5, 6, 7 || 5, 6, 7, 8 || 6, 7, 8, 9
    # y: 4, 5, 6    || 5, 6, 7    || 6, 7, 8    || 7, 8, 9    || 8, 9
        
    # Create empty array to hold reshaped array
    output = np.empty((num_rows, output_size))
    # Iterate over original array and extract windows of size 3
    for i in range(num_rows):
        output[i] = data[window_size+i:window_size+i+output_size]
    return output

def prepare_timeseries_data_y_diff(num_rows, data, window_size):
    output_size = cf["model"]["lstm_regression"]["output_dates"]
    output = np.empty((num_rows, 1))
    # Iterate over original array and extract windows of size 3
    for i in range(num_rows):
        output[i] = data[i+output_size+window_size - 1] - data[i+window_size - 1]
    return output

def prepare_timeseries_data_y_trend(num_rows, data, output_size):
    output = np.empty((num_rows, 2))
    # Iterate over original array and extract windows of size 3
    # (0,1) means up
    # (1,0) means down
    for i in range(num_rows - 1):
        # Go up
        if(data[i + output_size] > data[i]):
            output[i] = (0, 1)
        # Go down
        else:
            output[i] = (1, 0)
    return output