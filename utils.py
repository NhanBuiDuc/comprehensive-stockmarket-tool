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
  
    
