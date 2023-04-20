import os
from config import config as cf
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import pandas as pd
import numpy as np
import pandas_ta as ta


def file_exist(path, file_name):
    filepath = os.path.join(path, file_name)
    return os.path.isfile(filepath)


def download_stock_csv_file(path, file_name, symbol, window_size):
    api_key = cf["alpha_vantage"]["api_key"]
    ti = TechIndicators(key=api_key)

    daily_adjusted_df = daily_dataframe(api_key, symbol)

    willr = WILLR(daily_adjusted_df, window_size)
    smi = SMI(daily_adjusted_df, window_size)
    stochrsi = STOCHRSI(daily_adjusted_df, window_size)
    cci = CCI(daily_adjusted_df, window_size)
    macd = MACD(daily_adjusted_df, window_size)
    dm = DM(daily_adjusted_df, window_size)
    cfo = CFO(daily_adjusted_df, window_size)
    cmo = CMO(daily_adjusted_df, window_size)
    er = ER(daily_adjusted_df, window_size)
    mom = MOM(daily_adjusted_df, window_size)
    roc = ROC(daily_adjusted_df, window_size)
    stc = STC(daily_adjusted_df, window_size)
    slope = SLOPE(daily_adjusted_df, window_size)
    eri = ERI(daily_adjusted_df, window_size)
    bbands = BBANDS(daily_adjusted_df, window_size)
    sma = SMA(daily_adjusted_df, window_size)
    ema = EMA(daily_adjusted_df, window_size)
    vwap = VWAP(daily_adjusted_df, window_size)
    hma = HMA(daily_adjusted_df, window_size)
    cmf = CMF(daily_adjusted_df, window_size)

    daily_adjusted_df = pd.DataFrame({
        'close': daily_adjusted_df['4. close'],
        'open': daily_adjusted_df['1. open'],
        'high': daily_adjusted_df['2. high'],
        'low': daily_adjusted_df['3. low'],
        'adjusted close': daily_adjusted_df['5. adjusted close'],
        'volume': daily_adjusted_df['6. volume']
    })

    final_df = pd.concat([daily_adjusted_df, willr], axis=1)
    final_df = pd.concat([final_df, smi], axis=1)
    final_df = pd.concat([final_df, stochrsi], axis=1)
    final_df = pd.concat([final_df, cci], axis=1)
    final_df = pd.concat([final_df, macd], axis=1)
    final_df = pd.concat([final_df, dm], axis=1)
    final_df = pd.concat([final_df, cfo], axis=1)
    final_df = pd.concat([final_df, cmo], axis=1)
    final_df = pd.concat([final_df, er], axis=1)
    final_df = pd.concat([final_df, mom], axis=1)
    final_df = pd.concat([final_df, roc], axis=1)
    final_df = pd.concat([final_df, stc], axis=1)
    final_df = pd.concat([final_df, slope], axis=1)
    final_df = pd.concat([final_df, eri], axis=1)
    final_df = pd.concat([final_df, bbands], axis=1)
    final_df = pd.concat([final_df, sma], axis=1)
    final_df = pd.concat([final_df, ema], axis=1)
    final_df = pd.concat([final_df, vwap], axis=1)
    final_df = pd.concat([final_df, hma], axis=1)
    final_df = pd.concat([final_df, cmf], axis=1)

    # Save the data to a CSV file
    final_df.to_csv(f"{path}{file_name}")
    return final_df


def api_builder(function, url, symbol, api_key, datatype):
    query = "query?function=" + function + "&symbol=" + symbol + "&apikey=" + api_key + "datatype=" + datatype
    api = url + "/" + query
    return api


def read_csv_file(path, file_name):
    # construct the full file path
    file_path = f"{path}/{file_name}"

    # read the CSV file into a Pandas dataframe
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df


"""
Check csv file with symbol and function name.
If not exist download and read csv file and return dataframe
If exist read csv file and return dataframe
"""


def prepare_stock_dataframe(window_size, start, end, new_data):
    symbol = cf["alpha_vantage"]["symbol"]
    file_name = symbol + ".csv"
    path = "./csv/"
    if not file_exist(path, file_name):
        df = download_stock_csv_file(path, file_name, symbol, window_size)

    elif new_data:
        df = download_stock_csv_file(path, file_name, symbol, window_size)
    else:
        df = read_csv_file(path, file_name)

    if start is not None and end is not None:
        df = df.loc[start:end]
    elif start is not None:
        df = df.loc[start:]
    elif end is not None:
        df = df.loc[:end]
    df = df.dropna()
    return df


def prepare_timeseries_data(x, window_size, output_step, stride = 1):
    """
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
    """
    x = np.array(x)
    num_features = x.shape[-1]
    n_row = (len(x) - window_size + output_step) // stride + 1
    X = np.zeros((n_row, window_size, num_features))
    y = []
    for i in range(0, n_row):
        for j in range(0, window_size):
            X[i][j] = x[i * stride + j]
            if j == (window_size - 1):
                if X[i][j][0] < x[i * stride + j + output_step][0]:
                    y.append(1)
                else:
                    y.append(0)
    y = np.array(y)
    y = y.reshape(-1, 1)
    # return (all the element but the last one, return the last element)
    return X, y


def prepare_timeseries_data_y(num_rows, data, window_size, output_size):
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
        output[i] = data[window_size + i:window_size + i + output_size]
    return output


def prepare_data_y(num_rows, data, output_size):
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
        output[i] = data[i:i + output_size]
    return output


def prepare_data_y_trend(data, window_size, output_size, stride = 1):
    n_row = (len(data) - window_size + output_size) // stride + 1   
    output = []
    # Iterate over original array and extract windows of size 3
    # (1) means up
    # (0) means down
    for i in range(0, len(data) - n_row, stride):
        # Go up
        if data[i + output_size + (window_size - 1)] > data[i + (window_size - 1)]:
            output.append(1)
        # Go down
        else:
            output.append(0)
    return output


def prepare_timeseries_data_y_percentage(num_rows, data, window_size, output_size, stride = 1):
    output = np.zeros((num_rows, 1), dtype=float)
    for i in range(num_rows):
        change_percentage = ((data[i + window_size + output_size - 1] - data[window_size + i - 1]) * 100) / data[window_size + i - 1]
        if change_percentage > 2.5:
            output[i] = (1)
    return output


def daily_dataframe(api_key, symbol):
    ts = TimeSeries(key=api_key)
    data, meta_data = ts.get_daily_adjusted(symbol, outputsize="full")
    df = pd.DataFrame.from_dict(data, orient='index').astype(float)
    df = df.iloc[::-1]
    df.index = pd.to_datetime(df.index)
    return df


def WILLR(df, window_size):
    return ta.willr(df['2. high'], df['3. low'], df['4. close'], window_size)


def SMI(df, window_size):
    return ta.smi(df['4. close'], fast=window_size - window_size // 2, slow=window_size)


def STOCHRSI(df, window_size):
    if window_size < 14:
        return ta.stochrsi(df['4. close'], window_size, rsi_length=window_size * 2)
    else:
        return ta.stochrsi(df['4. close'], window_size, rsi_length=window_size)


def CCI(df, window_size):
    return ta.cci(df['2. high'], df['3. low'], df['4. close'], window_size)


def MACD(df, window_size):
    return ta.macd(df['4. close'], window_size)


def DM(df, window_size):
    return ta.dm(length=window_size, low=df['3. low'], high=df['2. high'])


def CFO(df, window_size):
    return ta.cfo(df['4. close'], window_size)


def CMO(df, window_size):
    return ta.cmo(df['4. close'], window_size)


def ER(df, window_size):
    return ta.er(df['4. close'], window_size)


def MOM(df, window_size):
    return ta.mom(df['4. close'], window_size)


def ROC(df, window_size):
    return ta.roc(df['4. close'], window_size)


def STC(df, window_size):
    return ta.stc(df['4. close'], window_size, window_size, window_size * 2)


def SLOPE(df, window_size):
    return ta.slope(df['4. close'], window_size)


def ERI(df, window_size):
    return ta.eri(df['2. high'], df['3. low'], df['4. close'], window_size)


def BBANDS(df, window_size):
    return ta.bbands(close=df['4. close'], lenght=window_size, std=2)


def SMA(df, window_size):
    return ta.sma(df['4. close'], window_size)


def EMA(df, window_size):
    return ta.ema(df['4. close'], window_size)


def VWAP(df, window_size):
    # dataframe = df.copy()
    # dataframe.index = pd.to_datetime(dataframe.index)
    return ta.vwap(high=df['2. high'], low=df['3. low'], close=df['4. close'], volume=df['6. volume'])


def HMA(df, window_size):
    """
    https://oxfordstrat.com/trading-strategies/hull-moving-average/
    """
    return ta.hma(df['4. close'], window_size)


def CMF(df, window_size):
    return ta.cmf(df['2. high'], df['3. low'], df['4. close'], df['6. volume'], lenght=window_size)
