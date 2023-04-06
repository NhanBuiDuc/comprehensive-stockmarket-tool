import datetime
from alpha_vantage.timeseries import TimeSeries 
from config import config as cf
import pandas as pd
import numpy as np
import pandas_ta as ta
import seaborn as sns
import matplotlib.pyplot as plt
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
def prepare_timeseries_data_y_trend_percentage(num_rows, data, output_size):
    output = np.empty((num_rows, 3))
    # Iterate over original array and extract windows of size 3
    # (0,1,p) means up
    # (1,0,p) means down
    for i in range(num_rows - 1):
        change_percentage =  (( data[i + output_size] - data[i] ) * 100 ) / data[i]
        # Go up
        if((change_percentage > 0)):
            output[i] = (0, 1, abs(change_percentage))
        # Go down
        elif ((change_percentage < 0)):
            output[i] = (1, 0, abs(change_percentage))
    return output
def prepare_tree_data_y_trend(num_rows, data, output_size):
    output = np.empty((num_rows, 1), dtype=int)
    # Iterate over original array and extract windows of size 3
    # (1) means up
    # (0) means down
    for i in range(num_rows - 1):
        # Go up
        if(data[i + output_size] > data[i]):
            output[i] = 1
        # Go down
        else:
            output[i] = 0
    return output
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_filter(dataframe, main_columns, max_columns, threshold=0.5):
    correlated_columns = []
    for column in main_columns:
        corr_matrix = dataframe.corr(method='spearman')
        corr_values = corr_matrix[column].abs().sort_values(ascending=False)
        correlated_columns += [col for col in corr_values[corr_values >= threshold].index.tolist() if col not in main_columns]
    result_df = dataframe[list(set(correlated_columns))[:max_columns]]
    result_df = pd.concat([result_df, dataframe[main_columns]], axis=1)
    
    # create a correlation table for the selected columns
    corr_table = pd.DataFrame(index=result_df.columns, columns=result_df.columns)
    for col1 in corr_table.columns:
        for col2 in corr_table.index:
            corr_value = result_df[[col1, col2]].corr(method='spearman').iloc[0, 1]
            corr_table.loc[col1, col2] = corr_value
    
    # replace missing values with 0
    corr_table = corr_table.replace(np.nan, 0)
    
    # plot the correlation table for the selected columns using heatmap
    sns.set(font_scale=1)
    plt.figure(figsize=(19,19))
    sns.heatmap(corr_table.astype(float), annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
    plt.title('Correlation Heatmap for Selected Columns')
    plt.show()
    
    # create a correlation table for all columns in the original dataframe
    corr_all = dataframe.corr(method='spearman')
    corr_all = corr_all.replace(np.nan, 0)
    
    # plot the correlation table for all columns using heatmap
    corr_filtered = corr_all.loc[result_df.columns, result_df.columns]
    plt.figure(figsize=(19,19))
    sns.heatmap(corr_filtered.astype(float), annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True)
    plt.title('Correlation Heatmap for Removed Columns in the Original DataFrame')
    plt.show()
    
    return result_df, result_df.shape[1]



def Willr(df, window_size):
    return ta.willr(df['2. high'], df['3. low'], df['4. close'], window_size)
def Smi(df, window_size):
    return ta.smi(df['4. close'], fast=window_size - window_size//2, slow=window_size)
def Stochrsi(df, window_size):
    if ( window_size < 14):
        return ta.stochrsi(df['4. close'], window_size, rsi_length= window_size * 2 )
    else:
        return ta.stochrsi(df['4. close'], window_size, rsi_length= window_size )
def Cci(df, window_size):
    return ta.cci(df['2. high'], df['3. low'], df['4. close'], window_size)
def Macd(df, window_size):
    return ta.macd(df['4. close'], fast=window_size, slow=window_size * 2)
def Dm(df, window_size):
    return ta.dm(df['2. high'], df['3. low'], window_size)
def Cfo(df, window_size):
    return ta.cfo(df['4. close'], window_size)
def Cmo(df, window_size):
    return ta.cmo(df['4. close'], window_size)
def Er(df, window_size):
    return ta.er(df['4. close'], window_size)
def Mom(df, window_size):
    return ta.mom(df['4. close'], window_size)
def Roc(df, window_size):
    return ta.roc(df['4. close'], window_size)
def Stc(df, window_size):
    return ta.stc(df['4. close'], window_size, window_size, window_size * 2)
def Slope(df, window_size):
    return ta.slope(df['4. close'], window_size)
def Eri(df, window_size):
    return ta.eri(df['2. high'], df['3. low'], df['4. close'], window_size)
def Bbands(df, window_size):
    return ta.bbands(close = df['4. close'], lenght = window_size, std = 2)
def Sma(df, window_size):
    return ta.sma(df['4. close'], window_size)
def Ema(df, window_size):
    return ta.ema(df['4. close'], window_size)
def Vwap(df, window_size):
    close = np.array(df['4. close'])
    vol = np.array(df['6. volume'])
    cum_price_volume = np.cumsum(close * vol)
    cum_volume = np.cumsum(vol)
    vwap = cum_price_volume[window_size-1:] / cum_volume[window_size-1:]
    vwap = vwap.tolist()
    vwap = [float('nan')]*(window_size-1) + vwap
    return vwap
def Hma(df, window_size):

    '''
    https://oxfordstrat.com/trading-strategies/hull-moving-average/
    '''
    return ta.hma(df['4. close'], window_size)
def Cmf(df, window_size):
    return ta.cmf(df['2. high'], df['3. low'], df['4. close'], df['6. volume'], lenght = window_size)
def MACD(df):
    '''
    Moving Average Convergence Divergence (MACD)
    The MACD is a popular indicator to that is used to identify a security's trend.
    While APO and MACD are the same calculation, MACD also returns two more series
    called Signal and Histogram. The Signal is an EMA of MACD and the Histogram is
    the difference of MACD and Signal.
    This STI is calculated by taking the difference of the EA of 26 days from the EA of 12 days.
    Args:
        close (pd.Series): Series of 'close's
        fast (int): The short period. Default: 12
        slow (int): The long period. Default: 26
        signal (int): The signal period. Default: 9
    Returns:
        pd.DataFrame: macd, histogram, signal columns.
    '''
    return ta.macd(close=df['4. close'])
def BBANDS(df, window_size):
    return ta.bbands(df, window_size)
def SO(df):
    return ta.stoch(df['2. high'], df['3. low'], df['4. close'], df['6. volume'])


def CMO(df, window_size):
    '''
    Chande Momentum Oscillator
    The CMO is another well-known STI that employs momentum to locate 
    the relative behavior of stock by demonstrating its strengths and weaknesses in a particular period 
    CMO = 100 * ((Su - Sd)/ ( Su + Sd ) )
    https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo#:~:text=The%20CMO%20indicator%20is%20created,%2D100%20to%20%2B100%20range.
    '''
    return ta.cmo(df, window_size)
def ROC(df, window_size):
    '''
    The Price Rate of Change
    ROC is the percentage change between the current price with respect to an earlier closing price n periods ago.
    ROC = [(Today’s Closing Price – Closing Price n periods ago) / Closing Price n periods ago] x 100
    no need to multiply by 100
    '''
    return ta.roc(df, window_size)
def DMI(df, window_size):
    '''
    Directional Movement Index
    This STI identifies the direction of price movement by comparing prior highs and lows.
    The directional movement index (DMI) is a technical indicator that measures both 
    the strength and direction of a price movement and is intended to reduce false signals.
    window_size typically = 14 days ----> remember to tranfer dataframe to datatime object
    Note: import pandas_ta 
    func return as dataframe have 3 cols: adx, dmp, dmn we take adx col only
    '''
    dmi = ta.trend.adx(df['2. high'], df['3. low'], df['4. close'], length= window_size)
    return dmi
def CCI(df, window_size):
    '''
    Commodity Channel Index
    CCI measures the difference between a security's price change and its average price change.
    window_size typically = 20 days
    '''
    cci = ta.momentum.cci(high=df['2. high'], low=df['3. low'], close=df['4. close'], window=window_size, constant=0.015)
    return cci
def CMF(df, window_size):
    """
    def CCI(df, window_size):
    """
    return ta.cmf(high=df['2. high'], low=df['3. low'], close=df['4. close'], volume=['6. volume'], length=window_size)
def prepare_dataset_and_indicators(data_df, window_size):
    willr = Willr(data_df, window_size)
    smi = Smi(data_df, window_size)
    stochrsi = Stochrsi(data_df, window_size)
    cci = Cci(data_df, window_size)
    macd = Macd(data_df, window_size)
    dm = Dm(data_df, window_size)
    cfo = Cfo(data_df, window_size)
    cmo = Cmo(data_df, window_size)
    er = Er(data_df, window_size)
    mom = Mom(data_df, window_size)
    roc = Roc(data_df, window_size)
    stc = Stc(data_df, window_size)
    slope = Slope(data_df, window_size)
    eri = Eri(data_df, window_size)
    bbands = Bbands(data_df, window_size)
    sma = Sma(data_df, window_size)
    ema = Ema(data_df, window_size)
    vwap = Vwap(data_df, window_size)
    hma = Hma(data_df, window_size)
    cmf = Cmf(data_df, window_size)
    dataset_df = pd.DataFrame({
        'close': data_df['4. close'], 
        'open': data_df['1. open'],
        'high': data_df['2. high'],
        'low': data_df['3. low'],
        'adjusted close': data_df['5. adjusted close'],
        'volume': data_df['6. volume']})
    dataset_df['willr'] =willr
    dataset_df['smi'] =smi.to_numpy()[:, 0]
    dataset_df['SMIs'] =smi.to_numpy()[:, 1]
    dataset_df['SMIo'] =smi.to_numpy()[:, 2]
    dataset_df['STOCHRSIk'] =stochrsi.to_numpy()[:, 0]
    dataset_df['STOCHRSId'] =stochrsi.to_numpy()[:, 1]
    dataset_df['cci'] =cci
    dataset_df['macd'] =macd.to_numpy()[:, 0]
    dataset_df['mach'] =macd.to_numpy()[:, 1]
    dataset_df['macs'] =macd.to_numpy()[:, 2]
    dataset_df['DMp'] =dm.to_numpy()[:, 0]
    dataset_df['DMn'] =dm.to_numpy()[:, 1]
    dataset_df['cfo'] =cfo
    dataset_df['cmo'] =cmo
    dataset_df['er'] =er
    dataset_df['mom'] =mom
    dataset_df['roc'] =roc
    dataset_df['stc'] =stc.to_numpy()[:, 0]
    dataset_df['STCmacd'] =stc.to_numpy()[:, 1]
    dataset_df['STCstoch'] =stc.to_numpy()[:, 2]
    dataset_df['slope'] =slope
    dataset_df['ERIbull'] =eri.to_numpy()[:, 0]
    dataset_df['ERIbear'] =eri.to_numpy()[:, 1]
    dataset_df['BBANDSl'] =bbands.to_numpy()[:, 0]
    dataset_df['BBANDSm'] =bbands.to_numpy()[:, 1]
    dataset_df['BBANDSu'] =bbands.to_numpy()[:, 2]
    dataset_df['BBANDSb'] =bbands.to_numpy()[:, 3]
    dataset_df['BBANDSp'] =bbands.to_numpy()[:, 4]
    dataset_df['sma'] =sma
    dataset_df['ema'] =ema
    dataset_df['vwap'] =vwap
    dataset_df['hma'] =hma
    dataset_df['cmf'] =cmf
    dataset_df = dataset_df.interpolate(method='linear', limit_direction='forward')
    dataset_df = dataset_df.dropna()
    return dataset_df