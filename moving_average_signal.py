import os 
import tensorflow as tf
import datetime
import util
from util import *
import numpy as np
import pandas as pd
from config import config as cf
import pandas_ta as ta

# window_size = [10,20,30,50,100,200] for MA
class Signal:
    def __init__(self):
        pass
    def get_start_date(self, date_length=200, end_date=None):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_day = end_date - datetime.timedelta(days=date_length)
        start_day = start_day.strftime(end_date, '%Y-%m-%d')
        return start_day
    def get_stock_dataframe(self, date_length=200):
        symbol = cf['alpha_vantage']['symbol']
        api_key = cf["alpha_vantage"]["api_key"]
        filename = symbol + '.csv'
        path = './technical_signal/'
        df = download_stock_csv_file(path, filename, symbol, date_length)
        df = df.iloc[-200:]
        df = pd.DataFrame({
            '4. close': df['close'],
            '1. open': df['open'],
            '2. high': df['high'],
            '3. low': df['low'],
            '6. volume': df['volume']
        })
        # Moving Average
        for i in [10, 20, 30, 50, 100, 200]:
            sma = SMA(df, i)
            ema = EMA(df, i)
            df = pd.concat([df, sma], axis=1)
            df = pd.concat([df, ema], axis=1)
        hma = HMA(df, 9)
        df = pd.concat([df, hma], axis=1)

        # Oscillators
        # for i in [12, 26]:
        #     macd = MACD(df, i)
        #     df = pd.concat([df, macd], axis=1)
        macd = MACD(df, 12)
        rsi = RSI(df, 14)
        stochrsi = STOCHRSI(df, window_size=14)
        cci = CCI(df, 20)
        willr = WILLR(df, 14)
        mom = MOM(df, 10)
        eri = ERI(df, 14)
        bbands = BBANDS(df, 5)
        df = pd.concat([df, rsi], axis=1)
        df = pd.concat([df, macd], axis=1)
        df = pd.concat([df, stochrsi], axis=1)
        df = pd.concat([df, willr], axis=1)
        df = pd.concat([df, cci], axis=1)
        df = pd.concat([df, mom], axis=1)
        df = pd.concat([df, eri], axis=1)
        df = pd.concat([df, bbands], axis=1)
        df.to_csv(f"{path}{filename}")
        return df
    def ema_signal(self, df, short, long):
        ema_fast = df['EMA_' + str(short)]
        ema_slow = df['EMA_' + str(long)]
        signal = []
        for i in range(1, len(df)):
            if ema_fast.iloc[i] is not None and ema_slow.iloc[i] is not None :
                if ema_fast.iloc[i] > ema_slow.iloc[i] and ema_fast.iloc[i-1] <= ema_slow.iloc[i-1]:
                    signal[i] = 1
                elif ema_fast.iloc[i] < ema_slow.iloc[i] and ema_fast.iloc[i-1] >= ema_slow.iloc[i-1]:
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal
    
    def smi_signal(self, df, period):
        signal = []
        for i in range(1, len(df)):
            if df['SMA_' + str(period)].iloc[i] is not None:
                if df['SMA_' + str(period)].iloc[i] > df['4. close'].iloc[i]:
                    signal[i] = -1
                elif df['SMA_' + str(period)].loc[i] < df['4. close'].iloc[i]:
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal
    
    def hma_signal(self, df, period=9):
        signal = []
        for i in range(1, len(df)):
            if df['HMA_' + str(period)][i] is not None:
                if df['HMA_' + str(period)][i] > df['Close'][i] and df['HMA_' + str(period)][i-1] <= df['Close'][i-1]:
                    signal[i] = 1
                elif df['HMA_' + str(period)][i] < df['Close'][i] and df['HMA_' + str(period)][i-1] >= df['Close'][i-1]:
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal
    
    def macd_signal(self, df):
        label_macd = 'MACD_12_26_9'
        label_macds = 'MACDs_12_26_9'
        signal = []
        for i in range(1, len(df)):
            if df[label_macd] is not None and df[label_macds] is not None:
                if df[label_macd][i] > df[label_macds][i] and df[label_macd][i-1] <= df[label_macds][i-1]:
                    signal[i] = 1
                elif df[label_macd][i] < df[label_macds][i] and df[label_macd][i-1] >= df[label_macds][i-1]:
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal
        
    def rsi_signal(self, df):
        signal = []
        label = 'RSI_14'
        
    
    def define_signal(self, path, filename, new_data):
        if new_data:
            df = self.get_stock_dataframe(date_length=200)
        else:
            if not file_exist(path, file_name=filename):
                df = self.get_stock_dataframe(date_length=200)
            else:
                df = read_csv_file(path, filename)
        
        
    
    
        









