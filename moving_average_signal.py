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
        eri = ERI(df, window_size=13)
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
    
    def sma_signal(self, df, period):
        signal = []
        for i in range(0, len(df)):
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
        # return macds, signal
        label_macd = 'MACD_12_26_9'
        label_macds = 'MACDs_12_26_9'
        signal = []
        for i in range(1, len(df)):
            if df[label_macd][i] is not None and df[label_macds][i] is not None and \
                    df[label_macd][i-1] is not None and df[label_macds][i-1] is not None:
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
        for i in range(0, len(df)):
            if df[label][i] is not None:
                if df[label][i] > 70:
                    signal[i] = -1
                elif df[label][i] < 30:
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def stochrsi_signal(self, df):
        label = 'STOCHRSIk_14_14_3_3'
        signal = []
        for i in range(0, len(df)):
            if df[label][i] is not None:
                if df[label][i] < 20:
                    signal[i] = 1
                elif df[label][i] > 80:
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def willr_signal(self, df):
        signal = []
        label = 'WILLR_14'
        for i in range(0, len(df)):
            if df[label][i] is not None:
                if df[label][i] > -20 & df[label][i] < 0:
                    signal[i] = -1
                elif df[label][i] > -100 & df[label][i] < -80:
                    signal[i] = 1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def mom_signal(self, df):
        label = 'MOM_10'
        signal = np.zeros(len(df))
        for i in range(1, len(df)):
            if df[label][i] is not None and df[label][i-1] is not None:
                if (df[label][i] > 0 and df[label][i-1] <= 0) or \
                        (df[label][i] > 0 and df[label][i-1] > 0 and signal[i-1] == 1):
                    signal[i] = 1
                elif (df[label][i] < 0 and df[label][i-1] >= 0) or \
                    (df[label][i] < 0 and df[label][i-1] < 0 and signal[i-1] == -1):
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def eri_signal(self, df):
        signal = []
        be = 'BEARP_13'
        bu = 'BULLP_13'
        for i in range(1, len(df)):
            if (df[bu][i] is not None and df[bu][i-1] is not None) or \
                (df[be][i] is not None and df[be][i-1] is not None):
                if (df[bu][i] > 0 and df[bu][i-1] < 0) or \
                        (df[bu][i] > 0 and df[bu][i-1] > 0 and signal[i-1] == 1):
                    signal[i] = 1
                elif (df[be][i] < 0 and df[be][i-1] > 0) or \
                        (df[be][i] < 0 and df[be][i-1] < 0 and signal[i-1] == -1):
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def cci_signal(self, df):
        signal = []
        label = 'CCI_20_0.015'
        for i in range(0, len(df)):
            if df[label][i] is not None:
                if df[label][i] > 100:
                    signal[i] = 1
                elif df[label][i] < 100:
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal

    def bbands_signal(self, df):
        signal = np.zeros(len(df))
        label = '_5_2.0'
        for i in range(1, len(df)):
            if df['BBL' + label][i] is not None and df['BBU' + label][i] is not None and df['BBB' + label][i] is not None:
                if (df['4. close'][i] > df['BBL' + label][i]) & \
                    (df['BBB' + label][i] > df['BBB' + label].rolling(14).mean()):
                    signal[i] = 1
                elif (df['4. close'][i] < df['BBL' + label][i]) & \
                        (df['BBB' + label][i] < df['BBB' + label].rolling(14).mean()):
                    signal[i] = -1
                else:
                    signal[i] = 0
            else:
                signal[i] = np.nan
        return signal



    def define_signal(self, path, filename, new_data):
        if new_data:
            df = self.get_stock_dataframe(date_length=200)
        else:
            if not file_exist(path, file_name=filename):
                df = self.get_stock_dataframe(date_length=200)
            else:
                df = read_csv_file(path, filename)
        
        
    
    
        









