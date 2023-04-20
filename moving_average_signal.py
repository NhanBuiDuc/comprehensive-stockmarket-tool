import os 
import tensorflow as tf
import datetime
import util
from util import *
import numpy as np
import pandas as pd
from config import config as cf

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
            'close': df['4. close'],
            'open': df['1. open'],
            'high': df['2. high'],
            'low': df['3. low'],
            'volume': df['6. volume']
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









