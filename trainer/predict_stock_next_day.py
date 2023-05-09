import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from alpha_vantage.timeseries import TimeSeries

import dataset
import util
from configs.config import config


class Predict_price_1_day:
    def __int__(self):
        pass

    def get_data_pred(config, new_data=False):
        if new_data:
            ts = TimeSeries(key=config["alpha_vantage"]["key"])
            data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"],
                                                    outputsize=config["alpha_vantage"]["outputsize"])

            state_date = "2018-01-01"
            data_date = [date for date in data.keys() if date >= state_date]
            data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data_date]
            data_close_price.reverse()
            # now from earliest to latest
            data_date.reverse()
            data_close_price = np.array(data_close_price)

            num_data_points = len(data_date)
            #display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points - 1]

        else:
            symbol = config['alpha_vantage']['symbol']


        return data_date, data_close_price, num_data_points


