from model import Model
import util as u
from datetime import datetime, timedelta
import NLP.util as nlp_u
from APP_FLASK import util
import numpy as np
import torch
class Predictor:
    def __init__(self):
        self.data_folder = f"./csv/"
        self.stock_list = ["AAPL", "AMAZN", "GOOGL", "MSFT", "TSLA"]
        self.window_size_list = [3, 7, 14]
        self.output_size_list = [3, 7, 14]
        self.pytorch_timeseries_model_type_dict = {
            1: "movement",
            2: "magnitude",
            3: "assembler",
            4: "lstm",
            5: "gru",
            6: "transformer",
            7: "pred_price_LSTM"
        },
        self.tensorflow_timeseries_model_type_dict = {
            1: "svm",
            2: "random_forest"
        }
    
    def batch_predict(self, symbol, model_type_list, window_size, output_size):
        result = []
        for model_type in model_type_list:
            result.append(self.predict(symbol, model_type, window_size, output_size))
        return result
    
    def predict(self,symbol, model_type, window_size, output_size):
        if model_type in self.pytorch_timeseries_model_type_dict:
            model_name = f'{self.model_type}_{self.symbol}_w{self.window_size}_o{self.output_step}_d{str(self.data_mode)}.pth'
        elif model_type in self.tensorflow_timeseries_model_type_dict:
            model_name = f'{self.model_type}_{self.symbol}_w{self.window_size}_o{self.output_step}_d{str(self.data_mode)}.pkl'
        model = Model(name=model_name)
        model = model.load_check_point(model_name)

        price_data, stock_data, news_data = self.prepare_data(symbol, window_size)
        X = np.concatenate((price_data, stock_data, news_data), axis=1)
        tensor_data = torch.tensor(X)
        output = model(tensor_data)
        threshold = 0.5
        converted_output = torch.where(output >= threshold, torch.tensor(1), torch.tensor(0))
        
        if torch.all(converted_output == 1):
            output_json = {
                f'{symbol}_{model_type}_{output_size}':  "UP"
            }
        elif torch.all(converted_output == 0):
            output_json = {
                f'{symbol}_{model_type}_{output_size}':  "DOWN"
            }     
        return output_json
    def prepare_data(self, symbol, window_size):
        end = datetime.now().strftime("%Y-%m-%d")
        # Convert the end date string to a datetime object
        end_date = datetime.strptime(end, "%Y-%m-%d")
        # Calculate the start date by subtracting 30 days from the end date
        start_date = end_date - timedelta(days=window_size * 2)
        start = start_date.strftime("%Y-%m-%d")
        stock_df = u.prepare_stock_dataframe(symbol, window_size, start, end, new_data=False)

        # Filter the DataFrame based on the date range
        stock_df = stock_df.loc[start_date:end_date]
        stock_data = stock_df.values[-window_size:]
        price_data = stock_data[:, :6]
        news_data = util.prepare_news_data(stock_df, symbol, window_size, 5, False)
        return price_data, stock_data, news_data
    def fetch_prediction():
        pass