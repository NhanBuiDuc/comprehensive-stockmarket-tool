from model import Model
import util as u
from datetime import datetime, timedelta
import NLP.util as nlp_u
from APP_FLASK import util
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from functools import reduce



class Predictor:
    def __init__(self):
        self
        self.data_folder = f"./csv/"
        # self.stock_list = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA"]
        self.stock_list = ["AAPL"]
        self.window_size_list = [3, 7, 14]
        self.output_size_list = [3, 7, 14]
        self.scaler = MinMaxScaler(feature_range=(-100, 100))

        self.config_dict = {
            "AAPL": {
                "svm": [[7,2], [7,1], [14,1]],
                "random_forest": [[3,0], [7,2], [3,0]],
                "xgboost": [[3,0], [7,2], [3,0]],
            },
            "AMZN": {
                "svm": [[3,1], [14,1], [7,1]],
                "random_forest": [[3,0], [3,0], [7,2]],
                "xgboost": [[14,1], [14,1], [14,1]]
            },
            "GOOGL": {
                "svm": [[7,2], [7,2], [3,2]],
                "random_forest": [[7,1], [3,1], [7,1]],
                "xgboost": [[7,1], [7,1], [7,1]]
            },
            "MSFT": {
                "svm": [[3,1], [3,0], [3,0]],
                "random_forest": [[7,2], [3,2], [3,0]],
                "xgboost": [[7,1], [7,2], [3,0]]
            },
            "TSLA": {
                "svm": [[7,2], [14,2], [14,1]],
                "random_forest": [[7,1], [14,1], [7,2]],
                "xgboost": [[3,1], [14,1], [3,2]]
            }
        }
        

        self.pytorch_timeseries_model_type_dict = [
            "movement",
            "magnitude",
            "assembler",
            "lstm",
            "gru",
            "transformer",
            "pred_price_LSTM"
        ]
        self.tensorflow_timeseries_model_type_dict = [
            "svm",
            "random_forest",
            "xgboost"

        ]
        self.svm_data_mode_dict = {
            3: 0,
            7: 1,
            14: 2,
        }
        self.rf_data_mode_dict = {
            3: 0,
            7: 1,
            14: 2,
        }
        self.xg_data_mode_dict = {
            3: 0,
            7: 1,
            14: 2,
        }

    
    def batch_predict(self, symbol, model_type_list, window_size, output_step):
        result = {}
        for model_type in model_type_list:
            result[model_type] = self.predict(symbol, model_type, window_size, output_step)
        print(result)
        return result
    
    # def predict(self, symbol, model_type, output_step):
    #     match model_type:
    #         case 'svm':
    #             data_mode = self.svm_data_mode_dict[output_step]
    #             model_name = f'{model_type}_{symbol}_w{window_size}_o{output_step}_d{data_mode}'
    #         case 'xgboost':
    #             data_mode = self.xg_data_mode_dict[output_step]
    #             model_name = f'{model_type}_{symbol}_w{window_size}_o{output_step}_d{data_mode}'
    #         case 'random_forest':
    #             data_mode = self.rf_data_mode_dict[output_step]
    #             model_name = f'{model_type}_{symbol}_w{window_size}_o{output_step}_d{data_mode}'
        
    #     model = Model(model_type=model_type)
    #     model = model.load_check_point(model_type, model_name)

    #     price_data, stock_data, news_data = self.prepare_data(symbol, window_size)
    #     X = np.concatenate((price_data, stock_data, news_data), axis=1)


    #     if data_mode == 0:
    #         if model_type == "transformer":
    #             stock_tensor = torch.tensor(stock_data).float().to("cuda")
    #             news_tensor = torch.tensor(news_data).float().to("cuda")
    #             model.structure.to("cuda")
    #             model.data_mode = 0
    #             output = model.structure(stock_tensor, news_tensor)
    #         else:
    #             tensor_data = torch.tensor(price_data)
    #             output = model.predict(tensor_data)
    #     elif data_mode == 1:
    #         if model_type == "transformer":
    #             stock_tensor = torch.tensor(stock_data).float().to("cuda")
    #             news_tensor = torch.tensor(news_data).float().to("cuda")
    #             model.structure.to("cuda")
    #             model.data_mode = 1
    #             output = model.structure(stock_tensor, news_tensor)
    #         else:
    #             tensor_data = torch.tensor(stock_data)
    #             output = model.predict(tensor_data)
    #     else:
    #         if model_type == "transformer":
    #             stock_tensor = torch.tensor(stock_data).float().to("cuda")
    #             news_tensor = torch.tensor(news_data).float().to("cuda")
    #             model.structure.to("cuda")
    #             model.data_mode = 2
    #             output = model.structure(stock_tensor, news_tensor)
    #         else:
    #             tensor_data = torch.tensor(X)
    #             output = model.predict(tensor_data)

    #     threshold = 0.5
    #     converted_output = torch.where(output >= threshold, torch.tensor(1), torch.tensor(0))
        
    #     if torch.all(converted_output == 1):
    #         output_json = {
    #             f'{model_type}_{symbol}_w{window_size}_o{output_step}': "UP"
    #         }
    #     elif torch.all(converted_output == 0):
    #         output_json = {
    #             f'{model_type}_{symbol}_w{window_size}_o{output_step}': "DOWN"
    #         }     
    #     return output_json

    def predict_1(self, symbol, model_type, output_step):
        if output_step == 3:
            data_mode = self.config_dict[symbol][model_type][0][1]
            window_size = self.config_dict[symbol][model_type][0][0]
        elif output_step == 7:
            data_mode = self.config_dict[symbol][model_type][1][1]
            window_size = self.config_dict[symbol][model_type][1][0]
        elif output_step == 14:
            data_mode = self.config_dict[symbol][model_type][2][1]
            window_size = self.config_dict[symbol][model_type][2][0]
        model_name = f'{model_type}_{symbol}_w{window_size}_o{output_step}_d{data_mode}'


        price_data, stock_data, news_data = self.prepare_data(symbol, window_size)
        X = np.concatenate((stock_data, news_data), axis=1)

        model = Model(model_type=model_type)
        model = model.load_check_point(model_type, model_name)
        if data_mode == 0:
            if model_type == "transformer":
                stock_tensor = torch.tensor(stock_data).float().to("cuda")
                news_tensor = torch.tensor(news_data).float().to("cuda")
                model.structure.to("cuda")
                model.data_mode = 0
                output = model.structure(stock_tensor, news_tensor)
            else:

                tensor_data = torch.tensor(price_data)[-1].unsqueeze(0)
                output = model.predict(tensor_data)
        elif data_mode == 1:
            if model_type == "transformer":
                stock_tensor = torch.tensor(stock_data).float().to("cuda")
                news_tensor = torch.tensor(news_data).float().to("cuda")
                model.structure.to("cuda")
                model.data_mode = 1
                output = model.structure(stock_tensor, news_tensor)
            else:
                tensor_data = torch.tensor(stock_data)[-1].unsqueeze(0)
                output = model.predict(tensor_data)
        else:
            if model_type == "transformer":
                stock_tensor = torch.tensor(stock_data).float().to("cuda")
                news_tensor = torch.tensor(news_data).float().to("cuda")
                model.structure.to("cuda")
                model.data_mode = 2
                output = model.structure(stock_tensor, news_tensor)
            else:
                tensor_data = torch.tensor(X)[-1].unsqueeze(0)
                output = model.predict(tensor_data)

        threshold = 0.5
        converted_output = torch.where(output >= threshold, torch.tensor(1), torch.tensor(0))
        
        if torch.all(converted_output == 1):
            output_json = {
                f'{model_type}_{symbol}_w{window_size}_o{output_step}': "UP"
            }
        elif torch.all(converted_output == 0):
            output_json = {
                f'{model_type}_{symbol}_w{window_size}_o{output_step}': "DOWN"
            }     
        return output_json


    def prepare_data(self, symbol, window_size):
        
        end = '2023-05-01'
        # Convert the end date string to a datetime object
        end_date = datetime.strptime(end, "%Y-%m-%d")
        # Calculate the start date by subtracting 30 days from the end date
        start_date = end_date - timedelta(days=window_size * 2)
        start = start_date.strftime("%Y-%m-%d")
        stock_df = u.prepare_stock_dataframe(symbol, window_size, start, end, new_data=False)
        
        # Filter the DataFrame based on the date range
        stock_df = stock_df.loc[start_date:end_date]
        stock_data = stock_df.values[-window_size:]
        price_data = stock_data[:, :5]
        #nlp_u.update_news_url(symbol)
        news_data = util.prepare_news_data(stock_df, symbol, window_size, 5, False)
        stock_data = self.scaler.fit_transform(stock_data)
        price_data = self.scaler.fit_transform(price_data)
        return price_data, stock_data, news_data

    def fetch_prediction(self):
        result = []
        for symbol in self.stock_list:
            for model_type in self.tensorflow_timeseries_model_type_dict:
                    for output_step in self.output_size_list:
                        result.append(self.predict_1(symbol, model_type, output_step))
        # result = dict(result)
        result = reduce(lambda d1, d2: d1.update(d2) or d1, result, {})
        print(result)
        return result


'''
APPLE:
RF: 3-3-0 
SVM: 3-7-1
XG: 3-3-0

MSFT:
RF: 14-3-0
SVM:14-3-0
XG:7-7-2

TSLA:
RF: 14-7-2
SVM: 7-14-2
XG: 14-3-2

GOOGL:
RF: 14-7-1
SVM: 7-7-2
XG: 14-7-1

AMZN:
RF: 14-7-2
SVM:14-7-1
XG:14-14-1

'''