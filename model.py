import torch.nn as nn
from config import config as cf
import torch
import model as m
class Assembly_regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression_model = LSTM_Regression()
        checkpoint = torch.load('./models/lstm_regression')
        self.regression_model.load_state_dict(checkpoint['model_state_dict'])

        self.forecasting_model_3 = m.Movement_3()
        model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_14"
        checkpoint = torch.load('./models/' + model_name)
        self.forecasting_model_3.load_state_dict(checkpoint['model_state_dict'])
        
        self.forecasting_model_7 = m.Movement_7()
        model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_14"
        checkpoint = torch.load('./models/' + model_name)
        self.forecasting_model_7.load_state_dict(checkpoint['model_state_dict'])

        self.forecasting_model_14 = m.Movement_14()
        model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_14"
        checkpoint = torch.load('./models/' + model_name)
        self.forecasting_model_14.load_state_dict(checkpoint['model_state_dict'])
        
        self.linear_1 = nn.Linear(4, 1)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

    def forward(self, x):
        batch_size = x.shape[0]
        latest_data_point = x[:, -1, 0].unsqueeze(1) 
        # Run the short-term and long-term forecasting models
        prob_3 = self.forecasting_model_3(x)[:, :2]
        delta_3 = self.forecasting_model_3(x)[:, 2:]

        prob_7 = self.forecasting_model_7(x)[:, :2]
        delta_7 = self.forecasting_model_7(x)[:, 2:]

        prob_14 = self.forecasting_model_14(x)[:, :2]
        delta_14 = self.forecasting_model_14(x)[:, 2:]
        
        max_probs, max_indices = torch.max(prob_3, dim=1)
        direction_3 = torch.where(max_indices == 0, -1, 1).unsqueeze(1) 
        delta_3 = latest_data_point + (direction_3 * delta_3 / 100) * latest_data_point

        max_probs, max_indices = torch.max(prob_7, dim=1)
        direction_7 = torch.where(max_indices == 0, -1, 1).unsqueeze(1) 
        delta_7 = latest_data_point + (direction_7 * delta_7 / 100)  * latest_data_point
        
        max_probs, max_indices = torch.max(prob_14, dim=1)
        direction_14 = torch.where(max_indices == 0, -1, 1).unsqueeze(1) 
        delta_14 = latest_data_point + (direction_14 * delta_14 / 100) * latest_data_point
        # Run the regression model
        delta_1 = self.regression_model(x)
    
        combined_delta = torch.concat([ delta_1, delta_3, delta_7, delta_14], dim= 1)
        delta = self.linear_1(combined_delta)

        # Compute the final output
        last_val = latest_data_point + delta
        return last_val
    
class LSTM_Regression(nn.Module):
    def __init__(self, input_size=12, window_size = 14, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.linear_1 = nn.Linear(14, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.dropout_1 = nn.Dropout(0.2)
        self.lstm_2 = nn.LSTM(input_size = 1, hidden_size=10, num_layers=10, batch_first=True)

        self.linear_3 = nn.Linear(100, 1)
        self.dropout_3 = nn.Dropout(0.2)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm_2.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.linear_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout_1(x)
        
        lstm_out, (h_n, c_n) = self.lstm_2(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # layer 2
        x = self.linear_3(x)
        x = self.dropout_3(x)
        return x
    
class LSTM_Classifier_1(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(14, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=2, batch_first=True)

        self.linear_2 = nn.Linear(14 , 2)
        self.tanh_2 = nn.Tanh()
        self.dropout_2 = nn.Dropout(0.2)

        self.sigmoid_3 = nn.Sigmoid()
        self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = self.linear_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout_1(x)
        x = self.tanh_1(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1:, :].reshape(batchsize, -1)
        x = self.linear_2(x)
        x = self.tanh_2(x)
        x = self.dropout_2(x)
        # x = self.sigmoid_3(x)
        x = self.softmax_3(x)

        return x
    
class LSTM_Classifier_7(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(14, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=2, batch_first=True)

        self.linear_2 = nn.Linear(28 , 2)
        self.tanh_2 = nn.Tanh()
        self.dropout_2 = nn.Dropout(0.2)

        self.sigmoid_3 = nn.Sigmoid()
        self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = self.linear_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout_1(x)
        x = self.tanh_1(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.linear_2(x)
        x = self.tanh_2(x)
        x = self.dropout_2(x)
        # x = self.sigmoid_3(x)
        x = self.softmax_3(x)

        return x

class LSTM_Classifier_14(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(14, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=2, batch_first=True)

        self.linear_2 = nn.Linear(28 , 2)
        self.tanh_2 = nn.Tanh()
        self.dropout_2 = nn.Dropout(0.2)

        self.sigmoid_3 = nn.Sigmoid()
        self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = self.linear_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout_1(x)
        x = self.tanh_1(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.linear_2(x)
        x = self.tanh_2(x)
        x = self.dropout_2(x)
        # x = self.sigmoid_3(x)
        x = self.softmax_3(x)

        return x
    
class Movement_3(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(14, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=14, batch_first=True)

        self.linear_2 = nn.Linear(196 , 3)
        self.tanh_2 = nn.Tanh()

        self.relu_3 = nn.ReLU()
        self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = self.linear_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout_1(x)
        x = self.tanh_1(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.linear_2(x)
        # x = self.tanh_2(x)
        x = x.clone()
        x[:, :2] = self.softmax_3(x[:, :2])
        x[:, 2:] = self.relu_3(x[:, 2:])
        return x
class Movement_7(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(14, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=14, batch_first=True)

        self.linear_2 = nn.Linear(196 , 3)
        self.tanh_2 = nn.Tanh()

        self.relu_3 = nn.ReLU()
        self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = self.linear_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout_1(x)
        x = self.tanh_1(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.linear_2(x)
        # x = self.tanh_2(x)
        x = x.clone()
        x[:, :2] = self.softmax_3(x[:, :2])
        x[:, 2:] = self.relu_3(x[:, 2:])
        return x
class Movement_14(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(14, 1)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=14, batch_first=True)

        self.linear_2 = nn.Linear(196 , 3)
        self.tanh_2 = nn.Tanh()

        self.relu_3 = nn.ReLU()
        self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        x = self.linear_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout_1(x)
        x = self.tanh_1(x)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        x = self.linear_2(x)
        # x = self.tanh_2(x)
        x = x.clone()
        x[:, :2] = self.softmax_3(x[:, :2])
        x[:, 2:] = self.relu_3(x[:, 2:])
        return x