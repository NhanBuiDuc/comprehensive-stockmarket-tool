import torch.nn as nn
from config import config as cf
import torch
import model as m
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import math
class Assembly_regression(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.regression_model = Diff_1()
        model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "attn_diff_1"
        checkpoint = torch.load('./models/' + model_name)
        self.regression_model.load_state_dict(checkpoint['model_state_dict'])

        self.forecasting_model_3 = m.Movement_3()
        model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "attn_movement_3"
        checkpoint = torch.load('./models/' + model_name)
        self.forecasting_model_3.load_state_dict(checkpoint['model_state_dict'])
        self.forecasting_data_mask_3 = checkpoint['features']
        self.forecasting_model_7 = m.Movement_7()
        model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "attn_movement_7"
        checkpoint = torch.load('./models/' + model_name)
        self.forecasting_model_7.load_state_dict(checkpoint['model_state_dict'])

        self.forecasting_model_14 = m.Movement_14()
        model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_14"
        checkpoint = torch.load('./models/' + model_name)
        self.forecasting_model_14.load_state_dict(checkpoint['model_state_dict'])
        
        self.linear_1 = nn.Linear(5, 1)
        self.dropout_1 = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation
        
        self.linear_2 = nn.Linear(2, 1)
        # Adding dropout layer
        self.dropout = nn.Dropout(p=dropout_rate)

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
    
        combined_delta = torch.cat([delta_1, delta_3, delta_7, delta_14, latest_data_point], dim=1)
        
        # Adding dropout to the combined delta
        
        delta = self.linear_1(combined_delta)
        return delta


class Movement_3(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size, input_size=self.input_size)
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.linear = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 3)
        self.tanh = nn.Tanh()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
    def forward(self, x):
        batchsize = x.shape[0]
        #Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear(x)
        x = x.clone()
        x[:, :2] = self.softmax(x[:, :2])
        x[:, 2:] = self.relu(x[:, 2:])
        return x
class Movement_7(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size, input_size=self.input_size)
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.linear = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 3)
        self.tanh = nn.Tanh()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
    def forward(self, x):
        batchsize = x.shape[0]
        #Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear(x)
        x = x.clone()
        x[:, :2] = self.softmax(x[:, :2])
        x[:, 2:] = self.relu(x[:, 2:])
        return x

class Movement_14(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size, input_size=self.input_size)
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.linear = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 3)
        self.tanh = nn.Tanh()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
    def forward(self, x):
        batchsize = x.shape[0]
        #Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear(x)
        x = x.clone()
        x[:, :2] = self.softmax(x[:, :2])
        x[:, 2:] = self.relu(x[:, 2:])
        return x

class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(CausalDilatedConv1d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class CausalDilatedConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, dilation_base):
        super(CausalDilatedConvNet, self).__init__()
        self.layers = nn.ModuleList()
        self.dilation_base = dilation_base
        for i in range(num_layers):
            dilation = self.dilation_base ** i # exponentially increasing dilation
            padding = self.dilation_base ** (i) ** (kernel_size - 1)
            layer = CausalDilatedConv1d(in_channels, out_channels, kernel_size, dilation, padding)
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Diff_1(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.autoencoder = CausalDilatedConvNet(window_size=self.window_size)
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
        self.linear = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 3)
        self.relu = nn.ReLU()

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
        #Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear(x)
        x = x.clone()
        x = self.relu(x[:, 2:])
        return x


def find_divisor(a):
    # Start with a divisor of 2 (the smallest even number)
    divisor = 2
    # Keep looping until we find a divisor that works
    while True:
        # Check if a is divisible by the current divisor
        if a % divisor == 0:
            return divisor
        # If not, increment the divisor and try again
        divisor += 1

