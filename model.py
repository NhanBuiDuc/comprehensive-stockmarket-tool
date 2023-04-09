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
        self.forecasting_data_mask_7 = checkpoint['features']
        
        self.forecasting_model_14 = m.Movement_14()
        model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "movement_14"
        checkpoint = torch.load('./models/' + model_name)
        self.forecasting_model_14.load_state_dict(checkpoint['model_state_dict'])
        self.forecasting_data_mask_14 = checkpoint['features']

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

        self.kernel_size = self.window_size
        self.dilation_base = 3

        # Calculate the number of layers
        num_layers = int(
            (math.log( ( ((self.window_size - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1) ) + 1)) / (math.log(self.dilation_base))
        )

        self.autoencoder = CausalDilatedConvNet(window_size= self.window_size, out_channels = self.window_size, kernel_size = self.kernel_size, num_layers=num_layers, dilation_base=self.dilation_base)
        self.lstm = nn.LSTM(50, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
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

        self.kernel_size = self.window_size
        self.dilation_base = 3

        # Calculate the number of layers
        num_layers = int(
            (math.log( ( ((self.window_size - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1) ) + 1)) / (math.log(self.dilation_base))
        )

        self.autoencoder = CausalDilatedConvNet(input_size = self.input_size, in_channels = self.window_size, out_channels = self.window_size, kernel_size = self.kernel_size, num_layers=num_layers, dilation_base=self.dilation_base)
        self.lstm = nn.LSTM(50, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
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

        self.kernel_size = self.window_size
        self.dilation_base = 3

        # Calculate the number of layers
        num_layers = int(
            (math.log( ( ((self.window_size - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1) ) + 1)) / (math.log(self.dilation_base))
        )

        self.autoencoder = CausalDilatedConvNet(input_size = self.input_size, in_channels = self.window_size, out_channels = self.window_size, kernel_size = self.kernel_size, num_layers=num_layers, dilation_base=self.dilation_base)
        self.lstm = nn.LSTM(50, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
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
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.autoencoder_final_dim = 32
        self.kernel_size = self.input_size
        self.dilation_base = 3

        # Calculate the number of layers
        num_layers = int(
            (math.log( ( ((self.input_size - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1) ) + 1)) / (math.log(self.dilation_base))
        )

        self.autoencoder = CausalDilatedConvNet(input_size = self.input_size, in_channels = self.window_size, out_channels = self.window_size, kernel_size = self.kernel_size, num_layers=num_layers, dilation_base=self.dilation_base)
        self.lstm = nn.LSTM(31, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        x = self.conv(x)
        return x
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_size = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding = padding)
    
    def forward(self, x):
        x = self.conv(x)
        return x
    
class CausalDilatedConvNet(nn.Module):
    def __init__(self, window_size, out_channels, kernel_size, num_layers, dilation_base):
        super(CausalDilatedConvNet, self).__init__()
        self.dilation_layers = nn.ModuleList()
        self.causal_1d_layers = nn.ModuleList()
        self.causal_full_layers = nn.ModuleList()
        self.window_size = window_size
        self.out_channels = out_channels
        self.dilation_base = dilation_base
        self.num_layers = num_layers
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.linear = nn.Linear(in_features=132,out_features=50)

        self.relu = nn.ReLU()
        for i in range(num_layers):
            dilation = self.dilation_base ** i # exponentially increasing dilation
            padding = ((self.dilation_base ** i)) * (kernel_size - 1)
            layer = CausalDilatedConv1d(in_channels = self.window_size, 
                                        out_channels= out_channels, 
                                        kernel_size = kernel_size, 
                                        dilation= dilation, 
                                        padding = padding)
            self.dilation_layers.append(layer)
        for i in range(num_layers):
            kernel_size = 1
            padding = kernel_size -1
            layer = CausalConv1d(self.window_size, out_channels, kernel_size = kernel_size, padding = padding)
            self.causal_1d_layers.append(layer)   
        for i in range(num_layers):
            kernel_size = self.window_size
            padding = kernel_size -1
            layer = CausalConv1d(self.window_size, out_channels, kernel_size = kernel_size, padding = padding)
            self.causal_full_layers.append(layer) 

    def forward(self, x):
        x1 = x.clone()
        x2 = x.clone()
        x3 = x.clone()
        for layer in self.dilation_layers:
            x1 = layer(x1)
        for layer in self.dilation_layers:
            x2 = layer(x2)
        for layer in self.causal_full_layers:
            x3 = layer(x3)
            
        concat = torch.cat([x1, x2, x3], dim=2)
        concat = self.linear(concat)
        concat = self.relu(concat)
        return concat
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
        input_size = 11
        self.kernel_size = 3
        self.dilation_base = 3

        # Calculate the number of layers
        num_layers = int(
            (math.log( ( ((self.input_size - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1) ) + 1)) 
            / (math.log(self.dilation_base))
        )

        self.autoencoder = CausalDilatedConvNet(in_channels = self.window_size, out_channels = self.window_size, kernel_size = self.kernel_size, num_layers=num_layers, dilation_base=self.dilation_base)
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

