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
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, use_attn, attn_num_heads, attn_multi_head_scaler):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.use_attn = use_attn
        self.attn_num_heads = attn_num_heads
        self.autoencoder_final_dim = 32
        self.attn_multi_head_scaler = attn_multi_head_scaler
        self.embed_dim = attn_multi_head_scaler * find_divisor(self.input_size)
        self.autoencoder = Conv1DAutoencoder(window_size=self.window_size)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=self.embed_dim)
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
        if(self.use_attn):
            self.linear = nn.Linear(self.input_size, 3)
        self.linear = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 3)
        self.tanh = nn.Tanh()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.multihead_attn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
    def forward(self, x):
        batchsize = x.shape[0]
        if(self.use_attn):
            #Data extract
            x = self.autoencoder(x)
            x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
            attn_output, attn_weights = self.multihead_attn(x, x, x)
            x = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
            lstm_out, (h_n, c_n) = self.lstm(x)
            x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
            x = self.linear(x)
            x = x.clone()
            x[:, :2] = self.softmax(x[:, :2])
            x[:, 2:] = self.relu(x[:, 2:])
            return x

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
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, use_attn, attn_num_heads, attn_multi_head_scaler):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.use_attn = use_attn
        self.attn_num_heads = attn_num_heads
        self.autoencoder_final_dim = 32
        self.attn_multi_head_scaler = attn_multi_head_scaler
        self.embed_dim = attn_multi_head_scaler * find_divisor(self.input_size)
        self.autoencoder = Conv1DAutoencoder(window_size=self.window_size)
        self.multihead_attn = nn.MultiheadAttention(embed_dim = self.input_size // 2, num_heads=1)
        self.lstm = nn.LSTM(input_size = self.input_size // 2, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
        if(self.use_attn):
            self.linear = nn.Linear(self.input_size, 3)
        self.linear = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 3)
        self.tanh = nn.Tanh()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.multihead_attn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
    def forward(self, x):
        batchsize = x.shape[0]
        if(self.use_attn):
            #Data extract
            x = self.autoencoder(x)
            x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
            attn_output, attn_weights = self.multihead_attn(x, x, x)
            x = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
            lstm_out, (h_n, c_n) = self.lstm(x)
            x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
            x = self.linear(x)
            x = x.clone()
            x[:, :2] = self.softmax(x[:, :2])
            x[:, 2:] = self.relu(x[:, 2:])
            return x

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
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, use_attn, attn_num_heads, attn_multi_head_scaler):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.use_attn = use_attn
        self.attn_num_heads = attn_num_heads
        self.autoencoder_final_dim = 32
        self.attn_multi_head_scaler = attn_multi_head_scaler
        self.embed_dim = attn_multi_head_scaler * find_divisor(self.input_size)
        self.autoencoder = Conv1DAutoencoder(window_size=self.window_size)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=self.embed_dim)
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
        if(self.use_attn):
            self.linear = nn.Linear(self.input_size, 3)
        self.linear = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layers, 3)
        self.tanh = nn.Tanh()

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        self.init_weights()

    def init_weights(self):
        for name, param in self.multihead_attn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
    def forward(self, x):
        batchsize = x.shape[0]
        if(self.use_attn):
            #Data extract
            x = self.autoencoder(x)
            x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
            attn_output, attn_weights = self.multihead_attn(x, x, x)
            x = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
            lstm_out, (h_n, c_n) = self.lstm(x)
            x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
            x = self.linear(x)
            x = x.clone()
            x[:, :2] = self.softmax(x[:, :2])
            x[:, 2:] = self.relu(x[:, 2:])
            return x

        #Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear(x)
        x = x.clone()
        x[:, :2] = self.softmax(x[:, :2])
        x[:, 2:] = self.relu(x[:, 2:])
        return x

class Conv1DAutoencoder(nn.Module):
    def __init__(self, window_size):
        super(Conv1DAutoencoder, self).__init__()
        
        # Encoder layers
        self.conv1 = nn.Conv1d(in_channels=window_size, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.relu3 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        
        x = self.pool(x3)
    
        return x



class Diff_1(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, use_attn, attn_num_heads, attn_multi_head_scaler):
        super().__init__()
        self.input_size = input_size
        self.input_shape = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.use_attn = use_attn
        self.attn_num_heads = attn_num_heads
        self.autoencoder_final_dim = 32
        self.attn_multi_head_scaler = attn_multi_head_scaler
        self.embed_dim = attn_multi_head_scaler * find_divisor(self.input_size)
        self.autoencoder = Conv1DAutoencoder(window_size=self.window_size)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.input_size, num_heads=self.embed_dim)
        self.lstm = nn.LSTM(input_size = self.input_size, hidden_size=self.lstm_hidden_layer_size, num_layers=self.lstm_num_layers, batch_first=True)
        if(self.use_attn):
            self.linear = nn.Linear(self.input_size, 1)
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
        if(self.use_attn):
            #Data extract
            x = self.autoencoder(x)
            x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
            attn_output, attn_weights = self.multihead_attn(x, x, x)
            x = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
            lstm_out, (h_n, c_n) = self.lstm(x)
            x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
            x = self.linear(x)
            x = self.relu(x[:, 2:])
            return x

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
