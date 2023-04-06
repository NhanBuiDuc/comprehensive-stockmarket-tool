import torch.nn as nn
from config import config as cf
import torch
import model as m
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
class Assembly_regression(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.regression_model = LSTM_Regression()
        model_name = cf["alpha_vantage"]["symbol"] +  "_"  + "diff_1"
        checkpoint = torch.load('./models/' + model_name)
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
        combined_delta = self.dropout(combined_delta)
        
        delta = self.linear_1(combined_delta)
        delta = self.dropout_1(delta)

        # last_val = torch.cat([latest_data_point, delta], dim=1)
        # last_val = self.linear_2(last_val)
        return delta


class Movement_3(nn.Module):
    def __init__(self, input_size, window_size, lstm_hidden_layer_size, lstm_num_layers, output_steps, attn_num_heads):
        super().__init__()
        self.input_size = (window_size, input_size)
        self.window_size = window_size
        self.lstm_hidden_layer_size = lstm_hidden_layer_size
        self.lstm_num_layers = lstm_num_layers
        self.output_steps = output_steps
        self.attn_num_heads = attn_num_heads

        self.autoencoder = Conv1DAutoencoder(window_size=self.window_size)
        self.lstm = nn.LSTM(input_size = 32, hidden_size=14, num_layers=14, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=30, num_heads=self.attn_num_heads)

        self.linear_2 = nn.Linear(196, 3)
        self.tanh_2 = nn.Tanh()

        self.relu_3 = nn.ReLU()
        self.softmax_3 = nn.Softmax(dim=1)  # Apply softmax activation

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
        #Data extract
        x = self.autoencoder(x)
        # x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        # attn_output, attn_weights = self.multihead_attn(x, x, x)
        # x = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_2(x)
        x = x.clone()
        x[:, :2] = self.softmax_3(x[:, :2])
        x[:, 2:] = self.relu_3(x[:, 2:])
        return x
    

class Movement_7(nn.Module):
    def __init__(self, input_size=14, window_size=14, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2, num_heads=3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_heads = 3

        self.linear_1 = nn.Linear(14, 2)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(64 * 6 * 6, 512)
        # self.fc2 = nn.Linear(512, num_classes)

        self.autoencoder = Conv1DAutoencoder()
        self.lstm = nn.LSTM(input_size = 36, hidden_size=14, num_layers=14, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=36, num_heads=self.num_heads)

        self.linear_2 = nn.Linear(196, 3)
        self.tanh_2 = nn.Tanh()

        self.relu_3 = nn.ReLU()
        self.softmax_3 = nn.Softmax(dim=1)  # Apply softmax activation

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

        # x = self.linear_1(x)
        # x = self.sigmoid_1(x)
        # x = self.dropout_1(x)
        # x = self.tanh_1(x)

        x = self.autoencoder(x)
        x = x.permute(1, 0, 2, 3).reshape(14, batchsize, -1)  # (seq_len, batch_size, embed_dim)
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        x = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_2(x)
        x = x.clone()
        x[:, :2] = self.softmax_3(x[:, :2])
        x[:, 2:] = self.relu_3(x[:, 2:])
        return x
    

class Movement_14(nn.Module):
    def __init__(self, input_size=14, window_size=14, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2, num_heads=3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_heads = 3

        self.autoencoder = Conv1DAutoencoder(window_size=self.window_size)
        self.lstm = nn.LSTM(input_size = 36, hidden_size=14, num_layers=14, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=32, num_heads=self.num_heads)

        self.linear_2 = nn.Linear(196, 3)
        self.tanh_2 = nn.Tanh()

        self.relu_3 = nn.ReLU()
        self.softmax_3 = nn.Softmax(dim=1)  # Apply softmax activation

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

        # x = self.linear_1(x)
        # x = self.sigmoid_1(x)
        # x = self.dropout_1(x)
        # x = self.tanh_1(x)

        x = self.autoencoder(x)
        x = x.permute(1, 0, 2, 3).reshape(14, batchsize, -1)  # (seq_len, batch_size, embed_dim)
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        x = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_2(x)
        x = x.clone()
        x[:, :2] = self.softmax_3(x[:, :2])
        x[:, 2:] = self.relu_3(x[:, 2:])
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
        
        # Decoder layers
        self.tconv3 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.tbn3 = nn.BatchNorm1d(num_features=128)
        self.trelu3 = nn.ReLU()
        self.tconv2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.tbn2 = nn.BatchNorm1d(num_features=64)
        self.trelu2 = nn.ReLU()
        self.tconv1 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.tbn1 = nn.BatchNorm1d(num_features=32)
        self.trelu1 = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()
        
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
        
        # Decoder
        x = self.up(x)
        x3 = self.tconv3(x)
        x3 = self.tbn3(x3)
        x3 = self.trelu3(x3+x2)
        x2 = self.tconv2(x3)
        x2 = self.tbn2(x2)
        x2 = self.trelu2(x2+x1)
        x1 = self.tconv1(x2)
        x1 = self.tbn1(x1)
        x1 = self.trelu1(x1)
        x = self.sigmoid(x1)
        return x



class LSTM_Regression(nn.Module):
    def __init__(self, input_size=12, window_size = 14, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_heads = 3
        self.lstm = nn.LSTM(input_size = 36, hidden_size=10, num_layers=10, batch_first=True)
        self.autoencoder = Conv1DAutoencoder()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=36, num_heads=self.num_heads)
        self.linear_3 = nn.Linear(100, 1)
        self.dropout_3 = nn.Dropout(0.2)
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
        x = self.autoencoder(x)
        x = x.permute(1, 0, 2, 3).reshape(14, batchsize, -1)  # (seq_len, batch_size, embed_dim)
        attn_output, attn_weights = self.multihead_attn(x, x, x)
        x = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # layer 2
        x = self.linear_3(x)
        x = self.dropout_3(x)
        return x
