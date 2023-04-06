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

        self.autoencoder = ConvAutoencoder()
        self.lstm = nn.LSTM(input_size = 30, hidden_size=14, num_layers=14, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=30, num_heads=self.num_heads)

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

        self.autoencoder = ConvAutoencoder()
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

        self.linear_1 = nn.Linear(14, 2)
        self.sigmoid_1 = nn.Sigmoid()
        self.tanh_1 = nn.Tanh()
        self.dropout_1 = nn.Dropout(0.2)

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(64 * 6 * 6, 512)
        # self.fc2 = nn.Linear(512, num_classes)

        self.autoencoder = ConvAutoencoder()
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
    

class ConvAutoencoder(nn.Module):
    def __init__(self, n=10):
        super(ConvAutoencoder, self).__init__()

        self.n = n

        # Encoder layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder layers
        self.t_conv1 = nn.ConvTranspose2d(in_channels=8, out_channels=10, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.t_conv2 = nn.ConvTranspose2d(in_channels=10, out_channels=14, kernel_size=3, stride=1)

    def forward(self, x):
        # Encoder
        batchsize = x.shape[0]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # Decoder
        x = self.t_conv1(x)
        x = self.relu3(x)

        x = self.t_conv2(x)
        x = x.squeeze(1)

        return x
        
class LSTM_Regression(nn.Module):
    def __init__(self, input_size=12, window_size = 14, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_heads = 3
        self.lstm = nn.LSTM(input_size = 36, hidden_size=10, num_layers=10, batch_first=True)
        self.autoencoder = ConvAutoencoder()
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
    
# class Movement_3(nn.Module):
#     def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size

#         self.linear_1 = nn.Linear(14, 1)
#         self.sigmoid_1 = nn.Sigmoid()
#         self.tanh_1 = nn.Tanh()
#         self.dropout_1 = nn.Dropout(0.2)

#         self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=14, batch_first=True)

#         self.linear_2 = nn.Linear(196 , 3)
#         self.tanh_2 = nn.Tanh()

#         self.relu_3 = nn.ReLU()
#         self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                  nn.init.constant_(param, 0.0)
#             elif 'weight_ih' in name:
#                  nn.init.kaiming_normal_(param)
#             elif 'weight_hh' in name:
#                  nn.init.orthogonal_(param)

#     def forward(self, x):
#         batchsize = x.shape[0]

#         x = self.linear_1(x)
#         x = self.sigmoid_1(x)
#         x = self.dropout_1(x)
#         x = self.tanh_1(x)

#         lstm_out, (h_n, c_n) = self.lstm(x)
#         x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

#         x = self.linear_2(x)
#         # x = self.tanh_2(x)
#         x = x.clone()
#         x[:, :2] = self.softmax_3(x[:, :2])
#         x[:, 2:] = self.relu_3(x[:, 2:])
#         return x
    
# class Movement_7(nn.Module):
#     def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size

#         self.linear_1 = nn.Linear(14, 1)
#         self.sigmoid_1 = nn.Sigmoid()
#         self.tanh_1 = nn.Tanh()
#         self.dropout_1 = nn.Dropout(0.2)

#         self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=14, batch_first=True)

#         self.linear_2 = nn.Linear(196 , 3)
#         self.tanh_2 = nn.Tanh()

#         self.relu_3 = nn.ReLU()
#         self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                  nn.init.constant_(param, 0.0)
#             elif 'weight_ih' in name:
#                  nn.init.kaiming_normal_(param)
#             elif 'weight_hh' in name:
#                  nn.init.orthogonal_(param)

#     def forward(self, x):
#         batchsize = x.shape[0]

#         x = self.linear_1(x)
#         x = self.sigmoid_1(x)
#         x = self.dropout_1(x)
#         x = self.tanh_1(x)

#         lstm_out, (h_n, c_n) = self.lstm(x)
#         x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

#         x = self.linear_2(x)
#         # x = self.tanh_2(x)
#         x = x.clone()
#         x[:, :2] = self.softmax_3(x[:, :2])
#         x[:, 2:] = self.relu_3(x[:, 2:])
#         return x
 
# class Movement_14(nn.Module):
#     def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size

#         self.linear_1 = nn.Linear(14, 1)
#         self.sigmoid_1 = nn.Sigmoid()
#         self.tanh_1 = nn.Tanh()
#         self.dropout_1 = nn.Dropout(0.2)

#         self.lstm = nn.LSTM(input_size = 1, hidden_size=14, num_layers=14, batch_first=True)

#         self.linear_2 = nn.Linear(196 , 3)
#         self.tanh_2 = nn.Tanh()

#         self.relu_3 = nn.ReLU()
#         self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

#         self.init_weights()

#     def init_weights(self):
#         for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                  nn.init.constant_(param, 0.0)
#             elif 'weight_ih' in name:
#                  nn.init.kaiming_normal_(param)
#             elif 'weight_hh' in name:
#                  nn.init.orthogonal_(param)

#     def forward(self, x):
#         batchsize = x.shape[0]

#         x = self.linear_1(x)
#         x = self.sigmoid_1(x)
#         x = self.dropout_1(x)
#         x = self.tanh_1(x)

#         lstm_out, (h_n, c_n) = self.lstm(x)
#         x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

#         x = self.linear_2(x)
#         # x = self.tanh_2(x)
#         x = x.clone()
#         x[:, :2] = self.softmax_3(x[:, :2])
#         x[:, 2:] = self.relu_3(x[:, 2:])
#         return x
