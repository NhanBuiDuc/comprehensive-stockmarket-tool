import torch.nn as nn
from config import config as cf
import torch
class Assembly_regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression_model = LSTM_Regression()
        checkpoint = torch.load('./models/lstm_regression')
        self.regression_model.load_state_dict(checkpoint['model_state_dict'])

        self.forecasting_model_1 = LSTM_Classifier_1()
        checkpoint = torch.load('./models/lstm_classification_1')
        self.forecasting_model_1.load_state_dict(checkpoint['model_state_dict'])
        
        self.forecasting_model_7 = LSTM_Classifier_7()
        checkpoint = torch.load('./models/lstm_classification_7')
        self.forecasting_model_7.load_state_dict(checkpoint['model_state_dict'])

        self.forecasting_model_14 = LSTM_Classifier_14()
        checkpoint = torch.load('./models/lstm_classification_14')
        self.forecasting_model_14.load_state_dict(checkpoint['model_state_dict'])
        
        self.linear_1 = nn.Linear(5, 1)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

    def forward(self, x):
        batch_size = x.shape[0]
        # Run the short-term and long-term forecasting models
        prob_1 = self.forecasting_model_1(x)
        prob_7 = self.forecasting_model_7(x)
        prob_14 = self.forecasting_model_14(x)
        max_probs, max_indices = torch.max(prob_1, dim=1)
        direction = torch.where(max_indices == 0, -1, 1)
        direction = direction.unsqueeze(1)
        # Run the regression model
        delta = self.regression_model(x)
        prob_delta = torch.where(delta < 0, -1, 1)
    
        delta = torch.concat([direction, delta, prob_delta], dim= 1)
        delta = self.linear_1(delta)
        x = x[:, -1, 0].unsqueeze(1)
        # Compute the final output
        last_val = x + delta
        return last_val
    
class LSTM_Regression(nn.Module):
    def __init__(self, input_size=12, window_size = 14, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.linear_1 = nn.Linear(12, 1)
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

        self.linear_1 = nn.Linear(12, 1)
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

        self.linear_1 = nn.Linear(12, 1)
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

        self.linear_1 = nn.Linear(12, 1)
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
class LSTM_Convolution_Classifier_14(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size = 12, hidden_size=64, num_layers=64, batch_first=True)

        self.conv_2 = nn.Conv2d(in_channels=14, out_channels=32, kernel_size=3)
        self.selu_2 = nn.SELU()
        self.dropout_2 = nn.Dropout(0.2)

        self.linear_3 = nn.Linear(14, 2)
        self.sigmoid_3 = nn.Sigmoid()
        # self.softmax_3 =nn.Softmax(dim=1)  # Apply softmax activation

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
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out.reshape(batchsize, -1) 
        x = self.linear_2(x)
        x = self.selu_2(x)
        x = self.dropout_2(x)
        x = self.linear_3(x)
        x = self.sigmoid_3(x)

        return x