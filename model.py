import torch.nn as nn
from config import config as cf
import torch
class Assembly_regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.regression_model = LSTM_Regression()
        checkpoint = torch.load('./models/lstm_regression')
        self.regression_model.load_state_dict(checkpoint['model_state_dict'])

        self.forecasting_model_1 = LSTM_Binary()
        checkpoint = torch.load('./models/lstm_binary1')
        self.forecasting_model_1.load_state_dict(checkpoint['model_state_dict'])

        self.forecasting_model_14 = LSTM_Binary()
        checkpoint = torch.load('./models/lstm_binary14')
        self.forecasting_model_14.load_state_dict(checkpoint['model_state_dict'])
        
        self.linear_1 = nn.Linear(4, 2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

        # define a trainable parameter
        self.up_magnitude = nn.Parameter(torch.randn(size=(1,), dtype =torch.float, requires_grad=True)).to("cuda")
        self.down_magnitude = nn.Parameter(torch.randn(size=(1,), dtype =torch.float, requires_grad=True)).to("cuda")

    def forward(self, x):
        batch_size = x.shape[0]
        # Run the short-term and long-term forecasting models
        prob_1 = self.forecasting_model_1(x)

        prob_14 = self.forecasting_model_14(x)
    
        prob = torch.cat([prob_1,prob_14], dim=1)
        prob = self.linear_1(prob)
        prob = self.softmax(prob)
        max_probs, max_indices = torch.max(prob, dim=1)

        direction = torch.where(max_indices == 0, -1, 1)
        up_magnitude = torch.abs(self.up_magnitude)
        # up_magnitude.unsqueeze_(1)
        down_magnitude = -torch.abs(self.down_magnitude)
        # down_magnitude.unsqueeze_(1)

        change_magnitude = torch.where(direction == -1, down_magnitude, up_magnitude)
        change_magnitude = change_magnitude.unsqueeze(1)
        # Run the regression model
        delta = self.regression_model(x)
        delta = delta + change_magnitude
        x = x[:, -1, 0].unsqueeze(1)
        # Compute the final output
        last_val = x + delta
        return last_val
    
class LSTM_Regression(nn.Module):
    def __init__(self, input_size=12, window_size = 14, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(12, 32)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(64, 1)
        
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

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions
    
class LSTM_Binary(nn.Module):
    def __init__(self, input_size=12, window_size=14, hidden_layer_size=32, num_layers=2, output_size = 14, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(12, 32)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)  # Apply softmax activation

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

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)
        
        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        predictions = self.softmax(predictions)  # Apply softmax activation
        return predictions
