import torch.nn as nn
from config import config as cf
class assembly_regression():
    def __init__(self, regression_model, forescasting_model):
        self.regression_model = regression_model
        self.forescasting_model = forescasting_model
    def forwards(sellf, x):
        pass
    def predict(self, x):
        values = self.regression_model.predict(x)
        direction = self.forescasting_model.predict(x)
        direction[direction == 0] = -1
        values = values * direction
        return values
class LSTM_Classification():
    pass

class LSTM_Regression(nn.Module):
    def __init__(self, input_size=12, window_size = 14, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(12, 32)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(64, output_size)
        
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