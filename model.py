import torch.nn as nn
from configs.config import config as cf
import torch
import math
import benchmark as bm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class Model:
    def __init__(self, parameters=None, name=None, num_feature=None, model_type=None, full_name=None):
        self.num_feature = num_feature
        self.model_type = model_type
        self.structure = None
        self.name = name
        self.full_name = full_name
        self.parameters = parameters
        self.train_stop_lr = None
        self.train_stop_epoch = None
        self.state_dict = None
        self.pytorch_timeseries_model_type_dict = cf["pytorch_timeseries_model_type_dict"]
        self.tensorflow_timeseries_model_type_dict = cf["tensorflow_timeseries_model_type_dict"]
        if self.name is not None:
            self.construct_structure()

    def construct_structure(self):

        if self.model_type == self.pytorch_timeseries_model_type_dict[1]:
            parameters = self.parameters["model"][self.name]
            self.structure = Movement(self.num_feature, **parameters)

        elif self.model_type == self.pytorch_timeseries_model_type_dict[2]:
            pass
        elif self.model_type == self.pytorch_timeseries_model_type_dict[3]:
            pass
        elif self.model_type == self.pytorch_timeseries_model_type_dict[4]:
            self.parameters = self.parameters["model"][self.name]
            self.structure = bm.LSTM_bench_mark(self.num_feature, **self.parameters)
        elif self.model_type == self.pytorch_timeseries_model_type_dict[5]:
            self.parameters = self.parameters["model"][self.name]
            self.structure = bm.GRU_bench_mark(self.num_feature, **self.parameters)
        elif self.model_type == self.pytorch_timeseries_model_type_dict[6]:
            self.parameters = self.parameters["model"][self.name]
            self.structure = TransformerClassifier(self.num_feature, **self.parameters)
        elif self.model_type == self.tensorflow_timeseries_model_type_dict[1]:
            self.structure = SVC()

        elif self.model_type == self.tensorflow_timeseries_model_type_dict[2]:
            self.structure = RandomForestClassifier()

    def load_check_point(self, file_name):
        check_point = torch.load('./models/' + file_name)
        self = check_point["model"]
        # self.construct_structure()
        self.structure.load_state_dict(check_point['state_dict'])
        return self

    def predict(self, x):
        if self.model_type in self.pytorch_timeseries_model_type_dict.values():
            y = self.structure(x)
            return y
        elif self.model_type in self.tensorflow_timeseries_model_type_dict.values():
            y = self.structure.predict(x)
            return y


class Movement(nn.Module):
    def __init__(self, num_feature, **param):
        super(Movement, self).__init__()
        self.__dict__.update(param)
        self.num_feature = num_feature
        # self.autoencoder = Autoencoder(1, self.window_size, **self.conv1D_param)
        self.autoencoder = Autoencoder(self.num_feature, self.window_size, **self.conv1D_param)
        self.autoencoder_2 = Autoencoder(61, self.window_size, **self.conv1D_param)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.bn_1 = nn.BatchNorm1d(num_features=826)
        self.bn_2 = nn.BatchNorm1d(num_features=854)
        self.bn_3 = nn.BatchNorm1d(num_features=1134)
        self.bn_4 = nn.BatchNorm1d(num_features=1708)
        self.sigmoid = nn.Sigmoid()
        self.soft_max = nn.Softmax(dim=1)
        self.drop_out = nn.Dropout(self.drop_out)
        self.linear_1 = nn.Linear(1708, 500)
        self.linear_2 = nn.Linear(500, 100)
        self.linear_3 = nn.Linear(100, 1)

        self.linear_4 = nn.Linear(1163, 1)
        self.lstm = nn.LSTM(self.conv1D_param["output_size"], hidden_size=self.lstm_hidden_layer_size,
                            num_layers=self.lstm_num_layer,
                            batch_first=True)

        # self.lstm = nn.LSTM(self.num_feature, hidden_size=self.lstm_hidden_layer_size,
        #                     num_layers=self.lstm_num_layer,
        #                     batch_first=True)
        self.lstm_1 = nn.LSTM(59, hidden_size=self.lstm_hidden_layer_size,
                              num_layers=self.lstm_num_layer,
                              batch_first=True)
        self.lstm_2 = nn.LSTM(81, hidden_size=self.lstm_hidden_layer_size,
                              num_layers=self.lstm_num_layer,
                              batch_first=True)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm_1.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]
        # Data extract
        x1 = x.clone()
        x_c = x.clone()
        x = self.autoencoder(x)
        x = torch.concat([x, x1], dim=2)
        x = x.reshape(batchsize, -1)
        x = self.bn_1(x)
        x = x.reshape(batchsize, self.window_size, -1)
        x = self.drop_out(x)
        x = self.relu(x)
        x1 = x.clone()
        lstm_out, (h_n, c_n) = self.lstm_1(x)
        # x = h_n.permute(1, 2, 0).reshape(batchsize, -1)
        x = h_n.permute(1, 2, 0)
        x = torch.concat([x, x1], dim=2)
        x = x.reshape(batchsize, -1)
        x = self.bn_2(x)
        x = x.reshape(batchsize, self.window_size, -1)
        x = self.relu(x)
        x1 = x.clone()
        x = self.autoencoder_2(x)
        x = torch.concat([x, x1], dim=2)
        x = x.reshape(batchsize, -1)
        x = self.bn_3(x)
        x = x.reshape(batchsize, self.window_size, -1)
        x = self.drop_out(x)
        x = self.relu(x)
        x1 = x.clone()
        lstm_out, (h_n, c_n) = self.lstm_2(x)
        # x = h_n.permute(1, 2, 0).reshape(batchsize, -1)
        x = h_n.permute(1, 2, 0)
        x = torch.concat([x, x1, x_c], dim=2)
        x = x.reshape(batchsize, -1)
        x = self.bn_4(x)
        x = x.reshape(batchsize, self.window_size, -1)
        x = self.relu(x)
        x = self.drop_out(x)
        x = x.reshape(batchsize, -1)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.sigmoid(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, num_feature, window_size, output_size, **param):
        super(Autoencoder, self).__init__()
        self.__dict__.update(param)
        self.num_feature = num_feature
        self.window_size = window_size
        self.main_layer = nn.ModuleList()
        self.sub_small_layer = nn.ModuleList()
        self.sub_big_layer = nn.ModuleList()
        self.output_size = output_size
        self.conv1D_type_dict = {
            1: "spatial",
            2: "temporal"
        }
        if self.conv1D_type_dict[self.type] == "spatial":
            # Spatial: 1d convolution will apply on num_feature dim, input and output chanels = window size
            # 1D feature map will slide, containing info of the different features of the same time step
            # Multiple 1D feature maps means multiple relevance between features, for each time step
            # The receptive field is calculated on num_feature(window_size, n)

            self.num_layer = \
                int((math.log((((self.num_feature - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1)) + 1)) / (
                    math.log(self.dilation_base))) + 1

            # Applying 1dconv among input lenght dim
            for i in range(self.num_layer):
                dilation = self.dilation_base ** i  # exponentially increasing dilation
                padding = (self.dilation_base ** i) * (self.kernel_size - 1)
                layer = CausalDilatedConv1d(in_channels=self.window_size,
                                            out_channels=self.window_size,
                                            kernel_size=self.kernel_size,
                                            dilation=dilation,
                                            padding=padding)
                self.main_layer.append(layer)

            for i in range(self.sub_small_num_layer):
                padding = (self.sub_small_kernel_size - 1)
                layer = CausalConv1d(self.window_size, self.window_size, kernel_size=self.sub_small_kernel_size,
                                     padding=padding)
                self.sub_small_layer.append(layer)
            for i in range(self.sub_big_num_layer):
                padding = (self.sub_big_kernel_size - 1)
                layer = CausalConv1d(self.window_size, self.window_size, kernel_size=self.sub_big_kernel_size,
                                     padding=padding)
                self.sub_big_layer.append(layer)

            # output_dim = (input_dim - kernel_size + 2 * padding) / stride + 1

            self.receptive_field_size = int((self.num_feature + (self.kernel_size - 1) * sum(
                self.dilation_base ** i for i in range(self.num_layer))) / self.max_pooling_kernel_size)

            self.small_sub_receptive_field_size = int((self.num_feature - self.sub_small_kernel_size + 2 * (
                    self.sub_small_kernel_size - 1) + 1) / self.max_pooling_kernel_size)
            self.big_sub_receptive_field_size = int((self.num_feature - self.sub_big_kernel_size + 2 * (
                    self.sub_big_kernel_size - 1) + 1) / self.max_pooling_kernel_size)
            self.receptive_field_size = int(
                self.receptive_field_size + self.small_sub_receptive_field_size + self.big_sub_receptive_field_size)
        elif self.conv1D_type_dict[self.type] == "temporal":
            # Temporal: 1d convolution will apply on window_size dim, input and output chanels = input_feature size
            # 1D feature map will slide, containing info of the same feature consecutively among the time step
            # Multiple 1D feature maps means changes day after daye of 1 feature, for each feature
            # The receptive field is calculated on num_feature(window_size, n)

            self.num_layer = int(
                (math.log((((self.window_size - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1)) + 1)) / (
                    math.log(self.dilation_base))) + 1

            # Applying 1dconv among input lenght dim
            for i in range(self.num_layer):
                dilation = self.dilation_base ** i  # exponentially increasing dilation
                padding = (self.dilation_base ** i) * (self.kernel_size - 1)
                layer = CausalDilatedConv1d(in_channels=self.num_feature,
                                            out_channels=self.num_feature,
                                            kernel_size=self.kernel_size,
                                            dilation=dilation,
                                            padding=padding)
                self.main_layer.append(layer)

            for i in range(self.sub_small_num_layer):
                padding = (self.sub_small_kernel_size - 1)
                layer = CausalConv1d(self.num_feature, self.num_feature, kernel_size=self.sub_small_kernel_size,
                                     padding=padding)
                self.sub_small_layer.append(layer)
            for i in range(self.sub_big_num_layer):
                padding = (self.sub_big_kernel_size - 1)
                layer = CausalConv1d(self.num_feature, self.num_feature, kernel_size=self.sub_big_kernel_size,
                                     padding=padding)
                self.sub_big_layer.append(layer)

            # output_dim = (input_dim - kernel_size + 2 * padding) / stride + 1

            self.receptive_field_size = int((self.window_size + (self.kernel_size - 1) * sum(
                self.dilation_base ** i for i in range(self.num_layer))) / self.max_pooling_kernel_size)

            self.small_sub_receptive_field_size = int((self.window_size - self.sub_small_kernel_size + 2 * (
                    self.sub_small_kernel_size - 1) + 1) / self.max_pooling_kernel_size)
            self.big_sub_receptive_field_size = int((self.window_size - self.sub_big_kernel_size + 2 * (
                    self.sub_big_kernel_size - 1 + 1)) / self.max_pooling_kernel_size)
            self.receptive_field_size = int(
                self.receptive_field_size + self.small_sub_receptive_field_size + self.big_sub_receptive_field_size) - 1

        self.maxpool = nn.MaxPool1d(kernel_size=self.max_pooling_kernel_size)
        self.linear_1 = nn.Linear(self.receptive_field_size, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch = x.shape[0]
        if self.conv1D_type_dict[self.type] == "temporal":
            x = x.permute(0, 2, 1)
        x1 = x.clone()
        x2 = x.clone()
        x3 = x.clone()
        for layer in self.main_layer:
            x1 = layer(x1)
        x1 = self.maxpool(x1)
        for layer in self.sub_small_layer:
            x2 = layer(x2)
        x2 = self.maxpool(x2)
        for layer in self.sub_big_layer:
            x3 = layer(x3)
        x3 = self.maxpool(x3)
        concat = torch.cat([x1, x2, x3], dim=2)
        out = self.linear_1(concat)
        return out


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
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, num_feature,  **param):
        super().__init__()
        self.__dict__.update(param)
        self.num_feature = num_feature
        self.transformer = nn.Transformer(d_model=self.num_feature, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers,
                                          dim_feedforward=self.dim_feedforward, dropout=self.dropout)
        self.fc = nn.Linear( self.num_feature * self.window_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch = x.shape[0]
        x = self.transformer(x, x)  # self-attention over the input sequence
        x = x.reshape(batch, -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class PredictPriceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        
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
        return predictions[:,-1]