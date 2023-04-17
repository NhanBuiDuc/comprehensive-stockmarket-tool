import torch.nn as nn
from config import config as cf
import torch
import math


class Model:
    def __init__(self, name, save_name, num_feature, model_type):
        self.num_feature = num_feature
        self.model_type = model_type
        self.structure = None
        self.name = name
        self.parameters = None
        self.train_stop_lr = None
        self.train_stop_epoch = None
        self.model_type_dict = {
            1: "movement",
            2: "magnitude",
            3: "assembler"
        }

        self.construct_structure()

    def predict(self):
        pass

    def construct_structure(self):
        self.parameters = cf["model"][self.model_type]

        if self.model_type == self.model_type_dict[1]:
            self.parameters = cf["model"][self.name]
            self.structure = Movement(self.num_feature, **self.parameters)
        elif self.model_type == self.model_type_dict[2]:
            pass
        elif self.model_type == self.model_type_dict[2]:
            pass

    def load_check_point(self):
        pass


class Autoencoder(nn.Module):
    def __init__(self, num_feature, window_size, **param):
        super(Autoencoder, self).__init__()
        self.__dict__.update(param)
        self.num_feature = num_feature
        self.window_size = window_size
        self.main_layer = nn.ModuleList()
        self.sub_small_layer = nn.ModuleList()
        self.sub_big_layer = nn.ModuleList()

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
                int((math.log((((self.num_feature - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1)) + 1)) / (math.log(self.dilation_base))) + 1

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

            self.receptive_field_size = int((self.num_feature + (self.kernel_size - 1) * sum(self.dilation_base ** i for i in range(self.num_layer))) / self.max_pooling_kernel_size)

            self.small_sub_receptive_field_size = int((self.num_feature - self.sub_small_kernel_size + 2 * (self.sub_small_kernel_size - 1) + 1) / self.max_pooling_kernel_size)
            self.big_sub_receptive_field_size = int((self.num_feature - self.sub_big_kernel_size + 2 * (self.sub_big_kernel_size - 1) + 1) / self.max_pooling_kernel_size)
            self.receptive_field_size = int(self.receptive_field_size + self.small_sub_receptive_field_size + self.big_sub_receptive_field_size)
        elif self.conv1D_type_dict[self.type] == "temporal":
            # Temporal: 1d convolution will apply on window_size dim, input and output chanels = input_feature size
            # 1D feature map will slide, containing info of the same feature consecutively among the time step
            # Multiple 1D feature maps means changes day after daye of 1 feature, for each feature
            # The receptive field is calculated on num_feature(window_size, n)

            self.num_layer = int((math.log((((self.window_size - 1) * (self.dilation_base - 1)) / (self.kernel_size - 1)) + 1)) / (math.log(self.dilation_base))) + 1

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

            self.receptive_field_size = int((self.window_size + (self.kernel_size - 1) * sum(self.dilation_base ** i for i in range(self.num_layer))) / self.max_pooling_kernel_size)

            self.small_sub_receptive_field_size = int((self.window_size - self.sub_small_kernel_size + 2 * (self.sub_small_kernel_size - 1)) / self.max_pooling_kernel_size)
            self.big_sub_receptive_field_size = int((self.window_size - self.sub_big_kernel_size + 2 * (self.sub_big_kernel_size - 1)) / self.max_pooling_kernel_size)
            self.receptive_field_size = int(self.receptive_field_size + self.small_sub_receptive_field_size + self.big_sub_receptive_field_size)

        self.maxpool = nn.MaxPool1d(kernel_size=self.max_pooling_kernel_size)
        self.linear_1 = nn.Linear(self.receptive_field_size, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch = x.shape[0]
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
        out = self.relu(out)
        return out


class Movement(nn.Module):
    def __init__(self, num_feature, **param):
        super(Movement, self).__init__()
        self.__dict__.update(param)
        self.num_feature = num_feature
        self.autoencoder = Autoencoder(self.num_feature, self.window_size, **self.conv1D_param)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(self.drop_out)
        self.linear_1 = nn.Linear(self.lstm_hidden_layer_size * self.lstm_num_layer, 10)
        self.linear_2 = nn.Linear(10, 1)
        self.linear_3 = nn.Linear(5, 1)

        self.lstm = nn.LSTM(self.conv1D_param["output_size"], hidden_size=self.lstm_hidden_layer_size,
                            num_layers=self.lstm_num_layer,
                            batch_first=True)
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
        # Data extract
        x = self.autoencoder(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)
        x = self.linear_1(x)
        x = self.drop_out(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.drop_out(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.drop_out(x)
        x = self.sigmoid(x)
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
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x
