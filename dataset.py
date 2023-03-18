from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

# normalize
# scaler = Normalizer()
# normalized_data_close_price = scaler.fit_transform(data_close_price)

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.scaler = Normalizer()
        x = np.expand_dims(x, 2) # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.x = self.scaler.fit_transform(x)
        self.y = self.scaler.fit_transform(y)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])