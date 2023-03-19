from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
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
        self.scaler = MaxAbsScaler()
        self.x = x.astype(np.float64)
        # Reshape the data
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        # Apply the scaler
        x = self.scaler.fit_transform(x)
        self.y = self.scaler.fit_transform(y)
        # Reshape the scaled data back to the original shape
        self.x = x.reshape((self.x.shape[0], self.x.shape[1], self.x.shape[2]))
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
    
class Classification_TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.scaler = MaxAbsScaler()
        self.y = y.astype(np.float32)
        self.x = x.astype(np.float64)
        # Reshape the data
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2]))
        # Apply the scaler
        x = self.scaler.fit_transform(x)
        # Reshape the scaled data back to the original shape
        self.x = x.reshape((self.x.shape[0], self.x.shape[1], self.x.shape[2]))
        self.x = self.x.astype(np.float32)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])