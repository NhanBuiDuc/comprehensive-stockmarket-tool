import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load and preprocess data
data = pd.read_csv("stock_prices.csv")
close_price = data["Close"].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close_price = scaler.fit_transform(close_price)

# Split data into train and test sets
train_size = int(len(scaled_close_price) * 0.7)
test_size = len(scaled_close_price) - train_size
train_data = scaled_close_price[0:train_size,:]
test_data = scaled_close_price[train_size:len(scaled_close_price),:]

# Prepare data for LSTM input
def create_lstm_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 7
train_X, train_Y = create_lstm_dataset(train_data, look_back)
test_X, test_Y = create_lstm_dataset(test_data, look_back)

# Reshape data for LSTM input
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

# Define LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train LSTM model
model.fit(train_X, train_Y, epochs=100, batch_size=1, verbose=2)

# Make predictions on test data
test_predictions = model.predict(test_X)

# Inverse scaling of predicted and actual values
test_predictions = scaler.inverse_transform(test_predictions)
test_Y = scaler.inverse_transform(test_Y.reshape(-1,1))

# Calculate RMSE
rmse = np.sqrt(np.mean(((test_predictions - test_Y) ** 2)))
print("Test RMSE:", rmse)
