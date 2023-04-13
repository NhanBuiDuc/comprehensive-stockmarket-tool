import torch
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
def bench_mark_random_forest(dataset_train, dataset_val, dataset_test):
    pass
def create_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test, input_shape, output_size):
    model = Sequential()

    # Add LSTM layers
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))

    # Add output layer
    model.add(Dense(output_size))
    # Print model summary
    model.summary()
    model.compile(optimizer='adam',
                loss='mean_squared_error')

    history_LSTM = model.fit(X_train, y_train, epochs=100, batch_size=cf['training']['movement_3']['batch_size'], validation_data=(X_val, y_val))
    loss, _ = model.evaluate(X_test, y_test)
    print('Test Loss MSE: ', loss)
    return model, history_LSTM, loss