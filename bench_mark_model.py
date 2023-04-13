import torch
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from config import config as cf
from keras.optimizers import Adam
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def bench_mark_random_forest(X_train, y_train, X_val, y_val, X_test, y_test):


    # create random forest regressor object
    rf = RandomForestRegressor()

    # train the regressor using the training data
    rf.fit(X_train, y_train)

    # evaluate the regressor on the validation data
    val_error = mean_squared_error(y_val, rf.predict(X_val))

    # evaluate the regressor on the test data
    test_error = mean_squared_error(y_test, rf.predict(X_test))

def create_lstm_model(X_train, y_train, X_val, y_val, X_test, y_test):
    model = Sequential()

    # Add LSTM layers
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=True, activation='relu'))
    # Add output layer
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    # Print model summary
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mean_absolute_error'])

    history_LSTM = model.fit(X_train, y_train, epochs=100, batch_size=cf['training']['movement_3']['batch_size'], validation_data=(X_val, y_val))
    loss, _ = model.evaluate(X_test, y_test)
    print('Test Loss MSE: ', loss)
    return model, history_LSTM, loss
