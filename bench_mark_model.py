import torch
import numpy as np
import statsmodels.api as sm

def bench_mark_arima(X_train, y_train, X_val, y_val, X_test, y_test):
    
    # Define the order of the ARIMA model
    order = (1, 1, 1)

    # Train the ARIMA model on the training set
    model = sm.tsa.ARIMA(y_train, order=order).fit()

    # Make predictions on the validation and test sets
    y_val_pred = model.predict(start=X_val.index[0], end=X_val.index[-1], dynamic=False)
    y_test_pred = model.predict(start=X_test.index[0], end=X_test.index[-1], dynamic=False)

    # Calculate the evaluation metrics
    val_rmse = torch.sqrt(torch.mean((y_val - y_val_pred)**2))
    test_rmse = torch.sqrt(torch.mean((y_test - y_test_pred)**2))

    return val_rmse, test_rmse

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