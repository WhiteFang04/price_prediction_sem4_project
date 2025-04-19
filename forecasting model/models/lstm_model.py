#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

def run_lstm(df):
    df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y')
    data = df[['Close']].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    seq_length = 45
    X, y = [], []
    for i in range(seq_length, len(data_scaled)):
        X.append(data_scaled[i-seq_length:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    pred_scaled = model.predict(X_test)
    y_test_actual = scaler.inverse_transform(y_test)
    pred_actual = scaler.inverse_transform(pred_scaled)

    rmse = np.sqrt(mean_squared_error(y_test_actual, pred_actual))
    mae = mean_absolute_error(y_test_actual, pred_actual)
    r2 = r2_score(y_test_actual, pred_actual)

    last_seq = data_scaled[-seq_length:]
    future = []
    for _ in range(5):
        prediction = model.predict(last_seq.reshape(1, seq_length, 1))[0, 0]
        future.append(prediction)
        last_seq = np.append(last_seq[1:], [[prediction]], axis=0)

    forecast = scaler.inverse_transform(np.array(future).reshape(-1, 1)).flatten()
    dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=5, freq='B')

    return {
        "name": "LSTM",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "forecast": forecast,
        "dates": dates
    }

