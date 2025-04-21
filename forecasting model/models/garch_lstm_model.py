#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

# Preparing the GARCH + LSTM function version from the user's script
def run_garch_lstm(df):
    df.columns = df.columns.str.strip()
    df.sort_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.set_index('Date', inplace=True)
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)

    # Split data
    train_data = df[:-5].copy()
    test_data = df[-5:].copy()

    # Rolling GARCH Volatility for Training Data
    rolling_vols = []
    for i in range(45, len(train_data)):
        sub_returns = train_data['returns'].iloc[:i] * 100
        garch_model = arch_model(sub_returns, vol='GARCH', p=1, q=1)
        garch_fit = garch_model.fit(disp='off')
        forecast = garch_fit.forecast(horizon=1).variance.values[-1, 0]
        vol = np.sqrt(forecast) / 100
        rolling_vols.append(vol)

    # Trim train data to align with rolling volatility
    train_data_trimmed = train_data.iloc[45:].copy()
    train_data_trimmed['volatility'] = rolling_vols

    # Scaling returns and volatility
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(train_data_trimmed[['returns', 'volatility']])

    # Sequence generator
    def create_sequences(data, length=45):
        X, y = [], []
        for i in range(len(data) - length):
            X.append(data[i:i + length])
            y.append(data[i + length][0])  # predicting returns
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_features)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Generate forecast for next 5 days
    # Get last known 45 days' returns and volatility using full df
    full_df = df.copy()

    # Final GARCH volatility for prediction period
    sub_returns_full = full_df['returns'] * 100
    garch_model_full = arch_model(sub_returns_full, vol='GARCH', p=1, q=1)
    garch_fit_full = garch_model_full.fit(disp='off')
    forecast_vol_full = garch_fit_full.forecast(horizon=5).variance.values[-1]
    volatility_forecast = np.sqrt(forecast_vol_full) / 100

    # Prepare last 45-day sequence for prediction
    last_returns = full_df['returns'].iloc[-45:].values.reshape(-1, 1)
    last_vols = volatility_forecast[0] * np.ones((45, 1))  # broadcast average predicted vol
    last_input = np.hstack((last_returns, last_vols))

    last_input_scaled = scaler.transform(last_input)

    predicted_returns = []
    current_seq = last_input_scaled

    for _ in range(5):
        pred = model.predict(current_seq.reshape(1, 45, 2), verbose=0)[0, 0]
        predicted_returns.append(pred)
        next_input = np.append(current_seq[1:], [[pred, current_seq[-1, 1]]], axis=0)
        current_seq = next_input

    predicted_returns = scaler.inverse_transform(np.column_stack((predicted_returns, current_seq[-5:, 1])))[:, 0]

    # Convert predicted returns to prices
    last_price = full_df['Close'].iloc[-1]
    predicted_prices = []
    for ret in predicted_returns:
        next_price = last_price * np.exp(ret)
        predicted_prices.append(next_price)
        last_price = next_price

    # Evaluate on test set
    actual_prices = test_data['Close'].values
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)

    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')

    return {
        "name": "GARCH + LSTM (Rolling Vol)",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "forecast": predicted_prices,
        "dates": forecast_dates
    }
