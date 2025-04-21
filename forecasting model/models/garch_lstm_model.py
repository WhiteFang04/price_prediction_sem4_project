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

def run_garch_lstm(df):
    from arch import arch_model
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from keras.models import Sequential
    from keras.layers import LSTM, Dropout, Dense
    from keras.optimizers import Adam
    import pandas as pd

    df.columns = df.columns.str.strip()
    df.sort_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.set_index('Date', inplace=True)
    df['returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df.dropna(inplace=True)

    train_data = df[:-5]
    test_data = df[-5:]

    garch = arch_model(train_data['returns'] * 100, vol='GARCH', p=1, q=1)
    garch_fit = garch.fit(disp="off")
    forecast_vol = garch_fit.forecast(horizon=5).variance.values[-1]
    volatility = np.sqrt(forecast_vol) / 100

    scaler = MinMaxScaler()
    scaled_returns = scaler.fit_transform(train_data[['returns']])

    def create_sequences(data, length=60):
        X, y = [], []
        for i in range(len(data) - length):
            X.append(data[i:i + length])
            y.append(data[i + length])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_returns)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    scaled_test_returns = scaler.transform(test_data[['returns']])
    X_test, y_test = create_sequences(scaled_test_returns)

    predicted_returns = []
    last_seq = scaled_returns[-60:]
    for _ in range(5):
        pred = model.predict(last_seq.reshape(1, 60, 1), verbose=0)[0, 0]
        predicted_returns.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    predicted_returns = scaler.inverse_transform(np.array(predicted_returns).reshape(-1, 1))
    last_price = df['Close'].iloc[-6]
    predicted_prices = []
    for i in range(5):
        next_price = last_price * np.exp(predicted_returns[i][0])
        predicted_prices.append(next_price)
        last_price = next_price

    actual_prices = test_data['Close'].values
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)

    # Forecast next 5 days using full data
    garch_full = arch_model(df['returns'] * 100, vol='GARCH', p=1, q=1)
    garch_full_fit = garch_full.fit(disp="off")
    forecast_vol_full = garch_full_fit.forecast(horizon=5).variance.values[-1]
    volatility_full = np.sqrt(forecast_vol_full) / 100

    scaled_returns_full = scaler.fit_transform(df[['returns']])
    X_full, _ = create_sequences(scaled_returns_full)
    last_seq_full = scaled_returns_full[-60:]

    predicted_returns_full = []
    for _ in range(5):
        pred = model.predict(last_seq_full.reshape(1, 60, 1), verbose=0)[0, 0]
        predicted_returns_full.append(pred)
        last_seq_full = np.append(last_seq_full[1:], [[pred]], axis=0)

    predicted_returns_full = scaler.inverse_transform(np.array(predicted_returns_full).reshape(-1, 1))
    last_price_full = df['Close'].iloc[-1]
    predicted_prices_full = []
    for i in range(5):
        next_price = last_price_full * np.exp(predicted_returns_full[i][0])
        predicted_prices_full.append(next_price)
        last_price_full = next_price

    forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=5, freq='B')

    return {
        "name": "GARCH + LSTM",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "forecast": predicted_prices_full,
        "dates": forecast_dates
    }

