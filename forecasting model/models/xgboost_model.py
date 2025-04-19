#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def run_xgboost(df):
    df['Date'] = pd.to_datetime(df['Date'],format='%d-%m-%Y')
    df.columns = [col.strip() for col in df.columns]
    df = df.sort_values('Date')

    def create_features(df, lags=5):
        for i in range(1, lags + 1):
            df[f'lag_{i}'] = df['Close'].shift(i)
        df['rolling_mean_3'] = df['Close'].rolling(window=3).mean()
        df['rolling_std_3'] = df['Close'].rolling(window=3).std()
        df['rolling_mean_7'] = df['Close'].rolling(window=7).mean()
        df['rolling_std_7'] = df['Close'].rolling(window=7).std()
        df['dayofweek'] = df['Date'].dt.dayofweek
        return df

    df = create_features(df)
    df.dropna(inplace=True)

    features = [col for col in df.columns if 'lag_' in col or 'rolling' in col or col == 'dayofweek']
    X = df[features]
    y = df['Close']

    split_date = df['Date'].iloc[-30]
    train = df[df['Date'] < split_date]
    test = df[df['Date'] >= split_date]

    X_train, y_train = train[features], train['Close']
    X_test, y_test = test[features], test['Close']

    model = XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))

    # Forecast next 5 days
    last_known = df.copy()
    forecast_days = 5
    future_predictions = []
    for i in range(forecast_days):
        last_row = last_known.iloc[-1:].copy()
        next_date = last_row['Date'].values[0] + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)

        new_row = {'Date': next_date}
        for j in range(1, 6):
            new_row[f'lag_{j}'] = last_known['Close'].iloc[-j]
        new_row['rolling_mean_3'] = last_known['Close'].iloc[-3:].mean()
        new_row['rolling_std_3'] = last_known['Close'].iloc[-3:].std()
        new_row['rolling_mean_7'] = last_known['Close'].iloc[-7:].mean()
        new_row['rolling_std_7'] = last_known['Close'].iloc[-7:].std()
        new_row['dayofweek'] = next_date.weekday()

        X_future = pd.DataFrame([new_row])[features]
        predicted_close = model.predict(X_future)[0]

        new_row['Close'] = predicted_close
        future_predictions.append(predicted_close)
        last_known = pd.concat([last_known, pd.DataFrame([new_row])], ignore_index=True)

    forecast_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=5, freq='B')

    return {
        "name": "XGBoost",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "forecast": future_predictions,
        "dates": forecast_dates
    }

