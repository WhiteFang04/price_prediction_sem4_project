#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_prophet(df):
    df.columns = [col.strip() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna()
    df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    model = Prophet(seasonality_mode='multiplicative')
    model.fit(df[['ds', 'y']])
    forecast = model.make_future_dataframe(5)
    prediction = model.predict(forecast)

    df_merged = df.merge(prediction[['ds', 'yhat']], on='ds', how='inner')

    actual = df_merged['y']
    pred = df_merged['yhat']  # last predicted points matching actual length

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)

    return {
        "name": "Prophet",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "forecast": prediction['yhat'][-5:].values,
        "dates": prediction['ds'][-5:].dt.date
    }

