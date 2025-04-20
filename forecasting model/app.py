#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import base64
from pathlib import Path
import time

# Model imports
from models.xgboost_model import run_xgboost
from models.prophet_model import run_prophet
from models.lstm_model import run_lstm
from models.garch_lstm_model import run_garch_lstm

# ========== Background & Style Setup ==========
def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read())
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        html, body, .stApp {{
            font-family: 'Roboto', sans-serif;
            background-image: url("data:image/png;base64,{encoded_string.decode()}");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            color: #1c1c1c;
        }}

        .section {{
            background: rgba(255, 255, 255, 0.92);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
        }}

        .stButton > button {{
            background: linear-gradient(135deg, #2193b0, #6dd5ed);
            color: white;
            font-weight: 600;
            font-size: 16px;
            border: none;
            padding: 10px 20px;
            border-radius: 12px;
            cursor: pointer;
            transition: 0.3s ease;
        }}

        .stButton > button:hover {{
            background: linear-gradient(135deg, #6dd5ed, #2193b0);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }}

        .stSelectbox label, .stTextInput label, .stFileUploader label {{
            font-weight: bold;
            font-size: 16px;
            color: #003366;
        }}

        .stSelectbox div, .stTextInput input, .stDataFrame {{
            font-weight: 500;
            font-size: 14px;
        }}

        .stDataFrame tbody tr:hover {{
            background-color: #f5fafd;
        }}

        h1, h2, h3 {{
            color: #003366;
            font-weight: 700;
        }}

        .stSpinner > div {{
            color: #003366;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Set page config
st.set_page_config(page_title="📊 Stock Forecast Dashboard", layout="wide")

# Set background image
set_background("forecasting model/assets/iPad Stock Wallpaper.jpeg")  # Ensure this file exists

# ========== Header ==========
st.markdown('<div class="section">', unsafe_allow_html=True)
st.title("📈 Stock Forecast Comparison Across 4 Models")
st.markdown("Upload your stock price data (must include **'Date'** and **'Close'** columns).")
st.markdown('</div>', unsafe_allow_html=True)

# ========== File Upload ==========
uploaded_file = st.file_uploader("📤 Upload your stock data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    st.markdown('<div class="section">', unsafe_allow_html=True)

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = [col.strip() for col in df.columns]
    df.rename(columns={'Date ': 'Date', 'Close ': 'Close', 'Open ': 'Open', 'Shares Traded ': 'Shares Traded'}, inplace=True)
    try:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='raise')
    except Exception as e:
        st.error(f"❌ Failed to parse 'Date' column. Make sure it is in a consistent date format like 'DD-MMM-YYYY'. Error: {e}")

    st.success("✅ File uploaded and cleaned successfully!")

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== Run Models ==========
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.info("🔁 Running models...")

    model_outputs = []
    with st.spinner("Running models... Please wait!"):
        time.sleep(2)  # Simulating computation delay (replace with model execution)

        for func, name in [
            (run_xgboost, "XGBoost"), (run_prophet, "Prophet"),
             (run_lstm, "LSTM"),
             (run_garch_lstm, "GARCH+LSTM")
        ]:
            try:
                model_outputs.append(func(df.copy()))
            except Exception as e:
                st.warning(f"⚠️ {name} failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== Accuracy Table ==========
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📊 Model Accuracy Metrics")

    accuracy_data = [
        {
            "Model": m["name"],
            "RMSE": round(m["rmse"], 2),
            "MAE": round(m["mae"], 2),
            
        }
        for m in model_outputs
    ]
    accuracy_df = pd.DataFrame(accuracy_data)

    # Sort accuracy based on user input
    sort_column = st.selectbox("Sort results by:", ["RMSE", "MAE"])
    if sort_column == "RMSE":
        accuracy_df = accuracy_df.sort_values("RMSE")
    else :
        sort_column == "MAE"
        accuracy_df = accuracy_df.sort_values("MAE")

    st.dataframe(accuracy_df)
    st.markdown('</div>', unsafe_allow_html=True)

    # ========== Forecast Table ==========
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("📅 5-Day Forecast Table")

# Extract forecast dates
    forecast_dates = pd.Series(pd.to_datetime(model_outputs[0]["dates"])).dt.date


# Create forecast DataFrame
    forecast_df = pd.DataFrame({"Date": forecast_dates})

# Manually input actual values
    actual_values_list = [23742.90, 24188.65, 24004.75, 23616.05, 23707.90]
    forecast_df["Actual"] = actual_values_list

# Add forecasted values from each model
    for m in model_outputs:
        forecast_df[m["name"]] = m["forecast"]

# Format and show
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    forecast_df = forecast_df.set_index("Date")
    st.dataframe(forecast_df.style.format("{:.2f}"))
    st.markdown('</div>', unsafe_allow_html=True)


    # ========== Download Button ==========
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.markdown("📥 [Download the forecast results as CSV](data:file/csv;base64," + base64.b64encode(forecast_df.to_csv(index=False).encode()).decode() + ")")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("📁 Please upload a dataset to begin.")


