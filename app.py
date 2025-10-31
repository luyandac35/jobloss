# ============================================
# 🌍 Global Unemployment Dashboard & Forecast
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly

# --- 🧩 Compatibility Fix for NumPy >= 2.0 ---
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool

# --- 🌟 Streamlit Page Setup ---
st.set_page_config(page_title="Global Unemployment Dashboard", layout="wide", page_icon="📈")

# --- 🧭 Header Section ---
st.title("📊 Global Unemployment Analysis & Forecasting Dashboard")
st.markdown("""
This interactive dashboard visualizes **global unemployment trends** and uses 
**Prophet** to forecast future unemployment rates.
""")

# --- 📤 Upload or Use Default Dataset ---
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("global_unemployment_data.csv")

st.subheader("🗂 Dataset Preview")
st.dataframe(df.head())

# --- 🧹 Basic Cleaning ---
df.columns = df.columns.str.strip()
if 'Year' in df.columns:
    df['Year'] = pd.to_datetime(df['Year'], errors='coerce')

# --- 🌍 Country Selector ---
countries = df['Country Name'].unique().tolist()
selected_country = st.selectbox("Select a Country", countries, index=0)

country_data = df[df['Country Name'] == selected_country]

st.markdown(f"### 📅 Historical Unemployment Rate for **{selected_country}**")

# --- 📈 Line Chart (Historical Trend) ---
fig_line = px.line(
    country_data,
    x='Year',
    y='Unemployment Rate',
    title=f"{selected_country} — Unemployment Rate Over Time",
    markers=True
)
fig_line.update_traces(line_color='#007BFF')
st.plotly_chart(fig_line, use_container_width=True)

# --- 🥧 Pie Chart (Regional or Gender Split, if available) ---
if 'Gender' in df.columns:
    gender_data = df[df['Country Name'] == selected_country].groupby('Gender')['Unemployment Rate'].mean().reset_index()
    fig_pie = px.pie(gender_data, values='Unemployment Rate', names='Gender', title="Unemployment by Gender")
    st.plotly_chart(fig_pie, use_container_width=True)

# --- 🧾 Summary Statistics ---
st.subheader("📋 Summary Statistics")
st.write(country_data.describe())

# --- 🔮 Prophet Forecasting ---
st.markdown("## 🔮 Forecasting Future Unemployment Trends")

# Prepare data for Prophet
forecast_df = country_data[['Year', 'Unemployment Rate']].rename(columns={'Year': 'ds', 'Unemployment Rate': 'y'})
forecast_df = forecast_df.dropna()
forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])

if not forecast_df.empty:
    model = Prophet(yearly_seasonality=True)
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)

    # Display forecast data
    st.write("### Forecast Results (Next 5 Years)")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Plot forecast
    fig_forecast = plot_plotly(model, forecast)
    fig_forecast.update_layout(title="Forecasted Unemployment Rate", xaxis_title="Year", yaxis_title="Predicted Rate (%)")
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Components (Trend, Yearly)
    st.write("### Forecast Components")
    st.pyplot(model.plot_components(forecast))
else:
    st.warning("No valid data available for forecasting. Please check the dataset format.")

# --- 🧠 Footer ---
st.markdown("""
---
✅ **Developed by Simba | Powered by Prophet, Plotly & Streamlit**  
📅 Data Source: Global Unemployment Dataset
""")
