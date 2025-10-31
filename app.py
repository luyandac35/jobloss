import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# STREAMLIT APP CONFIGURATION
# ----------------------------
st.set_page_config(page_title="🏥 NHI / Employment Dashboard", layout="wide")

st.title("🏥 NHI / Employment Dashboard — Forecasting & Job-loss Risk")
st.write("""
This app analyzes global unemployment data, predicts job-loss risk (classification), 
forecasts trends (Prophet/ARIMA fallback), and highlights industry impact.
""")

# ----------------------------
# LOAD DATASET
# ----------------------------
uploaded_file = "global_unemployment_data.csv"
df = pd.read_csv(uploaded_file)

st.sidebar.subheader("Step 1 — Dataset Overview")

# Automatically detect possible date/year column
date_cols = [c for c in df.columns if 'date' in c.lower() or 'year' in c.lower()]
country_cols = [c for c in df.columns if 'country' in c.lower()]
unemployment_cols = [c for c in df.columns if 'unemploy' in c.lower() or 'rate' in c.lower()]

date_col = st.sidebar.selectbox("Select Date/Year column", date_cols if date_cols else df.columns)
country_col = st.sidebar.selectbox("Select Country column", country_cols if country_cols else df.columns)
unemp_col = st.sidebar.selectbox("Select Unemployment Rate column", unemployment_cols if unemployment_cols else df.columns)

# Clean data robustly
df = df[[date_col, country_col, unemp_col]].copy()
df.columns = ['Date', 'Country', 'Unemployment_Rate']

# Try to parse dates safely
try:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
except Exception:
    df['Date'] = pd.to_datetime(df['Date'], format='%Y', errors='coerce')

df = df.dropna(subset=['Date'])
df['Unemployment_Rate'] = pd.to_numeric(df['Unemployment_Rate'], errors='coerce')
df = df.dropna(subset=['Unemployment_Rate'])

# Avoid empty dataset
if df.empty:
    st.error("⚠️ Your dataset has no valid rows after cleaning. Please check the selected columns.")
    st.stop()

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
min_date, max_date = df['Date'].min(), df['Date'].max()

# Fallback if no valid dates
if pd.isna(min_date) or pd.isna(max_date):
    min_date, max_date = pd.Timestamp("2010-01-01"), pd.Timestamp("2025-12-31")

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="daterange"
)

filtered_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

countries = sorted(filtered_df['Country'].unique())
selected_countries = st.sidebar.multiselect("Select countries to view", countries, default=countries[:3])

filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

st.write(f"✅ Data loaded with **{len(filtered_df)}** records between {start_date} and {end_date}.")

# ----------------------------
# VISUALIZATION (EDA)
# ----------------------------
st.subheader("📊 Employment Trends Over Time")

fig = px.line(filtered_df, x="Date", y="Unemployment_Rate", color="Country",
              title="Unemployment Rate by Country Over Time")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# FORECASTING (Prophet)
# ----------------------------
st.subheader("📈 Forecast Future Unemployment (Prophet Model)")

forecast_country = st.selectbox("Select a country to forecast", countries)

country_data = df[df['Country'] == forecast_country][['Date', 'Unemployment_Rate']].rename(
    columns={"Date": "ds", "Unemployment_Rate": "y"}
)

# Train Prophet model safely
if len(country_data) < 10:
    st.warning("⚠️ Not enough data points to forecast for this country.")
else:
    model = Prophet()
    model.fit(country_data)

    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    st.write("**Forecast Summary:**")
    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12))

# ----------------------------
# INDUSTRY IMPACT (Simulated Example)
# ----------------------------
st.subheader("🏭 Industry Disparities and Job-loss Risk (Illustrative)")

industries = ["Manufacturing", "Retail", "Finance", "Technology", "Healthcare"]
impact_scores = np.random.rand(len(industries)) * 100

impact_df = pd.DataFrame({"Industry": industries, "AI_Displacement_Risk": impact_scores})

fig2 = px.bar(impact_df, x="Industry", y="AI_Displacement_Risk",
              title="Simulated AI Displacement Risk by Industry",
              color="AI_Displacement_Risk", color_continuous_scale="RdYlBu_r")
st.plotly_chart(fig2, use_container_width=True)

st.info("💡 This dashboard can be extended with actual industry-level AI displacement data for policy insights.")
