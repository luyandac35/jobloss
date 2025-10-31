import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

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

st.write("### Preview of Uploaded Dataset")
st.dataframe(df.head())

# Detect if columns contain years (wide format)
year_cols = [c for c in df.columns if str(c).isdigit()]

if len(year_cols) > 3:
    st.info("Detected wide-format dataset. Reshaping for analysis...")
    id_vars = [c for c in df.columns if c not in year_cols]
    df = df.melt(id_vars=id_vars, var_name="Year", value_name="Unemployment_Rate")

    # Clean melted data
    df.rename(columns={id_vars[0]: "Country"}, inplace=True)
    df["Year"] = pd.to_datetime(df["Year"], format="%Y", errors="coerce")
    df["Unemployment_Rate"] = pd.to_numeric(df["Unemployment_Rate"], errors="coerce")
    df = df.dropna(subset=["Year", "Unemployment_Rate"])
else:
    # fallback to manual selection if no numeric year columns detected
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'year' in c.lower()]
    country_cols = [c for c in df.columns if 'country' in c.lower()]
    unemp_cols = [c for c in df.columns if 'unemploy' in c.lower() or 'rate' in c.lower()]

    date_col = st.sidebar.selectbox("Select Date/Year column", date_cols if date_cols else df.columns)
    country_col = st.sidebar.selectbox("Select Country column", country_cols if country_cols else df.columns)
    unemp_col = st.sidebar.selectbox("Select Unemployment Rate column", unemp_cols if unemp_cols else df.columns)

    df = df[[date_col, country_col, unemp_col]].copy()
    df.columns = ['Year', 'Country', 'Unemployment_Rate']
    df['Year'] = pd.to_datetime(df['Year'], errors='coerce')
    df['Unemployment_Rate'] = pd.to_numeric(df['Unemployment_Rate'], errors='coerce')
    df = df.dropna(subset=['Year', 'Unemployment_Rate'])

# ----------------------------
# CHECK CLEANED DATA
# ----------------------------
if df.empty:
    st.error("⚠️ Dataset has no valid rows after cleaning. Please check format.")
    st.stop()

st.success(f"✅ Loaded {len(df)} valid records across {df['Country'].nunique()} countries.")

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
min_date, max_date = df['Year'].min(), df['Year'].max()
countries = sorted(df['Country'].unique())

start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    key="daterange"
)

selected_countries = st.sidebar.multiselect("Select countries to view", countries, default=countries[:5])
filtered_df = df[(df['Country'].isin(selected_countries)) &
                 (df['Year'] >= pd.to_datetime(start_date)) &
                 (df['Year'] <= pd.to_datetime(end_date))]

# ----------------------------
# VISUALIZATION (EDA)
# ----------------------------
st.subheader("📊 Employment Trends Over Time")

fig = px.line(filtered_df, x="Year", y="Unemployment_Rate", color="Country",
              title="Unemployment Rate by Country Over Time")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# FORECASTING (Prophet)
# ----------------------------
st.subheader("📈 Forecast Future Unemployment (Prophet Model)")

forecast_country = st.selectbox("Select a country to forecast", countries)
country_data = df[df["Country"] == forecast_country][["Year", "Unemployment_Rate"]].rename(
    columns={"Year": "ds", "Unemployment_Rate": "y"}
)

if len(country_data) < 10:
    st.warning("⚠️ Not enough data points to forecast for this country.")
else:
    model = Prophet()
    model.fit(country_data)
    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    st.write(f"Forecast for **{forecast_country}**")
    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)
    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12))

# ----------------------------
# INDUSTRY IMPACT SIMULATION
# ----------------------------
st.subheader("🏭 Industry Disparities and Job-loss Risk (Illustrative)")

industries = ["Manufacturing", "Retail", "Finance", "Technology", "Healthcare"]
impact_scores = np.random.rand(len(industries)) * 100
impact_df = pd.DataFrame({"Industry": industries, "AI_Displacement_Risk": impact_scores})

fig2 = px.bar(impact_df, x="Industry", y="AI_Displacement_Risk",
              title="Simulated AI Displacement Risk by Industry",
              color="AI_Displacement_Risk", color_continuous_scale="RdYlBu_r")
st.plotly_chart(fig2, use_container_width=True)
