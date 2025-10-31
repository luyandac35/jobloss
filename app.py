import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="🏥 NHI / Employment Dashboard", layout="wide")

st.title("🏥 NHI / Employment Dashboard — Forecasting & Job-loss Risk")
st.write("""
This dashboard analyzes global unemployment data, predicts job-loss risk (classification),
forecasts employment trends (Prophet), and highlights industry disparities in AI-driven displacement.
""")

# ----------------------------
# LOAD DATASET
# ----------------------------
uploaded_file = "global_unemployment_data.csv"
df = pd.read_csv(uploaded_file)

st.sidebar.subheader("Step 1 — Dataset Overview")

st.write("### Preview of Uploaded Dataset")
st.dataframe(df.head())

# ----------------------------
# DETECT WIDE FORMAT (YEARS AS COLUMNS)
# ----------------------------
year_cols = [c for c in df.columns if str(c).isdigit()]

if len(year_cols) > 3:
    st.info("Detected wide-format dataset. Reshaping for analysis...")
    id_vars = [c for c in df.columns if c not in year_cols]
    df = df.melt(id_vars=id_vars, var_name="Year", value_name="Unemployment_Rate")

    # Rename and clean
    df.rename(columns={id_vars[0]: "Country"}, inplace=True)
    df["Year"] = pd.to_datetime(df["Year"], format="%Y", errors="coerce")
    df["Unemployment_Rate"] = pd.to_numeric(df["Unemployment_Rate"], errors="coerce")
    df["Country"] = df["Country"].astype(str).replace("nan", np.nan)
    df = df.dropna(subset=["Year", "Unemployment_Rate", "Country"])
else:
    # fallback manual selection
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
    df['Country'] = df['Country'].astype(str).replace("nan", np.nan)
    df = df.dropna(subset=['Year', 'Unemployment_Rate', 'Country'])

# ----------------------------
# SAFETY CLEANUP
# ----------------------------
if df.empty:
    st.error("⚠️ Dataset has no valid rows after cleaning. Please check format.")
    st.stop()

# Convert all country names to string safely
df['Country'] = df['Country'].astype(str)

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
min_date, max_date = df['Year'].min(), df['Year'].max()
countries = sorted([c for c in df['Country'].unique() if isinstance(c, str)])

st.success(f"✅ Loaded {len(df)} valid records across {len(countries)} countries.")

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
# TASK 2 — FORECASTING (PROPHET)
# ----------------------------
st.subheader("📈 Task 2: Forecasting Employment Trends (Prophet)")

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
# TASK 1 — CLASSIFICATION (PREDICTING JOB LOSS RISK)
# ----------------------------
st.subheader("🤖 Task 1: Predicting Job-Loss Risk")

# Synthetic example classification
st.write("This simplified model classifies countries as 'High Risk' or 'Low Risk' based on unemployment trends.")

df_class = df.groupby("Country")["Unemployment_Rate"].agg(["mean", "std", "max", "min"]).reset_index()
df_class["Risk_Level"] = np.where(df_class["mean"] > df_class["mean"].median(), 1, 0)

X = df_class[["mean", "std", "max", "min"]]
y = df_class["Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

st.write(f"**Model Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

fig_risk = px.bar(df_class.sort_values("mean", ascending=False).head(20),
                  x="Country", y="mean", color="Risk_Level",
                  color_discrete_map={1: "red", 0: "green"},
                  title="Top 20 Countries by Average Unemployment Risk")
st.plotly_chart(fig_risk, use_container_width=True)

# ----------------------------
# TASK 3 — INDUSTRY IMPACT SIMULATION
# ----------------------------
st.subheader("🏭 Task 3: Analysing Industry Impact (AI Displacement Risk)")

industries = ["Manufacturing", "Retail", "Finance", "Technology", "Healthcare", "Education"]
impact_scores = np.random.uniform(30, 90, len(industries))

impact_df = pd.DataFrame({
    "Industry": industries,
    "AI_Displacement_Risk": impact_scores
})

fig2 = px.bar(impact_df, x="Industry", y="AI_Displacement_Risk",
              title="Simulated AI Displacement Risk by Industry",
              color="AI_Displacement_Risk", color_continuous_scale="YlOrRd")
st.plotly_chart(fig2, use_container_width=True)

st.write("""
### 🧭 Interpretation
- **High-risk industries** may require **reskilling programs** or **AI integration strategies**.
- **Forecast insights** support long-term **policy planning and employment resilience**.
""")
