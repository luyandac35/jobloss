# ============================================
# Streamlit Dashboard - Unemployment & NHI Project
# - Auto-loads global_unemployment_data.csv (if present)
# - Task1: Job Loss Risk (classification)
# - Task2: Employment Forecasting (Prophet with ARIMA fallback)
# - Task3: Industry Impact & segmentation
# ============================================

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------
# Compatibility patch (NumPy aliases)
# Must run before importing prophet in environments with numpy>=2.0
# ---------------------------
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int64
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool

# Try to import Prophet; if fails we'll fallback to ARIMA
use_prophet = True
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
except Exception as e:
    use_prophet = False
    # We'll use statsmodels ARIMA as fallback for forecasting
    from statsmodels.tsa.arima.model import ARIMA

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="NHI - Unemployment Dashboard", layout="wide")
st.title("🏥 NHI / Employment Dashboard — Forecasting & Job-loss Risk")
st.markdown(
    "This app analyzes global unemployment data, predicts job-loss risk (classification), "
    "forecasts trends (Prophet or ARIMA fallback), and highlights industry impact."
)

# ---------------------------
# Load dataset automatically if present, otherwise allow upload
# ---------------------------
DEFAULT_PATH = "global_unemployment_data.csv"

def load_data():
    if os.path.exists(DEFAULT_PATH):
        df = pd.read_csv(DEFAULT_PATH)
        st.sidebar.success(f"Loaded dataset from `{DEFAULT_PATH}`")
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.sidebar.success("Uploaded dataset")
        else:
            st.sidebar.info(f"No `{DEFAULT_PATH}` found. Upload a CSV to use the app.")
            st.stop()
    return df

df = load_data()

# ---------------------------
# Quick column normalization
# ---------------------------
df.columns = df.columns.str.strip()
st.sidebar.write("Columns detected:")
st.sidebar.write(list(df.columns))

# Basic column inference
# We'll try to detect: year/date, country, unemployment rate, industry, region, gender, age group
possible_date_cols = [c for c in df.columns if 'year' in c.lower() or 'date' in c.lower()]
possible_country_cols = [c for c in df.columns if 'country' in c.lower() or 'iso' in c.lower()]
possible_unemp_cols = [c for c in df.columns if 'unemploy' in c.lower() or 'unemp' in c.lower() or 'rate' in c.lower()]

# Heuristic column selection
date_col = possible_date_cols[0] if possible_date_cols else None
country_col = possible_country_cols[0] if possible_country_cols else None
unemp_col = possible_unemp_cols[0] if possible_unemp_cols else None

# If the user has unusual column names, allow them to select
with st.sidebar.expander("Column mapping (change if auto-detect wrong)"):
    date_col = st.selectbox("Date / Year column", options=[None] + list(df.columns), index=(1 if date_col else 0))
    country_col = st.selectbox("Country column", options=[None] + list(df.columns), index=(1 if country_col else 0))
    unemp_col = st.selectbox("Unemployment rate column", options=[None] + list(df.columns), index=(1 if unemp_col else 0))
    industry_col = st.selectbox("Industry column (optional)", options=[None] + list(df.columns))
    region_col = st.selectbox("Region column (optional)", options=[None] + list(df.columns))
    gender_col = st.selectbox("Gender column (optional)", options=[None] + list(df.columns))

# Validate key columns
if date_col is None or unemp_col is None or country_col is None:
    st.warning("Please map Date/Year, Country and Unemployment rate columns in the sidebar to proceed.")
    st.stop()

# ---------------------------
# Preprocess dataset for analysis
# ---------------------------
df_local = df.copy()

# Normalize date column to datetime (if it's year integer, convert to string then to datetime)
try:
    df_local[date_col] = pd.to_datetime(df_local[date_col], errors='coerce')
except Exception:
    # try interpret as year numbers
    try:
        df_local[date_col] = pd.to_datetime(df_local[date_col].astype(str) + "-01-01", errors='coerce')
    except Exception:
        pass

# Clean unemployment numeric column
df_local[unemp_col] = pd.to_numeric(df_local[unemp_col], errors='coerce')

# Drop rows lacking required fields
df_local = df_local.dropna(subset=[date_col, country_col, unemp_col]).reset_index(drop=True)

st.sidebar.write(f"Rows after cleaning: {len(df_local)}")

# ---------------------------
# Sidebar filters (country, region, year range)
# ---------------------------
countries = sorted(df_local[country_col].dropna().unique().astype(str))
selected_country = st.sidebar.selectbox("Country", options=countries, index=0)

min_date = df_local[date_col].min()
max_date = df_local[date_col].max()
start_date, end_date = st.sidebar.date_input("Date range", value=(min_date, max_date), key="daterange")

# Filter data for selected country and date range
mask = (df_local[country_col].astype(str) == str(selected_country)) & (df_local[date_col] >= pd.to_datetime(start_date)) & (df_local[date_col] <= pd.to_datetime(end_date))
country_data = df_local[mask].sort_values(by=date_col).reset_index(drop=True)

st.markdown(f"## 📍 Selected: {selected_country} — {country_data.shape[0]} records")

# ---------------------------
# Tabs for 3 tasks
# ---------------------------
tab1, tab2, tab3 = st.tabs(["📊 Job-loss Risk (Classification)", "📈 Employment Forecast (Time Series)", "🏭 Industry Impact"])

# ---------------------------
# TASK 1: Classification — create safe binary target
# Approach:
# - Create 'risk' label = 1 if unemployment rate is high or increased significantly YOY
# - This is deterministic and safe (no fabricated labels)
# ---------------------------
with tab1:
    st.header("Task 1 — Predicting Job-loss Risk (Classification)")

    st.markdown("""
    **Target creation rule (deterministic):**  
    A row is labeled **high risk (1)** if either:
    - Current unemployment rate > (median + 0.5 * IQR) for the country (statistical high), **or**
    - Year-over-year increase in unemployment rate > 1.0 percentage point (significant rise).
    Otherwise labeled **low risk (0)**.
    """)

    # Prepare features for modeling using available numeric columns
    df_clf = country_data.copy()
    df_clf = df_clf.sort_values(by=date_col).reset_index(drop=True)

    # Compute YOY change if there is a prior year
    df_clf['unemp_prev'] = df_clf[unemp_col].shift(1)
    df_clf['yoy_change'] = df_clf[unemp_col] - df_clf['unemp_prev']

    # Compute country-level thresholds
    med = df_clf[unemp_col].median()
    q1 = df_clf[unemp_col].quantile(0.25)
    q3 = df_clf[unemp_col].quantile(0.75)
    iqr = q3 - q1
    high_threshold = med + 0.5 * iqr

    # create label
    df_clf['risk_label'] = 0
    df_clf.loc[(df_clf[unemp_col] > high_threshold) | (df_clf['yoy_change'] > 1.0), 'risk_label'] = 1
    df_clf = df_clf.dropna(subset=[unemp_col])  # ensure no NaNs

    st.write("Label distribution (low=0, high=1):")
    st.write(df_clf['risk_label'].value_counts().to_frame("count"))

    # Feature selection: use numeric columns except target and date
    numeric_cols = df_clf.select_dtypes(include=[np.number]).columns.tolist()
    # exclude risk and prev, and potential leakage
    exclude = [unemp_col, 'unemp_prev', 'yoy_change', 'risk_label']
    features = [c for c in numeric_cols if c not in exclude]

    if not features:
        st.warning("No numeric features available for classification. The dataset may need feature engineering.")
    else:
        st.write("Features used for classification:", features)

        # Fill missing numeric with median
        df_clf[features] = df_clf[features].fillna(df_clf[features].median())

        # Optional: encode categorical features if we want to include them
        cat_features = []
        possible_cat = []
        for c in [industry_col, region_col, gender_col]:
            if c and c in df_clf.columns:
                possible_cat.append(c)
        # Add encoded versions of these categorical features
        for c in possible_cat:
            if c:
                le = LabelEncoder()
                df_clf[c + "_enc"] = le.fit_transform(df_clf[c].astype(str))
                features.append(c + "_enc")

        X = df_clf[features]
        y = df_clf['risk_label']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # Scaling
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train Decision Tree and Random Forest
        dt = DecisionTreeClassifier(random_state=42, max_depth=5)
        rf = RandomForestClassifier(random_state=42, n_estimators=200)

        dt.fit(X_train_s, y_train)
        rf.fit(X_train_s, y_train)

        # Predictions
        y_pred_dt = dt.predict(X_test_s)
        y_pred_rf = rf.predict(X_test_s)

        # Evaluation - show Random Forest accuracy prominently (you asked for RF accuracy)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        st.subheader("Random Forest Performance")
        st.write(f"**Accuracy:** {acc_rf:.4f}")

        st.text("Classification Report (Random Forest):")
        st.text(classification_report(y_test, y_pred_rf, digits=4))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_rf)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (Random Forest)")
        st.pyplot(fig)

        # Feature importances
        importances = rf.feature_importances_
        fi = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False)
        st.subheader("Feature Importances (Random Forest)")
        st.bar_chart(fi.set_index("feature").head(10))

# ---------------------------
# TASK 2: Forecasting
# ---------------------------
with tab2:
    st.header("Task 2 — Forecasting Employment / Unemployment Trends")

    st.markdown("""
    Forecasting method:
    - **Prophet** is used by default (if available).  
    - If Prophet import or fitting fails, the app falls back to **ARIMA** (statsmodels).
    """)

    # Prepare series for forecasting
    fc_df = country_data[[date_col, unemp_col]].rename(columns={date_col: 'ds', unemp_col: 'y'}).dropna()
    fc_df['ds'] = pd.to_datetime(fc_df['ds'], errors='coerce')
    fc_df = fc_df.dropna(subset=['ds']).sort_values('ds').reset_index(drop=True)

    st.write("Historical series length:", len(fc_df))

    if len(fc_df) < 3:
        st.warning("Not enough historical points for robust forecasting. Need at least 3 observations.")
    else:
        horizon = st.slider("Forecast horizon (years)", min_value=1, max_value=10, value=5)
        periods = horizon
        freq = 'Y'  # yearly frequency

        try:
            if use_prophet:
                st.write("Using Prophet for forecasting.")
                model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                model.fit(fc_df)
                future = model.make_future_dataframe(periods=periods, freq=freq)
                forecast = model.predict(future)

                st.write("Forecast preview:")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

                # Plot interactive plotly
                fig_forecast = plot_plotly(model, forecast)
                fig_forecast.update_layout(title=f"Forecasted {unemp_col} for {selected_country}")
                st.plotly_chart(fig_forecast, use_container_width=True)

                # Show components
                st.write("Forecast components:")
                comp_fig = model.plot_components(forecast)
                st.pyplot(comp_fig)

            else:
                # Fallback to ARIMA
                st.warning("Prophet not available — falling back to ARIMA (statsmodels).")
                # Convert to series with yearly frequency
                ts = fc_df.set_index('ds')['y'].asfreq('Y')
                ts = ts.fillna(method='ffill').fillna(method='bfill')
                model = ARIMA(ts, order=(1, 1, 1))
                fitted = model.fit()
                forecast_res = fitted.get_forecast(steps=periods)
                fc_df_out = forecast_res.summary_frame(alpha=0.05)
                fc_df_out = fc_df_out.reset_index().rename(columns={'index': 'ds', 'mean':'yhat', 'mean_ci_lower':'yhat_lower', 'mean_ci_upper':'yhat_upper'})
                st.write("ARIMA Forecast preview:")
                st.dataframe(fc_df_out.tail(periods))

                # Combine historical + forecast for plotting
                hist_plot = pd.DataFrame({'ds': ts.index, 'y': ts.values})
                plot_df = pd.concat([hist_plot, fc_df_out[['ds','yhat']].rename(columns={'yhat':'y'})], ignore_index=True)

                fig = px.line(plot_df, x='ds', y='y', title=f"ARIMA Forecast for {selected_country}")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Forecasting failed: {e}")

# ---------------------------
# TASK 3: Industry Impact / segmentation
# ---------------------------
with tab3:
    st.header("Task 3 — Industry Impact & Disparities")

    if industry_col and industry_col in df_local.columns:
        st.subheader("Industry-level Unemployment Overview")
        industry_df = df_local[df_local[country_col].astype(str) == str(selected_country)].groupby(industry_col)[unemp_col].mean().reset_index().sort_values(unemp_col, ascending=False)
        st.dataframe(industry_df.head(20))

        fig = px.bar(industry_df.head(15), x=industry_col, y=unemp_col, title=f"Avg Unemployment Rate by Industry — {selected_country}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No industry column mapped. Map an industry column in the sidebar to enable industry impact view.")

    st.subheader("Regional / Gender Disparities")
    if region_col and region_col in df_local.columns:
        region_df = df_local[df_local[country_col].astype(str) == str(selected_country)].groupby(region_col)[unemp_col].mean().reset_index().sort_values(unemp_col, ascending=False)
        figr = px.bar(region_df, x=region_col, y=unemp_col, title="Avg Unemployment Rate by Region")
        st.plotly_chart(figr, use_container_width=True)
    else:
        st.info("No region column mapped.")

    if gender_col and gender_col in df_local.columns:
        gender_df = df_local[df_local[country_col].astype(str) == str(selected_country)].groupby(gender_col)[unemp_col].mean().reset_index()
        figp = px.pie(gender_df, values=unemp_col, names=gender_col, title="Unemployment by Gender (avg rate)")
        st.plotly_chart(figp, use_container_width=True)
    else:
        st.info("No gender column mapped.")

# ---------------------------
# Footer / Notes
# ---------------------------
st.markdown("---")
st.markdown("""
**Notes & assumptions**
- The classification target is derived deterministically from unemployment rate (statistical threshold / YOY change).
- Forecasting uses Prophet by default. If Prophet is unavailable, ARIMA is used as a fallback.
- For production / evaluation you should replace the deterministic label with real job-loss outcome labels if available.
""")
