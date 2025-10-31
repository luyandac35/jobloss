# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="AI & Employment Dashboard", layout="wide")
st.title("📊 AI & Employment Impact Dashboard")

# Sidebar navigation
menu = st.sidebar.radio(
    "Navigate Dashboard",
    ["Data Overview", "Job Loss Prediction", "Employment Forecasting", "Industry Impact"]
)

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("global_unemployment_data.csv")  # change to your dataset
    return df

df = load_data()

# -------------------------------
# DATA OVERVIEW
# -------------------------------
if menu == "Data Overview":
    st.header("🧭 Data Overview & Summary")
    st.write("This section allows users to explore data structure, missing values, and summary statistics.")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("🧹 Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📊 Distribution by Key Variables")
    selected_col = st.selectbox("Select column to visualize:", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------------
# JOB LOSS PREDICTION (Classification)
# -------------------------------
elif menu == "Job Loss Prediction":
    st.header("🤖 Task 1 — Predicting Job Loss Risk")

    target_col = st.selectbox("Select Target Column (Categorical)", df.columns)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categorical data
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale numeric features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("🎯 Model Evaluation")
    st.write(f"**Random Forest Accuracy:** {accuracy:.2f}")

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Feature Importance
    st.subheader("🔥 Feature Importance")
    importance = pd.Series(model.feature_importances_, index=df.drop(columns=[target_col]).columns)
    imp_df = importance.sort_values(ascending=False).head(10).reset_index()
    imp_df.columns = ["Feature", "Importance"]

    fig = px.bar(imp_df, x="Feature", y="Importance", title="Top 10 Important Features (Random Forest Classifier)")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# FORECASTING EMPLOYMENT TRENDS
# -------------------------------
elif menu == "Employment Forecasting":
    st.header("📈 Task 2 — Forecasting Employment Trends (Prophet)")

    st.write("Use Prophet to forecast employment/unemployment trends over time.")

    date_col = st.selectbox("Select Date Column", df.columns)
    value_col = st.selectbox("Select Value Column to Forecast", df.columns)

    forecast_df = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
    model = Prophet()
    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=12, freq="M")
    forecast = model.predict(future)

    st.subheader("🔮 Forecast Results")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("📊 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

# -------------------------------
# INDUSTRY IMPACT ANALYSIS
# -------------------------------
elif menu == "Industry Impact":
    st.header("🏭 Task 3 — Analysing Industry Impact (Disparities / Segmentation)")

    st.write("This section highlights which sectors and demographics are most exposed to AI-related displacement risks.")

    sector_col = st.selectbox("Select Sector Column", df.columns)
    risk_col = st.selectbox("Select Risk Column", df.columns)

    impact_df = df.groupby(sector_col)[risk_col].mean().sort_values(ascending=False).reset_index()

    fig = px.bar(
        impact_df,
        x=sector_col,
        y=risk_col,
        title="Average Job Loss Risk by Sector",
        color=risk_col,
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### 🧩 Equity Notes
    Sectors with higher AI exposure may require targeted **reskilling programs**.
    Consider policy measures to support **high-risk sectors** such as retail and data entry.
    """)


