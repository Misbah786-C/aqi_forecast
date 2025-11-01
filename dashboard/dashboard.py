import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from joblib import load
import hopsworks
import logging

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="🌤️ AQI Forecast Dashboard", layout="wide")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
    <style>
    body, .main, .block-container {
        background: linear-gradient(180deg, #e3f2fd 0%, #ffffff 100%) !important;
        color: #000000 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    .title {font-size: 2.2em; font-weight: bold; color: #0d47a1 !important;}
    .subtitle {font-size: 1.05em; color: #37474f !important; margin-bottom: 20px;}
    .section-header {font-size: 1.3em; color: #0d47a1; font-weight: 600; margin-top: 20px;}
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
        text-align: center;
        transition: 0.3s;
    }
    .card:hover {transform: translateY(-4px); box-shadow: 0 6px 16px rgba(0,0,0,0.08);}
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("<div class='title'>🌤️ AQI Forecast Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>This dashboard loads your model and features from Hopsworks, generates predictions, and presents both EDA and forecast insights.</div>", unsafe_allow_html=True)

# ----------------------------
# Connect to Hopsworks & Load Data + Model
# ----------------------------
try:
    project = hopsworks.login()
    fs = project.get_feature_store()
    feature_group = fs.get_feature_group(name="aqi_features", version=1)
    df = feature_group.read()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], unit="ms")

    model = load("trained_model.joblib")
    latest_df = df.sort_values("timestamp_utc").tail(3)
    X = latest_df.drop(columns=["aqi_aqicn", "timestamp_utc"])
    predicted_aqi = model.predict(X)

    forecast_df = pd.DataFrame({
        "Date": latest_df["timestamp_utc"].dt.date,
        "Predicted_AQI": predicted_aqi
    })
    st.success("✅ Data and model loaded successfully from Feature Store.")
except Exception as e:
    st.error(f"❌ Failed to load data or model: {e}")
    st.stop()

# ----------------------------
# Helper Functions
# ----------------------------
def categorize_aqi(aqi):
    if aqi <= 50: return "Good 🌿"
    elif aqi <= 100: return "Moderate 😊"
    elif aqi <= 150: return "Unhealthy (Sensitive) 😐"
    elif aqi <= 200: return "Unhealthy 😷"
    elif aqi <= 300: return "Very Unhealthy 🤒"
    else: return "Hazardous ☠️"

def get_icon(aqi):
    if aqi <= 50: return "☀️"
    elif aqi <= 100: return "🌤️"
    elif aqi <= 150: return "🌥️"
    elif aqi <= 200: return "🌧️"
    else: return "🌫️"

# ====================================================
# SECTION 1 → AQI Forecast (Predictions)
# ====================================================
st.markdown("<div class='section-header'>📆 3-Day AQI Forecast</div>", unsafe_allow_html=True)

cols = st.columns(3)
for i, row in forecast_df.iterrows():
    category = categorize_aqi(row['Predicted_AQI'])
    icon = get_icon(row['Predicted_AQI'])
    with cols[i]:
        st.markdown(f"""
        <div class="card">
            <h3>{row['Date'].strftime('%A')}</h3>
            <h1>{icon}</h1>
            <p><b>{int(row['Predicted_AQI'])}</b> AQI</p>
            <p>{category}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ====================================================
# SECTION 2 → AQI Trend Visualization
# ====================================================
st.markdown("<div class='section-header'>📈 Predicted AQI Trend</div>", unsafe_allow_html=True)
fig, ax = plt.subplots(figsize=(7, 3))
sns.lineplot(data=forecast_df, x="Date", y="Predicted_AQI", marker="o", color="#1976d2", linewidth=2.5)
ax.set_facecolor("#ffffff")
ax.set_xlabel("Date")
ax.set_ylabel("Predicted AQI")
ax.set_title("Predicted AQI for Next 3 Days", fontsize=12, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig)

# ====================================================
# SECTION 3 → Exploratory Data Analysis (EDA)
# ====================================================
st.markdown("<div class='section-header'>🔍 Exploratory Data Analysis</div>", unsafe_allow_html=True)
st.write("These insights are generated from your stored feature data in the Feature Store.")

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.subheader("📊 AQI Over Time")
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.lineplot(data=df, x="timestamp_utc", y="aqi_aqicn", color="#0d47a1", linewidth=1.8)
    ax.set_xlabel("Timestamp (UTC)")
    ax.set_ylabel("AQI")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with eda_col2:
    st.subheader("🔥 Feature Correlation with AQI")
    corr = df.corr(numeric_only=True)["aqi_aqicn"].sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr.to_frame(), annot=True, cmap="coolwarm", center=0)
    st.pyplot(fig)

st.markdown("### 🌦️ AQI vs Weather Features")
weather_features = ["ow_temp", "ow_humidity", "ow_wind_speed"]
wf_cols = st.columns(len(weather_features))

for i, feature in enumerate(weather_features):
    with wf_cols[i]:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.scatterplot(data=df, x=feature, y="aqi_aqicn", alpha=0.6)
        ax.set_xlabel(feature)
        ax.set_ylabel("AQI")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

st.markdown("---")
st.info("✅ Dashboard complete — model predictions, EDA insights, and data visualizations loaded successfully.")
