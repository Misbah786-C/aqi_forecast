import os
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# ---------------------------------------
# Streamlit Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="🌆 Karachi AQI & Weather Forecast",
    page_icon="🌫️",
    layout="wide"
)

# ---------------------------------------
# Load Data
# ---------------------------------------
PRED_PATH = "data/predictions/latest_predictions.csv"

@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame()

forecast_df = load_data(PRED_PATH)

# ---------------------------------------
# AQI Category Helper with Health Comments
# ---------------------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return (
            "Good",
            "#009966",
            "Air quality is satisfactory, and air pollution poses little or no risk."
        )
    elif aqi <= 100:
        return (
            "Moderate",
            "#FFDE33",
            "Acceptable air quality, but some pollutants may be a concern for sensitive individuals."
        )
    elif aqi <= 150:
        return (
            "Unhealthy for Sensitive Groups",
            "#FF9933",
            "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
        )
    elif aqi <= 200:
        return (
            "Unhealthy",
            "#CC0033",
            "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."
        )
    elif aqi <= 300:
        return (
            "Very Unhealthy",
            "#660099",
            "Health alert: everyone may experience more serious health effects. Avoid outdoor exertion."
        )
    else:
        return (
            "Hazardous",
            "#7E0023",
            "Emergency conditions: the entire population is likely to be affected. Stay indoors and avoid exposure."
        )

# ---------------------------------------
# Custom Dark Theme Styling
# ---------------------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(180deg, #0d1b2a, #1b263b);
        color: white;
    }
    .stApp {
        background: linear-gradient(180deg, #0d1b2a, #1b263b);
        color: white;
    }
    div[data-testid="stMetricValue"] {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------
# Header
# ---------------------------------------
st.markdown("""
    <div style='text-align:center; padding:10px 0;'>
        <h1 style='color:#f0f0f0;'>🌆 Karachi AQI & Weather Forecast</h1>
        <p style='color:#aaa;'>Live AQI predictions powered by Hopsworks Feature Store & OpenWeather APIs</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

# ---------------------------------------
# AQI Forecast Section
# ---------------------------------------
if not forecast_df.empty:
    forecast_df["predicted_for_utc"] = pd.to_datetime(forecast_df["predicted_for_utc"])

    st.markdown("### 🌤 Next 3 Days AQI Forecast")
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Forecast Cards ---
    for i, row in enumerate(forecast_df.itertuples(), start=1):
        aqi_val = float(row.predicted_aqi)
        cat, color, comment = aqi_category(aqi_val)
        date_str = row.predicted_for_utc.strftime("%A, %b %d")

        st.markdown(f"""
        <div style="
            background-color:#1e2a3a;
            border-radius:15px;
            padding:25px;
            margin-bottom:20px;
            box-shadow:0 4px 8px rgba(0,0,0,0.3);
            border-left:8px solid {color};
        ">
            <h3 style="margin:0; color:#fff;">{date_str}</h3>
            <h1 style="margin:5px 0 0 0; color:{color}; font-size:48px;">{aqi_val:.0f} AQI</h1>
            <p style="color:{color}; font-weight:600; margin:2px 0;">{cat}</p>
            <p style="color:#ccc; font-size:14px;">{comment}</p>
            <p style="color:#777; font-size:13px;">Model v{int(row.model_version)} | Updated: {row.predicted_for_utc.strftime('%Y-%m-%d %H:%M UTC')}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- AQI Trend Chart ---
    fig = px.line(
        forecast_df,
        x="predicted_for_utc",
        y="predicted_aqi",
        title="📈 3-Day AQI Forecast Trend",
        markers=True,
        color_discrete_sequence=["#00ccff"]
    )

    fig.update_layout(
        title_x=0.35,
        paper_bgcolor="#1b263b",
        plot_bgcolor="#1b263b",
        font=dict(color="white"),
        xaxis=dict(title="Forecast Date", showgrid=True, gridcolor="#2f3e4e"),
        yaxis=dict(title="Predicted AQI", showgrid=True, gridcolor="#2f3e4e"),
        margin=dict(t=60, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("⚠️ No forecast data found. Please run your `predict.py` script first.")

# ---------------------------------------
# Footer
# ---------------------------------------
st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align:center; color:#999; font-size:13px;'>
        Data sourced via AQICN & OpenWeather | Model deployed with Hopsworks 💨
    </div>
""", unsafe_allow_html=True)
