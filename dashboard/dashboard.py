import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="AQI Forecast Dashboard", layout="wide")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
    <style>
    /* Background & text */
    body, .main, .block-container {
        background: linear-gradient(180deg, #e3f2fd 0%, #ffffff 100%) !important;
        color: #000000 !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* Title */
    .title {
        text-align: left;
        font-size: 2.2em;
        font-weight: bold;
        color: #0d47a1 !important;
        margin-bottom: -5px;
    }

    /* Subtext */
    .subtitle {
        text-align: left;
        font-size: 1.05em;
        color: #37474f !important;
        margin-bottom: 25px;
    }

    /* Card styling */
    .card {
        background: #ffffff;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
        text-align: center;
        transition: 0.3s;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    .card h3 {
        margin: 0;
        font-size: 1.2em;
        color: #000000 !important;
    }
    .card h1 {
        margin: 5px 0;
        font-size: 2.5em;
        color: #000000 !important;
    }
    .card p {
        font-size: 1em;
        color: #000000 !important;
        margin: 3px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("<div class='title'>🌤️ AQI Forecast Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>This dashboard shows predicted Air Quality Index (AQI) for the next 3 days — powered by your trained model from the Feature Store.</div>", unsafe_allow_html=True)

# ----------------------------
# Example 3-Day Data
# ----------------------------
future_df = pd.DataFrame({
    "Date": pd.date_range("2025-11-01", periods=3),
    "Predicted_AQI": [155, 140, 120]
})

# ----------------------------
# Helper Functions
# ----------------------------
def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good 🌿"
    elif aqi <= 100:
        return "Moderate 😊"
    elif aqi <= 150:
        return "Unhealthy (Sensitive) 😐"
    elif aqi <= 200:
        return "Unhealthy 😷"
    elif aqi <= 300:
        return "Very Unhealthy 🤒"
    else:
        return "Hazardous ☠️"

def get_icon(aqi):
    if aqi <= 50:
        return "☀️"
    elif aqi <= 100:
        return "🌤️"
    elif aqi <= 150:
        return "🌥️"
    elif aqi <= 200:
        return "🌧️"
    else:
        return "🌫️"

# ----------------------------
# AQI Forecast Cards (3-Day)
# ----------------------------
cols = st.columns(3)
for i, row in future_df.iterrows():
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

# ----------------------------
# AQI Trend + Summary Statistics (Side by Side)
# ----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 AQI Trend for Next 3 Days")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.lineplot(data=future_df, x="Date", y="Predicted_AQI", marker="o", color="#1976d2", linewidth=2.5)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    ax.set_xlabel("Date", fontsize=11, color="black")
    ax.set_ylabel("Predicted AQI", fontsize=11, color="black")
    ax.set_title("Predicted AQI for Next 3 Days", fontsize=12, color="black", fontweight="bold")
    ax.tick_params(colors="black")
    ax.grid(True, linestyle="--", alpha=0.3)
    st.pyplot(fig)

with col2:
    st.subheader("📊 Summary Statistics")
    styled_df = future_df.describe().style.set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#e3f2fd'), ('color', 'black')]},
        {'selector': 'tbody td', 'props': [('color', 'black')]}
    ])
    st.dataframe(styled_df, use_container_width=True)
