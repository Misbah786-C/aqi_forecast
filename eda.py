import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from dotenv import load_dotenv

# ------------------------------------------------
# Logging setup
# ------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ------------------------------------------------
# Path setup (auto-detect correct project root)
# ------------------------------------------------
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "features", "training_dataset.csv")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

print(f"📂 Project Root: {PROJECT_ROOT}")
print(f"📁 Dataset Path: {DATA_PATH}")

# ------------------------------------------------
# Load environment or GitHub secrets
# ------------------------------------------------
# Try GitHub secrets first
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
AQI_FORECAST_API_KEY = os.getenv("aqi_forecast_api_key")

# If not found, load from local .env
if not any([OPENWEATHER_API_KEY, AQICN_TOKEN, AQI_FORECAST_API_KEY]):
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
        OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
        AQICN_TOKEN = os.getenv("AQICN_TOKEN")
        AQI_FORECAST_API_KEY = os.getenv("aqi_forecast_api_key")
        logging.info("🔑 Loaded secrets from local .env file.")
    else:
        logging.warning("⚠️ No .env file or GitHub Secrets found — running locally with limited access.")
else:
    logging.info("🔑 Loaded secrets from GitHub Secrets environment.")

# ------------------------------------------------
# Load dataset
# ------------------------------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
df = df.sort_values("timestamp_utc").fillna(method="ffill")

logging.info(f"✅ Dataset loaded successfully with {len(df)} rows and {len(df.columns)} columns.")

# ------------------------------------------------
# Basic info
# ------------------------------------------------
print("\n📋 Dataset Info:")
print(df.info())
print("\n📊 Dataset Shape:", df.shape)
print("\n🔢 Columns:", df.columns.tolist())
print("\n🧮 Missing Values:\n", df.isnull().sum())

# ------------------------------------------------
# Descriptive Statistics
# ------------------------------------------------
print("\n📈 Descriptive Statistics:")
print(df.describe().T)

# ------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
else:
    logging.warning("⚠️ Not enough numeric columns for correlation heatmap — skipping.")

# ------------------------------------------------
# Time Series Trends
# ------------------------------------------------
if "timestamp_utc" in df.columns and "aqi_aqicn" in df.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp_utc"], df["aqi_aqicn"], label="AQI", color="red")
    plt.xlabel("Time")
    plt.ylabel("AQI Value")
    plt.title("AQI Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    logging.warning("⚠️ Columns for time series plot not found — skipping.")

# ------------------------------------------------
# Distribution plots
# ------------------------------------------------
if len(numeric_cols) > 0:
    df[numeric_cols].hist(figsize=(14, 12), bins=30)
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.show()

# ------------------------------------------------
# Relationship plots
# ------------------------------------------------
selected_cols = [c for c in ["aqi_aqicn", "ow_temp", "ow_humidity", "ow_pm2_5", "ow_pm10"] if c in df.columns]
if len(selected_cols) >= 2:
    sns.pairplot(df, vars=selected_cols, diag_kind="kde")
    plt.suptitle("Pairwise Feature Relationships", y=1.02)
    plt.show()
else:
    logging.warning("⚠️ Not enough columns found for pairplot — skipping.")

logging.info("✅ EDA completed successfully.")
