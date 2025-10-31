import os
import hopsworks
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# ---------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------------------------------------------------------
# Load API key from environment (GitHub Secrets or system env)
# ---------------------------------------------------------------
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")  # Make sure this env var is set

if not HOPSWORKS_API_KEY:
    logging.error("❌ Missing 'AQI_FORECAST_API_KEY' in environment variables.")
    exit(1)

# ---------------------------------------------------------------
# Connect to Hopsworks
# ---------------------------------------------------------------
try:
    logging.info("🔐 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    logging.info("✅ Connected to Hopsworks successfully.")
except Exception as e:
    logging.error(f"❌ Hopsworks connection failed: {e}")
    exit(1)

# ---------------------------------------------------------------
# Fetch latest feature dataset
# ---------------------------------------------------------------
try:
    logging.info("📥 Fetching latest feature data...")
    feature_group = fs.get_feature_group(name="aqi_features", version=1)
    df = feature_group.read()
    logging.info(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    logging.error(f"❌ Failed to fetch feature data: {e}")
    exit(1)

# ---------------------------------------------------------------
# Run EDA
# ---------------------------------------------------------------
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Summary stats
summary = f"""
📊 EDA Summary
--------------------------
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Missing Values: {df.isnull().sum().sum()}
Numeric Columns: {len(df.select_dtypes(include='number').columns)}
"""

summary_file = os.path.join(OUTPUT_DIR, "eda_summary.txt")
with open(summary_file, "w") as f:
    f.write(summary)

logging.info(f"📄 Summary saved to {summary_file}")

# 2. Correlation heatmap
try:
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    heatmap_file = os.path.join(OUTPUT_DIR, "heatmap.png")
    plt.savefig(heatmap_file)
    plt.close()
    logging.info(f"🖼 Heatmap saved to {heatmap_file}")
except Exception as e:
    logging.warning(f"⚠ Failed to generate heatmap: {e}")

logging.info("✅ EDA completed. All results saved in 'eda_outputs/' folder.")
