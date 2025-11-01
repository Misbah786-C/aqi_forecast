#!/usr/bin/env python3
import os
import joblib
import logging
from datetime import datetime
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("predict")

# --- Load Environment Variables ---
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")

# --- Features to Use for Prediction ---
FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday",
    "lag_1", "lag_2", "rolling_mean_3"
]

def get_artifact_files(model_dir):
    paths = {}
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name in ("model.joblib", "scaler.joblib"):
                paths[name] = os.path.join(root, name)
    return paths

def load_model_and_scaler(model_registry, model_name="rf_aqi_model"):
    models = model_registry.get_models(model_name)
    if not models:
        raise RuntimeError(f"No models found with name '{model_name}'.")
    latest = max(models, key=lambda m: m.version)
    log.info(f"Loading model '{model_name}' (version {latest.version})...")
    model_dir = latest.download()
    files = get_artifact_files(model_dir)
    model = joblib.load(files["model.joblib"])
    scaler = joblib.load(files["scaler.joblib"])
    log.info("Model and scaler loaded successfully.")
    return model, scaler, latest.version

def main():
    log.info("Starting AQI prediction...")

    # ✅ FIXED: Correct variable name
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # Read latest features
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()

    if df.empty:
        log.error("No data available for prediction.")
        return

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    latest = df.iloc[[-1]].copy()
    log.info(f"Using latest data from {latest['timestamp_utc'].iloc[0]}")

    for col in FEATURE_COLS:
        if col not in latest.columns:
            latest[col] = 0

    X = latest[FEATURE_COLS].astype("float64")

    # ✅ Fixed file extensions
    model, scaler, version = load_model_and_scaler(mr)
    X_scaled = scaler.transform(X)
    prediction = float(model.predict(X_scaled)[0])

    result = {
        "predicted_aqi": prediction,
        "predicted_at_utc": datetime.utcnow().isoformat(sep=" "),
        "model_version": version
    }

    log.info(f"Predicted AQI: {prediction:.2f} (model v{version})")
    print(result)

if __name__ == "__main__":
    main()
