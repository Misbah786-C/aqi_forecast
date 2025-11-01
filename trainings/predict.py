#!/usr/bin/env python3
import os
import joblib
import logging
from datetime import datetime, timedelta
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# -----------------------------------------------------
# Logging setup
# -----------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("predict")

# -----------------------------------------------------
# Load environment variables
# -----------------------------------------------------
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")

# -----------------------------------------------------
# Features to use
# -----------------------------------------------------
FEATURE_COLS = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday",
    "lag_1", "lag_2", "rolling_mean_3"
]

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------
def get_artifact_files(model_dir):
    paths = {}
    for root, _, files in os.walk(model_dir):
        for name in files:
            if name in ("model.joblib", "scaler.joblib"):
                paths[name] = os.path.join(root, name)
    return paths


def load_model_and_scaler(model_registry, model_name="rf_aqi_model"):
    models = model_registry.get_models(model_name)
    latest = max(models, key=lambda m: m.version)
    log.info(f"Loading model '{model_name}' (version {latest.version})...")
    model_dir = latest.download()

    files = get_artifact_files(model_dir)
    model = joblib.load(files["model.joblib"])
    scaler = joblib.load(files["scaler.joblib"])
    log.info("Model and scaler loaded successfully.")
    return model, scaler, latest.version


# -----------------------------------------------------
# Main forecast logic
# -----------------------------------------------------
def main():
    log.info("Starting 3-day AQI forecast generation...")

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()

    if df.empty:
        log.error("No data available for prediction.")
        return

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    latest = df.iloc[[-1]].copy()
    log.info(f"Using latest data from {latest['timestamp_utc'].iloc[0]}")

    # Add any missing columns as zeros
    for col in FEATURE_COLS:
        if col not in latest.columns:
            latest[col] = 0.0

    model, scaler, version = load_model_and_scaler(mr)

    # Handle unseen columns automatically
    if hasattr(scaler, "feature_names_in_"):
        valid_cols = [c for c in scaler.feature_names_in_ if c in latest.columns]
    else:
        valid_cols = [c for c in FEATURE_COLS if c in latest.columns]

    forecasts = []
    current_date = pd.to_datetime(latest["timestamp_utc"].iloc[0])

    for i in range(1, 4):  # next 3 days
        future = latest.copy()
        future["timestamp_utc"] = current_date + timedelta(days=i)
        future["day"] = future["timestamp_utc"].dt.day
        future["month"] = future["timestamp_utc"].dt.month
        future["weekday"] = future["timestamp_utc"].dt.weekday

        # Dummy lag simulation
        if "lag_1" in future.columns:
            future["lag_2"] = future["lag_1"]
            if "aqi_aqicn" in latest.columns:
                future["lag_1"] = latest["aqi_aqicn"].iloc[-1]
            else:
                future["lag_1"] = 0.0
            future["rolling_mean_3"] = (future["lag_1"] + future["lag_2"]) / 2

        X = future[valid_cols].astype("float64")
        X_scaled = scaler.transform(X)
        pred = float(model.predict(X_scaled)[0])

        forecasts.append({
            "forecast_day": i,
            "predicted_aqi": pred,
            "predicted_for_utc": (current_date + timedelta(days=i)).isoformat(sep=" "),
            "model_version": version
        })

    forecast_df = pd.DataFrame(forecasts)
    os.makedirs("data/predictions", exist_ok=True)
    forecast_df.to_csv("data/predictions/latest_predictions.csv", index=False)
    log.info("3-day forecast saved successfully.")
    print(forecast_df)


# -----------------------------------------------------
# Entry point
# -----------------------------------------------------
if __name__ == "__main__":
    main()
