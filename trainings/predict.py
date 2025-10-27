import os
import joblib
import pandas as pd
import numpy as np
import hopsworks
from dotenv import load_dotenv

# ------------------------------------------------
# 🔐 Connect to Hopsworks
# ------------------------------------------------
HOPSWORKS_API_KEY = os.environ.get("AQI_FORECAST_API_KEY")  # must match secret name

if not HOPSWORKS_API_KEY:
    raise ValueError(
        "❌ Missing Hopsworks API key! Please set 'AQI_FORECAST_API_KEY' as a GitHub Secret."
    )
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)

fs = project.get_feature_store()
mr = project.get_model_registry()
print("✅ Connected to Hopsworks")

# ------------------------------------------------
# 📦 Load latest feature group
# ------------------------------------------------
print("📦 Reading feature group...")
fg = fs.get_feature_group("aqi_features", version=1)
df = fg.read()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)
print(f"✅ Data loaded! Shape: {df.shape}")

# ------------------------------------------------
# 🤖 Load Random Forest model
# ------------------------------------------------
print("🤖 Loading latest RF model from Hopsworks...")

# ✅ FIXED: use correct name
rf_models = mr.get_models(name="rf_aqi_model")
if not rf_models:
    raise ValueError("❌ No model named 'rf_aqi_model' found in Hopsworks Model Registry!")

rf_model = max(rf_models, key=lambda m: m.version)
model_dir = rf_model.download()

# Expected paths inside the downloaded directory
model_path = os.path.join(model_dir, "model.joblib")
scaler_path = os.path.join(model_dir, "scaler.joblib")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("❌ model.joblib or scaler.joblib not found in downloaded folder!")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
print(f"✅ Loaded RF model version {rf_model.version}")

# ------------------------------------------------
# 🔮 Make predictions
# ------------------------------------------------
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

df = df.dropna(subset=feature_cols)
X = df[feature_cols].values
X_scaled = scaler.transform(X)
preds = model.predict(X_scaled)

# ------------------------------------------------
# 💾 Save predictions locally
# ------------------------------------------------
OUTPUT_PATH = "data/predictions/latest_predictions.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
pd.DataFrame({"predicted_aqi": preds[-10:]}).to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Predictions saved to: {OUTPUT_PATH}")
print(preds[-10:])
