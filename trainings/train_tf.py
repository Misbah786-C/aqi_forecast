import os
import time
import shutil
import logging
import numpy as np
import pandas as pd
import joblib
import hopsworks
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ------------------------------------------------
# Logging setup
# ------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------
# Load environment variables
# ------------------------------------------------
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("AQI_FORECAST_API_KEY")

# ------------------------------------------------
# Connect to Hopsworks
# ------------------------------------------------
logger.info("🔐 Connecting to Hopsworks...")
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
mr = project.get_model_registry()

# ------------------------------------------------
# Load feature data
# ------------------------------------------------
logger.info("📦 Loading feature group 'aqi_features' (version 1)...")
fg = fs.get_feature_group(name="aqi_features", version=1)

for attempt in range(3):
    try:
        df = fg.read()
        logger.info("✅ Data loaded via Arrow Flight.")
        break
    except Exception as e:
        logger.warning(f"⚠️ Attempt {attempt + 1} failed: {e}")
        time.sleep(3)
else:
    logger.info("🔁 Fallback to pandas engine...")
    df = fg.read(read_options={"engine": "pandas"})

logger.info(f"✅ Data shape: {df.shape}")

# ------------------------------------------------
# Clean and prepare data
# ------------------------------------------------
logger.info("🧹 Cleaning and preparing data...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

target_col = "aqi_aqicn"
feature_cols = [
    "ow_temp", "ow_pressure", "ow_humidity", "ow_wind_speed", "ow_wind_deg",
    "ow_clouds", "ow_co", "ow_no2", "ow_pm2_5", "ow_pm10",
    "hour", "day", "month", "weekday"
]

df = df.drop(columns=["timestamp_utc"], errors="ignore")
df = df.dropna(subset=[target_col] + feature_cols)
if df.empty:
    raise ValueError("🚨 Dataset empty after cleaning!")

# ------------------------------------------------
# Create LSTM sequences
# ------------------------------------------------
SEQUENCE_LENGTH = 7
X_seq, y_seq = [], []
for i in range(SEQUENCE_LENGTH, len(df)):
    X_seq.append(df[feature_cols].iloc[i - SEQUENCE_LENGTH:i].values)
    y_seq.append(df[target_col].iloc[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

logger.info(f"✅ Sequence shapes: X={X_seq.shape}, y={y_seq.shape}")

# ------------------------------------------------
# Scaling
# ------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

nsamples, ntimesteps, nfeatures = X_seq.shape
X_scaled_flat = scaler_X.fit_transform(X_seq.reshape(nsamples * ntimesteps, nfeatures))
X_scaled = X_scaled_flat.reshape(nsamples, ntimesteps, nfeatures)
y_scaled = scaler_y.fit_transform(y_seq.reshape(-1, 1))

# ------------------------------------------------
# Build LSTM model
# ------------------------------------------------
logger.info("🧱 Building LSTM model...")
model = Sequential([
    LSTM(128, activation="tanh", return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))),
    Dropout(0.3),
    LSTM(64, activation="tanh"),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=["mae"]
)

# ------------------------------------------------
# Train model
# ------------------------------------------------
logger.info("🚀 Training LSTM model...")
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

history = model.fit(
    X_scaled, y_scaled,
    validation_split=0.2,
    epochs=100,
    batch_size=8,
    callbacks=[early_stop, reduce_lr],
    shuffle=True,
    verbose=1
)

train_loss = float(history.history["loss"][-1])
val_loss = float(history.history["val_loss"][-1])
val_mae = float(history.history["mae"][-1])

logger.info(f"✅ Training complete | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

# ------------------------------------------------
# Save artifacts locally (temp)
# ------------------------------------------------
MODEL_DIR = "models/tf_model"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "tf_lstm_model.keras")
SCALER_X_PATH = os.path.join(MODEL_DIR, "scaler_X.joblib")
SCALER_Y_PATH = os.path.join(MODEL_DIR, "scaler_y.joblib")

model.save(MODEL_PATH)
joblib.dump(scaler_X, SCALER_X_PATH)
joblib.dump(scaler_y, SCALER_Y_PATH)

# ------------------------------------------------
# Upload to Hopsworks Model Registry
# ------------------------------------------------
logger.info("☁️ Uploading model + scalers to Hopsworks Model Registry...")

model_meta = mr.python.create_model(
    name="tf_lstm_aqi_model",
    metrics={"train_loss": train_loss, "val_loss": val_loss, "val_mae": val_mae},
    description="TensorFlow LSTM model for Karachi AQI forecasting (trained via CI/CD)"
)
model_meta.save(MODEL_DIR)

logger.info("🎉 Model and scalers successfully uploaded to Hopsworks Model Registry!")

# ------------------------------------------------
# Clean up local files
# ------------------------------------------------
shutil.rmtree(MODEL_DIR)
logger.info("🧹 Local model artifacts cleaned up after upload.")
logger.info("🏁 TensorFlow training pipeline completed successfully.")
