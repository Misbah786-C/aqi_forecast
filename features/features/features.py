import os
import pandas as pd
from datetime import datetime, timezone
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

def load_latest_parquet_or_csv(parquet_path: str, csv_path: str, timestamp_col: str) -> pd.DataFrame:
    """
    Load data from parquet if exists, otherwise load CSV and save as parquet.
    Ensures the timestamp column exists.
    """
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        logging.info(f"Loaded data from {parquet_path}")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded data from {csv_path}, saving as parquet...")
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.to_parquet(parquet_path, index=False)
    else:
        raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} found!")

    # Ensure timestamp column exists
    if timestamp_col not in df.columns:
        df[timestamp_col] = datetime.now(timezone.utc).isoformat()

    return df

def build_features():
    # Absolute paths
    ow_csv = r"C:\projects\aqi_forecast\openweather_data.csv"
    ow_parquet = r"C:\projects\aqi_forecast\data\raw_openweather\latest_openweather.parquet"

    aqicn_csv = r"C:\projects\aqi_forecast\aqicn_data.csv"
    aqicn_parquet = r"C:\projects\aqi_forecast\data\raw_aqicn\latest_aqicn.parquet"

    out_dir = r"C:\projects\aqi_forecast\data\features"
    os.makedirs(out_dir, exist_ok=True)

    # Load datasets
    df_ow = load_latest_parquet_or_csv(ow_parquet, ow_csv, "ow_timestamp")
    df_aq = load_latest_parquet_or_csv(aqicn_parquet, aqicn_csv, "aqicn_timestamp")

    # Add prefixes to avoid duplicate column names
    df_ow = df_ow.add_prefix("ow_")
    df_aq = df_aq.add_prefix("aqicn_")

    # Rename timestamp columns back
    df_ow = df_ow.rename(columns={"ow_ow_timestamp": "ow_timestamp"})
    df_aq = df_aq.rename(columns={"aqicn_aqicn_timestamp": "aqicn_timestamp"})

    # Merge side by side
    df = pd.concat([df_ow, df_aq], axis=1)

    # Use OpenWeather timestamp as main
    df["timestamp_utc"] = pd.to_datetime(df["ow_timestamp"], utc=True, errors="coerce")
    df["timestamp_utc"].fillna(datetime.now(timezone.utc), inplace=True)

    # Add time-based features
    df["hour"] = df["timestamp_utc"].dt.hour
    df["day"] = df["timestamp_utc"].dt.day
    df["month"] = df["timestamp_utc"].dt.month
    df["weekday"] = df["timestamp_utc"].dt.weekday  # 0=Mon, 6=Sun

    # Save outputs
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ts_file = os.path.join(out_dir, f"features_{ts}.parquet")
    latest_file = os.path.join(out_dir, "latest_features.parquet")

    df.to_parquet(ts_file, index=False)
    df.to_parquet(latest_file, index=False)

    logging.info(f"Saved {ts_file} and updated {latest_file}")
    print(df.head())

if __name__ == "__main__":
    build_features()
