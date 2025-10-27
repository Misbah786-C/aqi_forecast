import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
CITY = os.getenv("CITY", "Karachi")

def fetch_aqicn():
    """Fetch live AQI data from AQICN API"""
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    response = requests.get(url).json()

    if response["status"] != "ok":
        print("❌ Failed to fetch AQICN data:", response)
        return None

    data = response["data"]
    iaqi = data["iaqi"]

    record = {
        "city": CITY,
        "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "aqi": data["aqi"],
        "pm2_5": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
    }

    df = pd.DataFrame([record])
    output_path = os.path.join(os.getcwd(), "aqicn_data.csv")

    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, index=False)

    print("✅ AQICN data fetched and saved successfully!")
    return df


if __name__ == "__main__":
    fetch_aqicn()
