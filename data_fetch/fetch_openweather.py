import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
LAT = os.getenv("LAT", "24.8607")  # Default: Karachi
LON = os.getenv("LON", "67.0011")
CITY = os.getenv("CITY", "Karachi")

def fetch_openweather():
    """Fetch current weather + pollution data from OpenWeather API"""
    base_url = "https://api.openweathermap.org/data/2.5/"
    
    weather_url = f"{base_url}weather?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}&units=metric"
    air_url = f"{base_url}air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"

    # Fetch data
    weather_data = requests.get(weather_url).json()
    air_data = requests.get(air_url).json()

    # Extract data
    data = {
        "city": CITY,
        "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "temp": weather_data["main"]["temp"],
        "humidity": weather_data["main"]["humidity"],
        "pressure": weather_data["main"]["pressure"],
        "wind_speed": weather_data["wind"]["speed"],
        "aqi": air_data["list"][0]["main"]["aqi"],
        "pm2_5": air_data["list"][0]["components"]["pm2_5"],
        "pm10": air_data["list"][0]["components"]["pm10"],
        "no2": air_data["list"][0]["components"]["no2"],
        "so2": air_data["list"][0]["components"]["so2"],
        "co": air_data["list"][0]["components"]["co"],
    }

    df = pd.DataFrame([data])
    output_path = os.path.join(os.getcwd(), "openweather_data.csv")

    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, index=False)

    print("✅ OpenWeather data fetched and saved successfully!")
    return df


if __name__ == "__main__":
    fetch_openweather()
