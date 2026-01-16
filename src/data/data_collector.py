import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import logging
import yaml
import os
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIWeatherDataCollector:
    """
    Collects weather and air quality data from OpenWeather API
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.api_key = self.config['openweather']['api_key']
        self.base_url = self.config['openweather']['base_url']
        self.cities = self.config['data_collection']['cities']

        # AQI breakpoints for PM2.5 (μg/m³)
        self.aqi_breakpoints = {
            'pm25': [
                (0, 12.0, 0, 50),
                (12.1, 35.4, 51, 100),
                (35.5, 55.4, 101, 150),
                (55.5, 150.4, 151, 200),
                (150.5, 250.4, 201, 300),
                (250.5, 350.4, 301, 400),
                (350.5, 500.4, 401, 500)
            ]
        }

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def get_city_coordinates(self, city: str) -> Optional[tuple]:
        """Get latitude and longitude for a city"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            return data['coord']['lat'], data['coord']['lon']

        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting coordinates for {city}: {e}")
            return None

    def collect_weather_data(self, city: str) -> Optional[Dict]:
        """Collect current weather data for a city"""
        try:
            coords = self.get_city_coordinates(city)
            if not coords:
                return None

            lat, lon = coords

            # Get current weather
            weather_url = f"{self.base_url}/weather"
            weather_params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }

            weather_response = requests.get(weather_url, params=weather_params)
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            # Get air pollution data
            pollution_url = f"{self.base_url}/air_pollution"
            pollution_params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key
            }

            pollution_response = requests.get(pollution_url, params=pollution_params)
            pollution_response.raise_for_status()
            pollution_data = pollution_response.json()

            # Combine data
            combined_data = {
                'city': city,
                'timestamp': datetime.now(),
                'latitude': lat,
                'longitude': lon,
                'temperature': weather_data['main']['temp'],
                'humidity': weather_data['main']['humidity'],
                'pressure': weather_data['main']['pressure'],
                'wind_speed': weather_data['wind']['speed'],
                'wind_direction': weather_data['wind'].get('deg', 0),
                'weather_main': weather_data['weather'][0]['main'],
                'weather_description': weather_data['weather'][0]['description'],
                'clouds': weather_data['clouds']['all'],
                'visibility': weather_data.get('visibility', 0),
                'rain_1h': weather_data.get('rain', {}).get('1h', 0),
                'snow_1h': weather_data.get('snow', {}).get('1h', 0),
                'aqi': pollution_data['list'][0]['main']['aqi'],
                'co': pollution_data['list'][0]['components']['co'],
                'no': pollution_data['list'][0]['components']['no'],
                'no2': pollution_data['list'][0]['components']['no2'],
                'o3': pollution_data['list'][0]['components']['o3'],
                'so2': pollution_data['list'][0]['components']['so2'],
                'pm2_5': pollution_data['list'][0]['components']['pm2_5'],
                'pm10': pollution_data['list'][0]['components']['pm10'],
                'nh3': pollution_data['list'][0]['components']['nh3']
            }

            return combined_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error collecting data for {city}: {e}")
            return None
        except KeyError as e:
            logger.error(f"Missing key in API response for {city}: {e}")
            return None

    def calculate_aqi_from_pm25(self, pm25: float) -> int:
        """Calculate AQI from PM2.5 concentration"""
        for low_conc, high_conc, low_aqi, high_aqi in self.aqi_breakpoints['pm25']:
            if low_conc <= pm25 <= high_conc:
                aqi = ((high_aqi - low_aqi) / (high_conc - low_conc)) * (pm25 - low_conc) + low_aqi
                return round(aqi)
        return 500  # Maximum AQI

    def collect_data_for_all_cities(self) -> pd.DataFrame:
        """Collect data for all configured cities"""
        all_data = []

        for city in self.cities:
            logger.info(f"Collecting data for {city}")
            data = self.collect_weather_data(city)
            if data:
                all_data.append(data)
            time.sleep(1)  # Rate limiting

        if all_data:
            df = pd.DataFrame(all_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['calculated_aqi'] = df['pm2_5'].apply(self.calculate_aqi_from_pm25)
            return df
        else:
            logger.warning("No data collected for any city")
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """Save collected data to CSV file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/aqi_weather_data_{timestamp}.csv"

        os.makedirs('data', exist_ok=True)
        df.to_csv(filename, index=False)
        logger.info(f"Data saved to {filename}")
        return filename

    def run_collection_cycle(self) -> pd.DataFrame:
        """Run a complete data collection cycle"""
        logger.info("Starting data collection cycle")
        df = self.collect_data_for_all_cities()

        if not df.empty:
            filename = self.save_data(df)
            logger.info(f"Collected data for {len(df)} cities")
        else:
            logger.warning("No data collected in this cycle")

        return df

if __name__ == "__main__":
    collector = AQIWeatherDataCollector()
    data = collector.run_collection_cycle()
    print(f"Collected data shape: {data.shape}")
    print(data.head())


