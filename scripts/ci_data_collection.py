#!/usr/bin/env python3
"""
CI/CD Data Collection Script for Karachi AQI
This script is specifically designed for GitHub Actions workflow
"""

import os
import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
import json

class CICIDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        self.air_quality_url = "http://api.openweathermap.org/data/2.5/air_pollution"

        # Karachi coordinates
        self.lat = 24.8607
        self.lon = 67.0011

        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        self.data_file = 'data/karachi_raw_data.csv'

    def get_current_weather(self):
        """Get current weather data for Karachi"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key,
                'units': 'metric'
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            weather_data = {
                'timestamp': datetime.now(),
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'wind_direction': data['wind'].get('deg', 0),
                'visibility': data.get('visibility', 10000),
                'weather_main': data['weather'][0]['main'],
                'weather_description': data['weather'][0]['description'],
                'clouds': data['clouds']['all']
            }

            return weather_data

        except Exception as e:
            print(f"❌ Error getting weather data: {e}")
            return None

    def get_air_quality(self):
        """Get air quality data for Karachi"""
        try:
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key
            }

            response = requests.get(self.air_quality_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # AQI categories mapping
            aqi_categories = {
                1: "Good",
                2: "Fair",
                3: "Moderate",
                4: "Poor",
                5: "Very Poor"
            }

            aqi_data = data['list'][0]
            main_aqi = aqi_data['main']['aqi']

            pollution_data = {
                'aqi': main_aqi,
                'aqi_category': aqi_categories.get(main_aqi, "Unknown"),
                'co': aqi_data['components']['co'],
                'no': aqi_data['components']['no'],
                'no2': aqi_data['components']['no2'],
                'o3': aqi_data['components']['o3'],
                'so2': aqi_data['components']['so2'],
                'pm2_5': aqi_data['components']['pm2_5'],
                'pm10': aqi_data['components']['pm10'],
                'nh3': aqi_data['components']['nh3']
            }

            return pollution_data

        except Exception as e:
            print(f"❌ Error getting air quality data: {e}")
            return None

    def collect_and_store_data(self):
        """Collect both weather and air quality data and store in CSV"""
        print("🌤️ Collecting current weather data for Karachi...")

        # Get weather data
        weather_data = self.get_current_weather()
        if not weather_data:
            print("❌ Failed to get weather data")
            return False

        # Get air quality data
        aqi_data = self.get_air_quality()
        if not aqi_data:
            print("❌ Failed to get air quality data")
            return False

        # Combine data
        complete_data = {**weather_data, **aqi_data}

        # Convert to DataFrame
        df = pd.DataFrame([complete_data])

        # Load existing data if file exists
        if os.path.exists(self.data_file):
            try:
                existing_df = pd.read_csv(self.data_file)
                # Convert timestamp column to datetime if it exists
                if 'timestamp' in existing_df.columns:
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                print(f"⚠️ Warning: Could not load existing data: {e}")

        # Remove duplicates based on timestamp (keep the latest)
        if len(df) > 1:
            df = df.drop_duplicates(subset=['timestamp'], keep='last')

        # Sort by timestamp
        df = df.sort_values('timestamp', ascending=False)

        # Save to CSV
        df.to_csv(self.data_file, index=False)

        print(f"✅ Successfully collected and stored data!")
        print(f"   AQI: {complete_data['aqi']} ({complete_data['aqi_category']})")
        print(f"   Temperature: {complete_data['temperature']}°C")
        print(f"   Humidity: {complete_data['humidity']}%")
        print(f"   Total records: {len(df)}")

        return True

def main():
    """Main function for CI/CD data collection"""
    api_key = os.getenv('OPENWEATHER_API_KEY')

    # Temporary fallback for testing - REMOVE THIS IN PRODUCTION
    if not api_key:
        print("⚠️  OPENWEATHER_API_KEY environment variable not set")
        print("🔧 Using fallback API key for testing (REMOVE IN PRODUCTION)")
        api_key = "da06b92d3139ce209b04dba2132ad4ce"  # Temporary fallback

    if not api_key:
        print("❌ No API key available - please set OPENWEATHER_API_KEY")
        sys.exit(1)

    print("🚀 Starting CI/CD data collection for Karachi AQI...")

    collector = CICIDataCollector(api_key)
    success = collector.collect_and_store_data()

    if success:
        print("🎉 Data collection completed successfully!")
        sys.exit(0)
    else:
        print("💥 Data collection failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
