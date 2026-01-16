import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenWeatherAPI:
    """
    Wrapper for OpenWeather API integration
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.api_key = self.config['openweather']['api_key']
        self.base_url = self.config['openweather']['base_url']

        # API endpoints
        self.endpoints = self.config['openweather']['endpoints']

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            params['appid'] = self.api_key

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.Timeout:
            logger.error("API request timed out")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")

        return None

    def get_current_weather(self, city: str, units: str = 'metric') -> Optional[Dict]:
        """Get current weather data for a city"""
        params = {
            'q': city,
            'units': units
        }

        data = self._make_request(self.endpoints['current'], params)

        if data:
            # Add timestamp
            data['timestamp'] = datetime.now()

            logger.info(f"Retrieved current weather for {city}")
            return data
        else:
            logger.warning(f"Failed to retrieve current weather for {city}")
            return None

    def get_weather_forecast(self, city: str, units: str = 'metric') -> Optional[Dict]:
        """Get 5-day weather forecast for a city"""
        # First get coordinates
        current_weather = self.get_current_weather(city)
        if not current_weather:
            return None

        lat = current_weather['coord']['lat']
        lon = current_weather['coord']['lon']

        params = {
            'lat': lat,
            'lon': lon,
            'units': units
        }

        data = self._make_request(self.endpoints['forecast'], params)

        if data:
            # Add timestamps
            for item in data.get('list', []):
                item['timestamp'] = datetime.fromtimestamp(item['dt'])

            logger.info(f"Retrieved forecast for {city}")
            return data
        else:
            logger.warning(f"Failed to retrieve forecast for {city}")
            return None

    def get_air_pollution(self, city: str) -> Optional[Dict]:
        """Get current air pollution data for a city"""
        # First get coordinates
        current_weather = self.get_current_weather(city)
        if not current_weather:
            return None

        lat = current_weather['coord']['lat']
        lon = current_weather['coord']['lon']

        params = {
            'lat': lat,
            'lon': lon
        }

        data = self._make_request(self.endpoints['air_pollution'], params)

        if data:
            # Add timestamp
            data['timestamp'] = datetime.now()

            logger.info(f"Retrieved air pollution data for {city}")
            return data
        else:
            logger.warning(f"Failed to retrieve air pollution data for {city}")
            return None

    def get_complete_weather_data(self, city: str) -> Optional[Dict]:
        """Get complete weather and air quality data for a city"""
        try:
            # Get current weather
            weather_data = self.get_current_weather(city)
            if not weather_data:
                return None

            # Get air pollution data
            pollution_data = self.get_air_pollution(city)
            if not pollution_data:
                return None

            # Combine data
            complete_data = {
                'city': city,
                'timestamp': datetime.now(),
                'coordinates': {
                    'lat': weather_data['coord']['lat'],
                    'lon': weather_data['coord']['lon']
                },
                'weather': {
                    'temperature': weather_data['main']['temp'],
                    'humidity': weather_data['main']['humidity'],
                    'pressure': weather_data['main']['pressure'],
                    'temp_min': weather_data['main']['temp_min'],
                    'temp_max': weather_data['main']['temp_max'],
                    'wind_speed': weather_data['wind']['speed'],
                    'wind_direction': weather_data['wind'].get('deg', 0),
                    'clouds': weather_data['clouds']['all'],
                    'visibility': weather_data.get('visibility', 0),
                    'weather_main': weather_data['weather'][0]['main'],
                    'weather_description': weather_data['weather'][0]['description'],
                    'rain_1h': weather_data.get('rain', {}).get('1h', 0),
                    'snow_1h': weather_data.get('snow', {}).get('1h', 0)
                },
                'air_quality': {
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
            }

            logger.info(f"Retrieved complete data for {city}")
            return complete_data

        except Exception as e:
            logger.error(f"Error getting complete weather data for {city}: {e}")
            return None

    def get_multiple_cities_data(self, cities: List[str]) -> List[Dict]:
        """Get data for multiple cities"""
        results = []

        for city in cities:
            data = self.get_complete_weather_data(city)
            if data:
                results.append(data)

        logger.info(f"Retrieved data for {len(results)}/{len(cities)} cities")
        return results

    def get_aqi_category(self, aqi_value: int) -> Tuple[str, str]:
        """Get AQI category and health implications"""
        categories = {
            1: ("Good", "Air quality is satisfactory, and air pollution poses little or no risk."),
            2: ("Moderate", "Air quality is acceptable. However, there may be a risk for some people."),
            3: ("Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects."),
            4: ("Unhealthy", "Everyone may begin to experience health effects."),
            5: ("Very Unhealthy", "Health alert: everyone may experience more serious health effects."),
            6: ("Hazardous", "Health warning of emergency conditions. The entire population is more likely to be affected.")
        }

        # Ensure AQI is within valid range
        aqi_value = max(1, min(6, aqi_value))

        category, description = categories[aqi_value]
        return category, description

    def calculate_aqi_from_pm25(self, pm25: float) -> int:
        """Calculate AQI from PM2.5 concentration using EPA formula"""
        # AQI breakpoints for PM2.5 (μg/m³)
        breakpoints = [
            (0.0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500)
        ]

        for low_conc, high_conc, low_aqi, high_aqi in breakpoints:
            if low_conc <= pm25 <= high_conc:
                aqi = ((high_aqi - low_aqi) / (high_conc - low_conc)) * (pm25 - low_conc) + low_aqi
                return round(aqi)

        return 500  # Maximum AQI

if __name__ == "__main__":
    # Example usage
    api = OpenWeatherAPI()

    # Test with a city
    city = "Delhi"
    data = api.get_complete_weather_data(city)

    if data:
        print(f"Weather data for {city}:")
        print(f"Temperature: {data['weather']['temperature']}°C")
        print(f"AQI: {data['air_quality']['aqi']}")
        print(f"PM2.5: {data['air_quality']['pm2_5']} μg/m³")

        # Calculate AQI from PM2.5
        calculated_aqi = api.calculate_aqi_from_pm25(data['air_quality']['pm2_5'])
        print(f"Calculated AQI: {calculated_aqi}")

        category, description = api.get_aqi_category(data['air_quality']['aqi'])
        print(f"AQI Category: {category}")
        print(f"Description: {description}")
    else:
        print(f"Failed to retrieve data for {city}")


