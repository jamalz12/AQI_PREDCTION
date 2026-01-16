import hopsworks
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import yaml
from typing import Dict, List, Optional
import great_expectations as ge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIFeatureStore:
    """
    Manages feature storage and retrieval using Hopsworks Feature Store
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.project_name = self.config['hopsworks']['project_name']
        self.api_key = self.config['hopsworks']['api_key']
        self.host = self.config['hopsworks']['host']

        self.project = None
        self.fs = None
        self.feature_group = None

        self._connect_to_hopsworks()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _connect_to_hopsworks(self):
        """Connect to Hopsworks project and feature store"""
        try:
            if self.api_key:
                hopsworks.login(api_key_value=self.api_key, host=self.host)
            else:
                hopsworks.login(host=self.host)

            self.project = hopsworks.connection().get_project(self.project_name)
            self.fs = self.project.get_feature_store()

            logger.info(f"Connected to Hopsworks project: {self.project_name}")

        except Exception as e:
            logger.error(f"Failed to connect to Hopsworks: {e}")
            raise

    def create_feature_group(self, name: str = "aqi_weather_features",
                           version: int = 1) -> None:
        """Create or get existing feature group for AQI data"""
        try:
            # Define feature group schema
            schema = {
                "city": "string",
                "timestamp": "timestamp",
                "latitude": "double",
                "longitude": "double",
                "temperature": "double",
                "humidity": "int",
                "pressure": "int",
                "wind_speed": "double",
                "wind_direction": "int",
                "weather_main": "string",
                "weather_description": "string",
                "clouds": "int",
                "visibility": "int",
                "rain_1h": "double",
                "snow_1h": "double",
                "aqi": "int",
                "co": "double",
                "no": "double",
                "no2": "double",
                "o3": "double",
                "so2": "double",
                "pm2_5": "double",
                "pm10": "double",
                "nh3": "double",
                "calculated_aqi": "int"
            }

            # Try to get existing feature group
            try:
                self.feature_group = self.fs.get_feature_group(name, version=version)
                logger.info(f"Using existing feature group: {name}_v{version}")
            except:
                # Create new feature group
                self.feature_group = self.fs.create_feature_group(
                    name=name,
                    version=version,
                    description="Air Quality Index and Weather Features",
                    primary_key=["city", "timestamp"],
                    event_time="timestamp",
                    online_enabled=True
                )
                logger.info(f"Created new feature group: {name}_v{version}")

        except Exception as e:
            logger.error(f"Error creating/getting feature group: {e}")
            raise

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality using Great Expectations"""
        try:
            # Convert to Great Expectations dataframe
            ge_df = ge.from_pandas(df)

            # Basic validations
            validations = [
                ge_df.expect_column_to_exist("timestamp"),
                ge_df.expect_column_to_exist("city"),
                ge_df.expect_column_values_to_not_be_null("timestamp"),
                ge_df.expect_column_values_to_not_be_null("city"),
                ge_df.expect_column_values_to_be_between("temperature", -50, 60),
                ge_df.expect_column_values_to_be_between("humidity", 0, 100),
                ge_df.expect_column_values_to_be_between("aqi", 1, 5),
                ge_df.expect_column_values_to_be_between("pm2_5", 0, 1000)
            ]

            # Check if all validations pass
            failed_validations = []
            for validation in validations:
                if not validation.success:
                    failed_validations.append(validation)

            if failed_validations:
                logger.warning(f"Data validation failed for {len(failed_validations)} checks")
                for validation in failed_validations:
                    logger.warning(f"Validation failed: {validation.expectation_config['expectation_type']}")

            return df

        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            return df

    def insert_data(self, df: pd.DataFrame) -> None:
        """Insert data into feature store"""
        try:
            if self.feature_group is None:
                self.create_feature_group()

            # Validate data before insertion
            validated_df = self.validate_data(df)

            # Insert data
            self.feature_group.insert(validated_df)
            logger.info(f"Inserted {len(validated_df)} rows into feature store")

        except Exception as e:
            logger.error(f"Error inserting data to feature store: {e}")
            raise

    def get_recent_data(self, hours: int = 24) -> pd.DataFrame:
        """Retrieve recent data from feature store"""
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            # Query feature group
            query = self.feature_group.select_all()
            df = query.filter(self.feature_group.timestamp >= start_time).read()

            logger.info(f"Retrieved {len(df)} rows of recent data")
            return df

        except Exception as e:
            logger.error(f"Error retrieving recent data: {e}")
            return pd.DataFrame()

    def get_training_data(self, days: int = 30) -> pd.DataFrame:
        """Get historical data for model training"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            query = self.feature_group.select_all()
            df = query.filter(
                (self.feature_group.timestamp >= start_time) &
                (self.feature_group.timestamp <= end_time)
            ).read()

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Retrieved {len(df)} rows for training")
            return df

        except Exception as e:
            logger.error(f"Error retrieving training data: {e}")
            return pd.DataFrame()

    def get_city_data(self, city: str, days: int = 7) -> pd.DataFrame:
        """Get data for a specific city"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            query = self.feature_group.select_all()
            df = query.filter(
                (self.feature_group.city == city) &
                (self.feature_group.timestamp >= start_time)
            ).read()

            df = df.sort_values('timestamp').reset_index(drop=True)
            logger.info(f"Retrieved {len(df)} rows for city {city}")
            return df

        except Exception as e:
            logger.error(f"Error retrieving data for city {city}: {e}")
            return pd.DataFrame()

    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all features"""
        descriptions = {
            "city": "City name",
            "timestamp": "Data collection timestamp",
            "latitude": "Geographic latitude",
            "longitude": "Geographic longitude",
            "temperature": "Temperature in Celsius",
            "humidity": "Relative humidity percentage",
            "pressure": "Atmospheric pressure in hPa",
            "wind_speed": "Wind speed in m/s",
            "wind_direction": "Wind direction in degrees",
            "weather_main": "Main weather condition",
            "weather_description": "Detailed weather description",
            "clouds": "Cloudiness percentage",
            "visibility": "Visibility in meters",
            "rain_1h": "Rain volume for last hour in mm",
            "snow_1h": "Snow volume for last hour in mm",
            "aqi": "Air Quality Index (1-5 scale)",
            "co": "Carbon monoxide concentration",
            "no": "Nitrogen monoxide concentration",
            "no2": "Nitrogen dioxide concentration",
            "o3": "Ozone concentration",
            "so2": "Sulfur dioxide concentration",
            "pm2_5": "PM2.5 concentration in μg/m³",
            "pm10": "PM10 concentration in μg/m³",
            "nh3": "Ammonia concentration",
            "calculated_aqi": "AQI calculated from PM2.5"
        }
        return descriptions

if __name__ == "__main__":
    # Example usage
    fs = AQIFeatureStore()

    # This would be called after collecting data
    # fs.insert_data(collected_dataframe)

    # Get feature descriptions
    descriptions = fs.get_feature_descriptions()
    print("Available features:")
    for feature, desc in descriptions.items():
        print(f"- {feature}: {desc}")


