"""
Karachi AQI Feature Store - Local Implementation
===============================================

A local feature store implementation for Karachi AQI data storage and retrieval.
Can be easily extended to work with Hopsworks or other feature stores.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KarachiFeatureStore:
    """
    Local feature store for Karachi AQI data
    """

    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Feature store files
        self.raw_data_file = self.data_dir / "karachi_raw_data.csv"
        self.processed_data_file = self.data_dir / "karachi_processed_data.csv"
        self.metadata_file = self.data_dir / "karachi_metadata.json"

        # Initialize metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        """Load feature store metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'city': 'Karachi',
                'country': 'Pakistan',
                'total_records': 0,
                'last_updated': None,
                'data_quality': {},
                'features': []
            }

    def _save_metadata(self):
        """Save feature store metadata"""
        self.metadata['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def insert_data(self, data_df):
        """
        Insert new data into the feature store

        Args:
            data_df: DataFrame with Karachi AQI data
        """
        try:
            # Validate data
            required_columns = ['timestamp', 'temperature', 'humidity', 'aqi', 'pm2_5']
            missing_columns = [col for col in required_columns if col not in data_df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Ensure city column exists
            if 'city' not in data_df.columns:
                data_df['city'] = 'Karachi'

            # Load existing data if available
            existing_data = pd.DataFrame()
            if self.raw_data_file.exists():
                existing_data = pd.read_csv(self.raw_data_file)
                existing_data['timestamp'] = pd.to_datetime(existing_data['timestamp'])

            # Combine with new data
            combined_data = pd.concat([existing_data, data_df], ignore_index=True)

            # Remove duplicates based on timestamp
            combined_data = combined_data.drop_duplicates(subset=['timestamp'], keep='last')

            # Sort by timestamp
            combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)

            # Save to feature store
            combined_data.to_csv(self.raw_data_file, index=False)

            # Update metadata
            self.metadata['total_records'] = len(combined_data)
            self.metadata['features'] = list(combined_data.columns)
            self.metadata['data_quality'] = {
                'missing_values': combined_data.isnull().sum().to_dict(),
                'date_range': {
                    'start': combined_data['timestamp'].min().isoformat() if not combined_data.empty else None,
                    'end': combined_data['timestamp'].max().isoformat() if not combined_data.empty else None
                }
            }
            self._save_metadata()

            logger.info(f"Inserted {len(data_df)} records into Karachi feature store")
            return True

        except Exception as e:
            logger.error(f"Error inserting data: {e}")
            return False

    def get_recent_data(self, hours=24):
        """
        Get recent data from feature store

        Args:
            hours: Number of hours of data to retrieve

        Returns:
            DataFrame with recent data
        """
        try:
            if not self.raw_data_file.exists():
                logger.warning("No data available in feature store")
                return pd.DataFrame()

            data = pd.read_csv(self.raw_data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Filter by time range
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = data[data['timestamp'] >= cutoff_time]

            logger.info(f"Retrieved {len(recent_data)} records from last {hours} hours")
            return recent_data

        except Exception as e:
            logger.error(f"Error retrieving recent data: {e}")
            return pd.DataFrame()

    def get_historical_data(self, days=30):
        """
        Get historical data for model training

        Args:
            days: Number of days of historical data

        Returns:
            DataFrame with historical data
        """
        try:
            if not self.raw_data_file.exists():
                logger.warning("No historical data available")
                return pd.DataFrame()

            data = pd.read_csv(self.raw_data_file)
            data['timestamp'] = pd.to_datetime(data['timestamp'])

            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days)
            historical_data = data[data['timestamp'] >= cutoff_date]

            logger.info(f"Retrieved {len(historical_data)} historical records from last {days} days")
            return historical_data

        except Exception as e:
            logger.error(f"Error retrieving historical data: {e}")
            return pd.DataFrame()

    def get_training_data(self, days=30, target_ahead=24):
        """
        Get training data with target variables

        Args:
            days: Historical data period
            target_ahead: Hours ahead to predict

        Returns:
            DataFrame ready for model training
        """
        try:
            data = self.get_historical_data(days)

            if data.empty:
                return pd.DataFrame()

            # Sort by timestamp
            data = data.sort_values('timestamp').reset_index(drop=True)

            # Create target variable (AQI 24 hours ahead)
            data[f'aqi_target_{target_ahead}h'] = data['aqi'].shift(-target_ahead)

            # Remove rows with NaN targets
            data = data.dropna(subset=[f'aqi_target_{target_ahead}h'])

            # Create time-based features
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['month'] = data['timestamp'].dt.month

            # Cyclical encoding for time features
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

            # Lag features
            lag_features = [1, 3, 6, 12, 24]
            for lag in lag_features:
                if len(data) > lag:
                    data[f'aqi_lag_{lag}h'] = data['aqi'].shift(lag)
                    data[f'pm25_lag_{lag}h'] = data['pm2_5'].shift(lag)
                    data[f'temp_lag_{lag}h'] = data['temperature'].shift(lag)

            # Remove rows with NaN values after creating lag features
            data = data.dropna()

            # Save processed data
            data.to_csv(self.processed_data_file, index=False)

            logger.info(f"Created training dataset with {len(data)} samples and {len(data.columns)} features")
            return data

        except Exception as e:
            logger.error(f"Error creating training data: {e}")
            return pd.DataFrame()

    def get_feature_descriptions(self):
        """Get descriptions of all features"""
        return {
            # Raw features
            'timestamp': 'Data collection timestamp',
            'temperature': 'Temperature in Celsius',
            'humidity': 'Relative humidity percentage',
            'pressure': 'Atmospheric pressure in hPa',
            'wind_speed': 'Wind speed in m/s',
            'aqi': 'Current Air Quality Index',
            'pm2_5': 'PM2.5 concentration (μg/m³)',
            'pm10': 'PM10 concentration (μg/m³)',
            'co': 'Carbon monoxide (μg/m³)',
            'no2': 'Nitrogen dioxide (μg/m³)',
            'o3': 'Ozone (μg/m³)',
            'so2': 'Sulfur dioxide (μg/m³)',

            # Time features
            'hour': 'Hour of day (0-23)',
            'day_of_week': 'Day of week (0-6, Monday=0)',
            'month': 'Month of year (1-12)',
            'hour_sin': 'Hour sine component (cyclical)',
            'hour_cos': 'Hour cosine component (cyclical)',

            # Lag features
            'aqi_lag_1h': 'AQI value 1 hour ago',
            'aqi_lag_3h': 'AQI value 3 hours ago',
            'aqi_lag_6h': 'AQI value 6 hours ago',
            'aqi_lag_12h': 'AQI value 12 hours ago',
            'aqi_lag_24h': 'AQI value 24 hours ago',

            # Target features
            'aqi_target_24h': 'AQI 24 hours ahead (target variable)'
        }

    def get_statistics(self):
        """Get feature store statistics"""
        try:
            stats = {
                'total_records': self.metadata.get('total_records', 0),
                'features_count': len(self.metadata.get('features', [])),
                'last_updated': self.metadata.get('last_updated'),
                'data_quality': self.metadata.get('data_quality', {})
            }

            # Additional statistics if data exists
            if self.raw_data_file.exists():
                data = pd.read_csv(self.raw_data_file)
                stats['aqi_stats'] = {
                    'mean': float(data['aqi'].mean()),
                    'std': float(data['aqi'].std()),
                    'min': float(data['aqi'].min()),
                    'max': float(data['aqi'].max()),
                    'median': float(data['aqi'].median())
                }

                # AQI categories distribution
                def categorize_aqi(aqi):
                    if aqi <= 50: return 'Good'
                    elif aqi <= 100: return 'Moderate'
                    elif aqi <= 150: return 'Unhealthy for Sensitive Groups'
                    elif aqi <= 200: return 'Unhealthy'
                    elif aqi <= 300: return 'Very Unhealthy'
                    else: return 'Hazardous'

                categories = data['aqi'].apply(categorize_aqi).value_counts()
                stats['aqi_categories'] = categories.to_dict()

            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def clear_data(self):
        """Clear all data from feature store (for testing)"""
        try:
            if self.raw_data_file.exists():
                self.raw_data_file.unlink()
            if self.processed_data_file.exists():
                self.processed_data_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()

            self.metadata = {
                'city': 'Karachi',
                'country': 'Pakistan',
                'total_records': 0,
                'last_updated': None,
                'data_quality': {},
                'features': []
            }

            logger.info("Feature store cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    fs = KarachiFeatureStore()

    # Generate sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=10), periods=10, freq='H'),
        'temperature': np.random.normal(30, 3, 10),
        'humidity': np.random.normal(65, 8, 10),
        'aqi': np.random.normal(100, 20, 10),
        'pm2_5': np.random.normal(45, 15, 10)
    })

    # Insert data
    fs.insert_data(sample_data)

    # Get statistics
    stats = fs.get_statistics()
    print("Feature Store Statistics:")
    print(json.dumps(stats, indent=2, default=str))


