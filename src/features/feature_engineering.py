import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import yaml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIFeatureEngineering:
    """
    Feature engineering for AQI prediction models
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.lag_features = self.config['features']['lag_features']
        self.rolling_windows = self.config['features']['rolling_windows']
        self.target_ahead = self.config['features']['target_ahead']

        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.city_encoder = LabelEncoder()
        self.weather_encoder = LabelEncoder()

        # Feature columns
        self.numeric_features = [
            'temperature', 'humidity', 'pressure', 'wind_speed',
            'wind_direction', 'clouds', 'visibility', 'rain_1h',
            'snow_1h', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5',
            'pm10', 'nh3'
        ]

        self.categorical_features = [
            'city', 'weather_main'
        ]

        self.target_columns = [
            'aqi', 'calculated_aqi', 'pm2_5', 'pm10'
        ]

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()

        # Extract time components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_year'] = df['timestamp'].dt.dayofyear

        # Create cyclical features for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

        # Weekend indicator
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        return df

    def create_lag_features(self, df: pd.DataFrame, group_col: str = 'city') -> pd.DataFrame:
        """Create lag features for time series prediction"""
        df = df.copy()
        df = df.sort_values(['city', 'timestamp'])

        # Create lag features for numeric columns
        for col in self.numeric_features + self.target_columns:
            if col in df.columns:
                for lag in self.lag_features:
                    lag_col_name = f'{col}_lag_{lag}h'
                    df[lag_col_name] = df.groupby(group_col)[col].shift(lag)

        logger.info(f"Created lag features for {len(self.numeric_features + self.target_columns)} columns")
        return df

    def create_rolling_features(self, df: pd.DataFrame, group_col: str = 'city') -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        df = df.sort_values(['city', 'timestamp'])

        # Create rolling features for numeric columns
        for col in self.numeric_features + self.target_columns:
            if col in df.columns:
                for window in self.rolling_windows:
                    # Rolling mean
                    df[f'{col}_rolling_mean_{window}h'] = (
                        df.groupby(group_col)[col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )

                    # Rolling std
                    df[f'{col}_rolling_std_{window}h'] = (
                        df.groupby(group_col)[col]
                        .rolling(window=window, min_periods=1)
                        .std()
                        .reset_index(level=0, drop=True)
                    )

        logger.info(f"Created rolling features for {len(self.numeric_features + self.target_columns)} columns")
        return df

    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal and trend features"""
        df = df.copy()

        # Seasonal decomposition (simple approach)
        for city in df['city'].unique():
            city_mask = df['city'] == city
            city_data = df[city_mask].copy()

            # Simple trend (cumulative mean)
            df.loc[city_mask, 'trend'] = city_data['pm2_5'].expanding().mean()

            # Seasonal component (deviation from trend)
            df.loc[city_mask, 'seasonal'] = city_data['pm2_5'] - df.loc[city_mask, 'trend']

        return df

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()

        if fit:
            # Fit encoders
            self.city_encoder.fit(df['city'])
            self.weather_encoder.fit(df['weather_main'])

        # Transform categorical columns
        df['city_encoded'] = self.city_encoder.transform(df['city'])
        df['weather_encoded'] = self.weather_encoder.transform(df['weather_main'])

        # Create dummy variables for weather
        weather_dummies = pd.get_dummies(df['weather_main'], prefix='weather')
        df = pd.concat([df, weather_dummies], axis=1)

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between weather variables"""
        df = df.copy()

        # Temperature-humidity interaction
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100

        # Wind speed-direction interaction (simplified)
        df['wind_power'] = df['wind_speed'] ** 2

        # Pollution-weather interactions
        df['pollution_temp_ratio'] = df['pm2_5'] / (df['temperature'] + 273.15)  # Kelvin conversion

        # Visibility-pollution relationship
        df['visibility_pollution_ratio'] = df['visibility'] / (df['pm2_5'] + 1)  # Avoid division by zero

        return df

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction"""
        df = df.copy()

        # Create future targets (what we want to predict)
        for target in self.target_columns:
            target_col = f'{target}_future_{self.target_ahead}h'
            df[target_col] = df.groupby('city')[target].shift(-self.target_ahead)

        # Create classification targets (AQI categories)
        df['aqi_category'] = pd.cut(df['aqi'],
                                   bins=[0, 50, 100, 150, 200, 300, 500],
                                   labels=['good', 'moderate', 'unhealthy_sensitive',
                                          'unhealthy', 'very_unhealthy', 'hazardous'])

        # Create binary target (good vs poor air quality)
        df['poor_aqi'] = (df['aqi'] > 100).astype(int)

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df = df.copy()

        # Forward fill for time series data
        df = df.groupby('city').apply(lambda x: x.fillna(method='ffill')).reset_index(drop=True)

        # Backward fill for remaining missing values
        df = df.groupby('city').apply(lambda x: x.fillna(method='bfill')).reset_index(drop=True)

        # Fill remaining missing values with median for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        return df

    def scale_numeric_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numeric features"""
        df = df.copy()

        numeric_cols_to_scale = [col for col in self.numeric_features if col in df.columns]

        if fit:
            self.scaler.fit(df[numeric_cols_to_scale])

        scaled_features = self.scaler.transform(df[numeric_cols_to_scale])
        scaled_df = pd.DataFrame(scaled_features,
                                columns=[f"{col}_scaled" for col in numeric_cols_to_scale],
                                index=df.index)

        df = pd.concat([df, scaled_df], axis=1)

        return df

    def prepare_features_for_modeling(self, df: pd.DataFrame,
                                    target_variable: str = 'pm2_5_future_24h') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare complete feature set for modeling"""
        logger.info("Starting feature engineering pipeline...")

        # Basic preprocessing
        df = self.handle_missing_values(df)
        df = self.create_time_features(df)

        # Feature engineering
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_seasonal_features(df)
        df = self.encode_categorical_features(df)
        df = self.create_interaction_features(df)
        df = self.create_target_variables(df)

        # Remove rows with NaN targets (due to shifting)
        df = df.dropna(subset=[target_variable])

        logger.info(f"Feature engineering complete. Dataset shape: {df.shape}")

        return df

    def split_data(self, df: pd.DataFrame, target_variable: str = 'pm2_5_future_24h',
                  test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets"""
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = ['timestamp', 'city', 'weather_main', 'weather_description',
                       'aqi_category'] + [col for col in df.columns if col.endswith('_future_24h')]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[target_variable]

        # First split: train and temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size, random_state=42, shuffle=False
        )

        # Second split: validation and test
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1-val_ratio, random_state=42, shuffle=False
        )

        logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_feature_importance_template(self) -> Dict[str, str]:
        """Get feature descriptions for interpretability"""
        feature_descriptions = {
            # Time features
            'hour': 'Hour of day',
            'day_of_week': 'Day of week (0-6)',
            'month': 'Month of year',
            'is_weekend': 'Weekend indicator',

            # Weather features
            'temperature': 'Temperature in Celsius',
            'humidity': 'Relative humidity (%)',
            'pressure': 'Atmospheric pressure (hPa)',
            'wind_speed': 'Wind speed (m/s)',
            'clouds': 'Cloud cover (%)',

            # Pollution features
            'co': 'Carbon monoxide (μg/m³)',
            'no2': 'Nitrogen dioxide (μg/m³)',
            'o3': 'Ozone (μg/m³)',
            'so2': 'Sulfur dioxide (μg/m³)',
            'pm2_5': 'PM2.5 (μg/m³)',
            'pm10': 'PM10 (μg/m³)',

            # Lag features (examples)
            'pm2_5_lag_1h': 'PM2.5 value 1 hour ago',
            'pm2_5_lag_24h': 'PM2.5 value 24 hours ago',

            # Rolling features (examples)
            'temperature_rolling_mean_24h': '24-hour average temperature',
            'pm2_5_rolling_std_6h': '6-hour PM2.5 standard deviation',

            # Interaction features
            'temp_humidity_interaction': 'Temperature × humidity interaction',
            'wind_power': 'Wind speed squared',
        }

        return feature_descriptions

if __name__ == "__main__":
    # Example usage
    feature_engineer = AQIFeatureEngineering()

    # This would be called with actual data from feature store
    # processed_df = feature_engineer.prepare_features_for_modeling(raw_df)
    # X_train, X_val, X_test, y_train, y_val, y_test = feature_engineer.split_data(processed_df)

    print("Feature engineering module initialized")
    print(f"Will create lag features for: {feature_engineer.lag_features} hours")
    print(f"Will create rolling features for: {feature_engineer.rolling_windows} hours")
    print(f"Target prediction horizon: {feature_engineer.target_ahead} hours")


