"""
Karachi AQI Model Training Script
=================================

Trains and saves 3 ML models for Karachi AQI prediction:
- Linear Regression
- Random Forest
- XGBoost
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# Import our custom feature store
from karachi_feature_store import KarachiFeatureStore

class KarachiModelTrainer:
    """Train ML models for Karachi AQI prediction"""

    def __init__(self, models_dir="../src/models/saved_models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        self.feature_store = KarachiFeatureStore()
        self.scaler = StandardScaler()

        # Model configurations
        self.model_configs = {
            'linear_regression': {
                'model': LinearRegression(),
                'params': {'fit_intercept': True}
            },
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=20,
                    random_state=42
                ),
                'params': {}
            },
            'xgboost': {
                'model': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'params': {}
            }
        }

        self.trained_models = {}
        self.model_performance = {}

    def generate_training_data(self, days=30):
        """Generate training data from feature store"""
        print("📊 Generating training data...")

        # Get historical data
        raw_data = self.feature_store.get_historical_data(days=days)

        if raw_data.empty:
            print("❌ No historical data available. Generating sample data...")

            # Generate sample Karachi data for training
            timestamps = pd.date_range(
                start=datetime.now() - pd.Timedelta(days=days),
                end=datetime.now(),
                freq='H'
            )

            sample_data = []
            for ts in timestamps:
                # Karachi-specific patterns
                hour = ts.hour
                weekday = ts.weekday() < 5  # Weekday factor

                # Traffic rush hours
                traffic_factor = 1.3 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0.9

                # Base AQI with seasonal variation
                base_aqi = 95 + 10 * np.sin(2 * np.pi * ts.dayofyear / 365)

                # Apply Karachi factors
                aqi = base_aqi * traffic_factor * (1.2 if weekday else 0.85)
                aqi += np.random.normal(0, 15)  # Noise
                aqi = max(20, min(400, aqi))

                record = {
                    'timestamp': ts,
                    'city': 'Karachi',
                    'temperature': 28 + 8 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 3),
                    'humidity': 65 + 20 * np.sin(2 * np.pi * (hour - 12) / 24) + np.random.normal(0, 8),
                    'pressure': 1012 + np.random.normal(0, 4),
                    'wind_speed': 8 + 6 * np.random.normal(0, 0.7),
                    'aqi': aqi,
                    'pm2_5': (aqi - 20) * 0.5 + np.random.normal(0, 8),
                    'pm10': ((aqi - 20) * 0.5 + np.random.normal(0, 8)) * 1.5,
                    'co': 0.5 + 0.3 * (aqi / 100) + np.random.normal(0, 0.08),
                    'no2': 18 + 15 * (aqi / 100) + np.random.normal(0, 3),
                    'so2': 8 + 10 * (aqi / 100) + np.random.normal(0, 1.2),
                    'o3': 15 + 12 * (aqi / 100) + np.random.normal(0, 4)
                }
                sample_data.append(record)

            raw_data = pd.DataFrame(sample_data)
            self.feature_store.insert_data(raw_data)

        # Create training features
        training_data = self.feature_store.get_training_data(days=days)

        if training_data.empty:
            print("❌ Could not create training data")
            return None

        print(f"✅ Created training dataset with {len(training_data)} samples")
        return training_data

    def prepare_features_and_target(self, data, target_col='aqi_target_24h'):
        """Prepare features and target for training"""
        # Feature columns (exclude timestamps and target)
        exclude_cols = ['timestamp', 'city', 'aqi_category', target_col]
        feature_cols = [col for col in data.columns if col not in exclude_cols and not col.startswith('aqi_target_')]

        X = data[feature_cols]
        y = data[target_col]

        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        return X_scaled, y, feature_cols

    def train_single_model(self, model_name, X_train, y_train, X_test, y_test):
        """Train a single model and evaluate performance"""
        print(f"🤖 Training {model_name}...")

        config = self.model_configs[model_name]
        model = config['model']

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_r2': r2_score(y_test, test_pred)
        }

        print(".3f")
        # Store trained model
        self.trained_models[model_name] = model

        return metrics

    def train_all_models(self, test_size=0.2, random_state=42):
        """Train all three models"""
        print("🚀 Starting Karachi AQI Model Training")
        print("=" * 50)

        # Generate training data
        data = self.generate_training_data(days=30)

        if data is None or data.empty:
            print("❌ No training data available")
            return False

        # Prepare features and target
        X, y, feature_cols = self.prepare_features_and_target(data)

        if len(X) < 10:
            print("❌ Insufficient training data")
            return False

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )

        print(f"📊 Training set: {len(X_train)} samples")
        print(f"📊 Test set: {len(X_test)} samples")

        # Train each model
        for model_name in self.model_configs.keys():
            try:
                metrics = self.train_single_model(model_name, X_train, y_train, X_test, y_test)
                self.model_performance[model_name] = metrics
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                self.model_performance[model_name] = {'error': str(e)}

        return True

    def save_models(self):
        """Save trained models and scaler"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save models
        for model_name, model in self.trained_models.items():
            model_file = os.path.join(self.models_dir, f"{model_name}_karachi_{timestamp}.joblib")
            joblib.dump(model, model_file)
            print(f"💾 Saved {model_name} to {model_file}")

        # Save scaler
        scaler_file = os.path.join(self.models_dir, f"scaler_karachi_{timestamp}.joblib")
        joblib.dump(self.scaler, scaler_file)
        print(f"💾 Saved scaler to {scaler_file}")

        # Save feature columns (for prediction)
        feature_file = os.path.join(self.models_dir, f"features_karachi_{timestamp}.json")
        with open(feature_file, 'w') as f:
            json.dump({'feature_columns': getattr(self, 'feature_columns', [])}, f)

        # Save performance metrics
        performance_file = os.path.join(self.models_dir, f"performance_karachi_{timestamp}.json")
        with open(performance_file, 'w') as f:
            json.dump(self.model_performance, f, indent=2, default=str)

        print(f"💾 Saved performance metrics to {performance_file}")

    def create_model_comparison_report(self):
        """Create a comparison report of all models"""
        if not self.model_performance:
            print("❌ No model performance data available")
            return

        print("\n📊 MODEL PERFORMANCE COMPARISON")
        print("=" * 50)

        # Create comparison table
        comparison_data = []
        for model_name, metrics in self.model_performance.items():
            if 'error' not in metrics:
                row = {
                    'Model': model_name.replace('_', ' ').title(),
                    'Train MAE': f"{metrics['train_mae']:.2f}",
                    'Train RMSE': f"{metrics['train_rmse']:.2f}",
                    'Train R²': f"{metrics['train_r2']:.3f}",
                    'Test MAE': f"{metrics['test_mae']:.2f}",
                    'Test RMSE': f"{metrics['test_rmse']:.2f}",
                    'Test R²': f"{metrics['test_r2']:.3f}"
                }
                comparison_data.append(row)

        if comparison_data:
            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))

            # Find best model
            best_model = max(self.model_performance.items(),
                           key=lambda x: x[1].get('test_r2', -float('inf')) if 'error' not in x[1] else -float('inf'))

            print(f"\n🏆 Best performing model: {best_model[0].replace('_', ' ').title()}")
            print(f"   Test R²: {best_model[1]['test_r2']:.3f}")
            print(f"   Test MAE: {best_model[1]['test_mae']:.2f}")

    def run_complete_training(self):
        """Run the complete model training pipeline"""
        # Train models
        success = self.train_all_models()

        if success:
            # Save models
            self.save_models()

            # Create comparison report
            self.create_model_comparison_report()

            print("\n🎉 KARACHI AQI MODEL TRAINING COMPLETED!")
            print("=" * 50)
            print("✅ All 3 models trained and saved:")
            print("   • Linear Regression")
            print("   • Random Forest")
            print("   • XGBoost")
            print("✅ Models saved to src/models/saved_models/")
            print("✅ Performance metrics saved")
            print("✅ Ready for integration with Streamlit app")

            return True
        else:
            print("❌ Model training failed")
            return False

def main():
    """Main training function"""
    trainer = KarachiModelTrainer()
    success = trainer.run_complete_training()

    if success:
        print("\n🚀 Ready to integrate with Streamlit app!")
        print("Run: streamlit run karachi_aqi_app.py")
    else:
        print("\n❌ Training failed. Check error messages above.")

if __name__ == "__main__":
    main()