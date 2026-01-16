import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
import yaml
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AQIModelTrainer:
    """
    Train and manage multiple AQI prediction models
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.models = {}
        self.model_configs = self.config['models']
        self.training_config = self.config['training']

        # Create models directory
        os.makedirs('src/models/saved_models', exist_ok=True)

        # Initialize models
        self._initialize_models()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def _initialize_models(self):
        """Initialize all configured models"""
        if self.model_configs['linear_regression']['enabled']:
            self.models['linear_regression'] = LinearRegression(
                **self.model_configs['linear_regression']['hyperparameters']
            )

        if self.model_configs['random_forest']['enabled']:
            self.models['random_forest'] = RandomForestRegressor(
                **self.model_configs['random_forest']['hyperparameters']
            )

        if self.model_configs['xgboost']['enabled']:
            self.models['xgboost'] = XGBRegressor(
                **self.model_configs['xgboost']['hyperparameters']
            )

        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")

    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Train a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        logger.info(f"Training {model_name}...")

        model = self.models[model_name]

        # Train the model
        if X_val is not None and y_val is not None:
            # Use validation set for early stopping if supported
            if hasattr(model, 'fit') and 'early_stopping_rounds' in model.get_params():
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                         early_stopping_rounds=self.training_config['early_stopping_rounds'],
                         verbose=False)
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # Make predictions
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val) if X_val is not None else None

        # Calculate metrics
        metrics = self._calculate_metrics(y_train, train_pred, y_val, val_pred)

        logger.info(f"{model_name} training completed. Train R²: {metrics['train_r2']:.4f}")

        if val_pred is not None:
            logger.info(f"{model_name} validation R²: {metrics['val_r2']:.4f}")

        return {
            'model': model,
            'metrics': metrics,
            'feature_importance': self._get_feature_importance(model, X_train.columns) if hasattr(model, 'feature_importances_') else None
        }

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict]:
        """Train all enabled models"""
        results = {}

        for model_name in self.models.keys():
            try:
                result = self.train_model(model_name, X_train, y_train, X_val, y_val)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {'error': str(e)}

        return results

    def _calculate_metrics(self, y_train: pd.Series, train_pred: np.ndarray,
                          y_val: Optional[pd.Series] = None, val_pred: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred)
        }

        if y_val is not None and val_pred is not None:
            metrics.update({
                'val_mae': mean_absolute_error(y_val, val_pred),
                'val_mse': mean_squared_error(y_val, val_pred),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'val_r2': r2_score(y_val, val_pred)
            })

        return metrics

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(model, 'coef_'):
            # For linear models
            importance_dict = dict(zip(feature_names, np.abs(model.coef_)))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}

    def evaluate_on_test(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate a trained model on test data"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        test_pred = model.predict(X_test)

        metrics = {
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_r2': r2_score(y_test, test_pred)
        }

        logger.info(f"{model_name} test metrics: MAE={metrics['test_mae']:.4f}, R²={metrics['test_r2']:.4f}")

        return metrics

    def save_model(self, model_name: str, model_result: Dict, version: Optional[str] = None) -> str:
        """Save trained model to disk"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"src/models/saved_models/{model_name}_{version}.joblib"
        joblib.dump(model_result['model'], filename)

        logger.info(f"Model {model_name} saved to {filename}")
        return filename

    def load_model(self, model_name: str, version: str) -> Any:
        """Load trained model from disk"""
        filename = f"src/models/saved_models/{model_name}_{version}.joblib"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found")

        model = joblib.load(filename)
        logger.info(f"Model {model_name} loaded from {filename}")
        return model

    def log_to_mlflow(self, model_name: str, model_result: Dict[str, Any],
                     X_train: pd.DataFrame, run_name: Optional[str] = None):
        """Log model and metrics to MLflow"""
        try:
            with mlflow.start_run(run_name=run_name or f"{model_name}_training"):
                # Log parameters
                if model_name in self.model_configs:
                    for param, value in self.model_configs[model_name]['hyperparameters'].items():
                        mlflow.log_param(param, value)

                # Log metrics
                for metric_name, metric_value in model_result['metrics'].items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log feature importance if available
                if model_result.get('feature_importance'):
                    # Log top 20 features
                    top_features = list(model_result['feature_importance'].items())[:20]
                    for feature, importance in top_features:
                        mlflow.log_metric(f"feature_importance_{feature}", importance)

                # Log model
                if model_name == 'xgboost':
                    mlflow.xgboost.log_model(model_result['model'], "model")
                else:
                    mlflow.sklearn.log_model(model_result['model'], "model")

                logger.info(f"Logged {model_name} to MLflow")

        except Exception as e:
            logger.error(f"Error logging to MLflow: {e}")

    def get_model_comparison(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Create a comparison table of all trained models"""
        comparison_data = []

        for model_name, result in results.items():
            if 'error' not in result:
                metrics = result['metrics']
                row = {
                    'Model': model_name,
                    'Train_MAE': metrics.get('train_mae', np.nan),
                    'Train_RMSE': metrics.get('train_rmse', np.nan),
                    'Train_R2': metrics.get('train_r2', np.nan),
                    'Val_MAE': metrics.get('val_mae', np.nan),
                    'Val_RMSE': metrics.get('val_rmse', np.nan),
                    'Val_R2': metrics.get('val_r2', np.nan)
                }
                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        return model.predict(X)

    def get_best_model(self, results: Dict[str, Dict], metric: str = 'val_r2') -> str:
        """Get the name of the best performing model based on a metric"""
        best_model = None
        best_score = -np.inf if 'r2' in metric else np.inf

        for model_name, result in results.items():
            if 'error' not in result and metric in result['metrics']:
                score = result['metrics'][metric]

                if 'r2' in metric:
                    # Higher R² is better
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                else:
                    # Lower error metrics are better
                    if score < best_score:
                        best_score = score
                        best_model = model_name

        return best_model

if __name__ == "__main__":
    # Example usage
    trainer = AQIModelTrainer()

    print("Available models:")
    for model_name in trainer.models.keys():
        print(f"- {model_name}")

    print("\nModel configurations:")
    for model_name, config in trainer.model_configs.items():
        if config['enabled']:
            print(f"- {model_name}: {config['hyperparameters']}")


