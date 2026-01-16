import pandas as pd
import numpy as np
import logging
import yaml
from datetime import datetime
import os
from typing import Dict, Optional

# Import our custom modules
from src.features.feature_store import AQIFeatureStore
from src.features.feature_engineering import AQIFeatureEngineering
from src.models.model_trainer import AQIModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AQITrainingPipeline:
    """
    Complete training pipeline for AQI prediction models
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)

        # Initialize components
        self.feature_store = AQIFeatureStore(config_path)
        self.feature_engineer = AQIFeatureEngineering(config_path)
        self.model_trainer = AQIModelTrainer(config_path)

        # Training configuration
        self.training_days = 30  # Days of historical data to use
        self.target_variable = 'pm2_5_future_24h'  # What we're predicting

        # Results storage
        self.training_results = {}

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def collect_training_data(self) -> pd.DataFrame:
        """Collect training data from feature store"""
        logger.info(f"Collecting {self.training_days} days of training data...")

        df = self.feature_store.get_training_data(days=self.training_days)

        if df.empty:
            raise ValueError("No training data available in feature store")

        logger.info(f"Collected {len(df)} records for training")
        return df

    def prepare_features(self, raw_data: pd.DataFrame) -> tuple:
        """Prepare features for modeling"""
        logger.info("Preparing features for modeling...")

        # Feature engineering
        processed_df = self.feature_engineer.prepare_features_for_modeling(
            raw_data, target_variable=self.target_variable
        )

        if processed_df.empty:
            raise ValueError("Feature engineering resulted in empty dataset")

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.feature_engineer.split_data(
            processed_df, target_variable=self.target_variable
        )

        logger.info(f"Data split complete:")
        logger.info(f"  Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train all configured models"""
        logger.info("Training models...")

        results = self.model_trainer.train_all_models(X_train, y_train, X_val, y_val)

        # Log results to MLflow
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for model_name, result in results.items():
            if 'error' not in result:
                run_name = f"{model_name}_{timestamp}"
                self.model_trainer.log_to_mlflow(model_name, result, X_train, run_name)

        self.training_results = results
        return results

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate all trained models on test data"""
        logger.info("Evaluating models on test data...")

        test_results = {}

        for model_name in self.model_trainer.models.keys():
            try:
                test_metrics = self.model_trainer.evaluate_on_test(model_name, X_test, y_test)
                test_results[model_name] = test_metrics
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                test_results[model_name] = {'error': str(e)}

        return test_results

    def save_best_models(self, results: Dict) -> Dict[str, str]:
        """Save the best performing models"""
        logger.info("Saving best models...")

        saved_models = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, result in results.items():
            if 'error' not in result:
                try:
                    model_path = self.model_trainer.save_model(model_name, result, timestamp)
                    saved_models[model_name] = model_path
                except Exception as e:
                    logger.error(f"Error saving {model_name}: {e}")

        return saved_models

    def create_model_comparison_report(self, training_results: Dict, test_results: Dict) -> pd.DataFrame:
        """Create a comprehensive model comparison report"""
        logger.info("Creating model comparison report...")

        comparison_data = []

        for model_name in training_results.keys():
            if 'error' not in training_results[model_name]:
                train_metrics = training_results[model_name]['metrics']
                test_metrics = test_results.get(model_name, {})

                row = {
                    'Model': model_name,
                    'Train_MAE': train_metrics.get('train_mae', np.nan),
                    'Train_RMSE': train_metrics.get('train_rmse', np.nan),
                    'Train_R2': train_metrics.get('train_r2', np.nan),
                    'Val_MAE': train_metrics.get('val_mae', np.nan),
                    'Val_RMSE': train_metrics.get('val_rmse', np.nan),
                    'Val_R2': train_metrics.get('val_r2', np.nan),
                    'Test_MAE': test_metrics.get('test_mae', np.nan),
                    'Test_RMSE': test_metrics.get('test_rmse', np.nan),
                    'Test_R2': test_metrics.get('test_r2', np.nan)
                }
                comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Save comparison report
        os.makedirs('reports', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"reports/model_comparison_{timestamp}.csv"
        df.to_csv(report_path, index=False)

        logger.info(f"Model comparison report saved to {report_path}")
        return df

    def run_training_pipeline(self) -> Dict:
        """Run the complete training pipeline"""
        try:
            logger.info("Starting AQI training pipeline...")

            # Step 1: Collect training data
            raw_data = self.collect_training_data()

            # Step 2: Prepare features
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_features(raw_data)

            # Step 3: Train models
            training_results = self.train_models(X_train, y_train, X_val, y_val)

            # Step 4: Evaluate on test data
            test_results = self.evaluate_models(X_test, y_test)

            # Step 5: Save best models
            saved_models = self.save_best_models(training_results)

            # Step 6: Create comparison report
            comparison_df = self.create_model_comparison_report(training_results, test_results)

            # Step 7: Identify best model
            best_model = self.model_trainer.get_best_model(training_results, 'val_r2')

            pipeline_results = {
                'training_results': training_results,
                'test_results': test_results,
                'saved_models': saved_models,
                'comparison_report': comparison_df,
                'best_model': best_model,
                'data_shapes': {
                    'train': X_train.shape,
                    'validation': X_val.shape,
                    'test': X_test.shape
                }
            }

            logger.info("Training pipeline completed successfully!")
            logger.info(f"Best performing model: {best_model}")

            return pipeline_results

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

    def predict_with_best_model(self, features: pd.DataFrame) -> np.ndarray:
        """Make predictions using the best trained model"""
        if not self.training_results:
            raise ValueError("No trained models available. Run training pipeline first.")

        best_model_name = self.model_trainer.get_best_model(self.training_results, 'val_r2')

        logger.info(f"Making predictions with {best_model_name}")
        predictions = self.model_trainer.predict(best_model_name, features)

        return predictions

def main():
    """Main function to run the training pipeline"""
    try:
        # Initialize pipeline
        pipeline = AQITrainingPipeline()

        # Run training
        results = pipeline.run_training_pipeline()

        # Print results summary
        print("\n" + "="*60)
        print("AQI TRAINING PIPELINE RESULTS")
        print("="*60)

        print(f"\nBest Model: {results['best_model']}")
        print(f"Data Shapes: {results['data_shapes']}")

        print("\nModel Performance Summary:")
        comparison_df = results['comparison_report']
        print(comparison_df.round(4).to_string(index=False))

        print("\nSaved Models:")
        for model_name, path in results['saved_models'].items():
            print(f"- {model_name}: {path}")

        print("\nTraining pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())


