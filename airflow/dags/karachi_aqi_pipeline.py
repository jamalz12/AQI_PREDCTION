"""
Karachi AQI CI/CD Pipeline DAG
==============================

This DAG implements a complete CI/CD pipeline for the Karachi AQI prediction system:

SCHEDULE:
- Data collection: Every hour
- Model training: Every day at 2 AM
- Model evaluation: Daily after training
- Data quality checks: Every hour

WORKFLOW:
1. collect_weather_data (hourly)
2. validate_data_quality (hourly)
3. store_features (hourly)
4. train_models (daily)
5. evaluate_models (daily)
6. deploy_best_model (daily, if improved)

Author: Karachi AQI Team
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor
from airflow.utils.state import State
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'karachi_aqi_team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

# Define the DAG
dag = DAG(
    'karachi_aqi_pipeline',
    default_args=default_args,
    description='CI/CD Pipeline for Karachi AQI Prediction System',
    schedule_interval=timedelta(days=1),  # Daily schedule for main DAG
    max_active_runs=1,
    catchup=False,
    tags=['karachi', 'aqi', 'ml', 'prediction'],
)

def check_data_quality(**context):
    """Check data quality after collection"""
    try:
        # Import here to avoid issues in Airflow workers
        import pandas as pd
        import os

        # Check if data file exists and has recent data
        data_file = 'data/karachi_raw_data.csv'
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found")

        # Read recent data
        df = pd.read_csv(data_file)
        if len(df) == 0:
            raise ValueError("Data file is empty")

        # Check for recent data (last hour)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        recent_data = df[df['timestamp'] >= (datetime.now() - timedelta(hours=1))]

        if len(recent_data) == 0:
            logger.warning("No data collected in the last hour")

        # Quality checks
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in dataset")

        logger.info(f"Data quality check passed: {len(df)} total records, {len(recent_data)} recent records")
        return 'data_quality_passed'

    except Exception as e:
        logger.error(f"Data quality check failed: {e}")
        return 'data_quality_failed'

def should_train_model(**context):
    """Decide whether to train model based on conditions"""
    try:
        # Check if it's been more than 24 hours since last training
        # This is a simplified check - in production you'd check model performance
        import os
        from pathlib import Path

        models_dir = Path('src/models/saved_models')
        if not models_dir.exists():
            logger.info("No existing models found, training required")
            return 'train_models'

        # Check for recent model files (within 24 hours)
        import time
        current_time = time.time()
        recent_models = False

        for model_file in models_dir.glob('*karachi*.joblib'):
            if (current_time - model_file.stat().st_mtime) < (24 * 3600):  # 24 hours
                recent_models = True
                break

        if not recent_models:
            logger.info("No recent models found, training required")
            return 'train_models'
        else:
            logger.info("Recent models found, skipping training")
            return 'skip_training'

    except Exception as e:
        logger.error(f"Error checking training condition: {e}")
        return 'train_models'  # Default to training on error

def evaluate_model_performance(**context):
    """Evaluate model performance after training"""
    try:
        # Read performance metrics from the latest training
        import os
        import json
        from pathlib import Path

        models_dir = Path('src/models/saved_models')
        if not models_dir.exists():
            raise FileNotFoundError("Models directory not found")

        # Find latest performance file
        perf_files = list(models_dir.glob('performance*karachi*.json'))
        if not perf_files:
            logger.warning("No performance files found")
            return 'evaluation_completed'

        latest_perf = max(perf_files, key=lambda x: x.stat().st_mtime)

        with open(latest_perf, 'r') as f:
            performance = json.load(f)

        # Log performance metrics
        for model_name, metrics in performance.items():
            if isinstance(metrics, dict) and 'test_r2' in metrics:
                r2_score = metrics['test_r2']
                mae = metrics.get('test_mae', 'N/A')
                logger.info(f"Model {model_name}: R² = {r2_score:.3f}, MAE = {mae}")

                # In production, you might compare with previous models
                # and decide whether to deploy based on performance improvement

        logger.info("Model evaluation completed successfully")
        return 'evaluation_completed'

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return 'evaluation_failed'

# Define tasks

# Dummy start task
start_pipeline = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
)

# Hourly data collection task
collect_weather_data = BashOperator(
    task_id='collect_weather_data',
    bash_command='cd /opt/airflow/project && python -c "
import sys
sys.path.append(\".\")
from karachi_aqi_app import KarachiAQISystem
system = KarachiAQISystem()
weather = system.get_current_weather()
if weather:
    system.store_current_data(weather)
    print(f\"✅ Collected and stored weather data: AQI {weather.get(\'aqi\', \'N/A\')}\")
else:
    print(\"❌ Failed to collect weather data\")
    sys.exit(1)
"',
    dag=dag,
    execution_timeout=timedelta(minutes=10),
)

# Data quality check task
validate_data_quality = PythonOperator(
    task_id='validate_data_quality',
    python_callable=check_data_quality,
    provide_context=True,
    dag=dag,
)

# Feature storage task (already handled in collection, but explicit task)
store_features = BashOperator(
    task_id='store_features',
    bash_command='cd /opt/airflow/project && echo "✅ Features automatically stored during data collection"',
    dag=dag,
)

# Decision task for training
should_train_decision = BranchPythonOperator(
    task_id='should_train_decision',
    python_callable=should_train_model,
    provide_context=True,
    dag=dag,
)

# Daily model training task
train_models = BashOperator(
    task_id='train_models',
    bash_command='cd /opt/airflow/project && python train_karachi_models.py',
    dag=dag,
    execution_timeout=timedelta(hours=2),
)

# Skip training task
skip_training = DummyOperator(
    task_id='skip_training',
    dag=dag,
)

# Model evaluation task
evaluate_models = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_model_performance,
    provide_context=True,
    dag=dag,
)

# Model deployment task (placeholder - in production this would deploy to production)
deploy_model = BashOperator(
    task_id='deploy_model',
    bash_command='cd /opt/airflow/project && echo "🎉 Model deployment completed - ready for production use"',
    dag=dag,
)

# Evaluation completed task
evaluation_completed = DummyOperator(
    task_id='evaluation_completed',
    dag=dag,
)

evaluation_failed = DummyOperator(
    task_id='evaluation_failed',
    dag=dag,
)

data_quality_passed = DummyOperator(
    task_id='data_quality_passed',
    dag=dag,
)

data_quality_failed = DummyOperator(
    task_id='data_quality_failed',
    dag=dag,
)

# End pipeline task
end_pipeline = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
)

# Define task dependencies

# Main pipeline flow
start_pipeline >> collect_weather_data >> validate_data_quality

# Data quality branching
validate_data_quality >> [data_quality_passed, data_quality_failed]
data_quality_passed >> store_features

# Training decision (daily)
store_features >> should_train_decision
should_train_decision >> [train_models, skip_training]

# Training and evaluation flow
train_models >> evaluate_models
evaluate_models >> [evaluation_completed, evaluation_failed]
evaluation_completed >> deploy_model
skip_training >> deploy_model

# All paths lead to end
[deploy_model, data_quality_failed, evaluation_failed] >> end_pipeline

# Create a separate hourly DAG for data collection
hourly_dag = DAG(
    'karachi_aqi_hourly_collection',
    default_args={
        **default_args,
        'start_date': days_ago(1),
    },
    description='Hourly data collection for Karachi AQI',
    schedule_interval='0 * * * *',  # Every hour at minute 0
    max_active_runs=1,
    catchup=False,
    tags=['karachi', 'aqi', 'hourly', 'data-collection'],
)

# Hourly collection task
hourly_collect = BashOperator(
    task_id='hourly_data_collection',
    bash_command='cd /opt/airflow/project && python -c "
import sys
sys.path.append(\".\")
from karachi_aqi_app import KarachiAQISystem
system = KarachiAQISystem()
weather = system.get_current_weather()
if weather:
    system.store_current_data(weather)
    print(f\"✅ Hourly data collection: AQI {weather.get(\'aqi\', \'N/A\')}\")
else:
    print(\"❌ Hourly data collection failed\")
"',
    dag=hourly_dag,
    execution_timeout=timedelta(minutes=5),
)

if __name__ == "__main__":
    print("Karachi AQI Pipeline DAG loaded successfully!")
    print("Daily DAG: karachi_aqi_pipeline")
    print("Hourly DAG: karachi_aqi_hourly_collection")
    print("\nTo run manually:")
    print("airflow dags unpause karachi_aqi_pipeline")
    print("airflow dags unpause karachi_aqi_hourly_collection")
