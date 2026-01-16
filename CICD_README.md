# 🚀 Karachi AQI CI/CD Pipeline

A comprehensive CI/CD pipeline for automated data collection and model training in the Karachi AQI prediction system.

## 📋 Pipeline Overview

The CI/CD pipeline automatically runs:

- **Data Collection**: Every hour (collects weather and AQI data)
- **Model Training**: Every day at 2 AM (re-trains ML models)
- **Data Quality Checks**: Continuous validation
- **Model Evaluation**: Performance monitoring

## 🛠️ Available CI/CD Solutions

### 1. 🌪️ Apache Airflow (Recommended)
**Best for**: Production deployments, complex workflows, monitoring

### 2. 🤖 GitHub Actions
**Best for**: Simple automation, GitHub-hosted repositories

### 3. ⏰ Cron Jobs (Simple)
**Best for**: Basic scheduling on single machines

---

## 🚀 Quick Start with Apache Airflow

### Prerequisites
- Docker and Docker Compose
- 4GB+ RAM available
- Internet connection

### Step 1: Install Docker Dependencies
```bash
# Install requirements for Airflow
pip install -r requirements-airflow.txt
```

### Step 2: Start Airflow
```bash
# Start all services (Airflow + PostgreSQL)
docker-compose up -d

# Check status
docker-compose ps
```

### Step 3: Access Airflow UI
- **URL**: http://localhost:8080
- **Username**: admin
- **Password**: admin

### Step 4: Enable DAGs
```bash
# Connect to Airflow container
docker exec -it karachi_aqi_airflow_webserver bash

# Enable the DAGs
airflow dags unpause karachi_aqi_pipeline
airflow dags unpause karachi_aqi_hourly_collection
```

### Step 5: Monitor Pipeline
- Visit http://localhost:8080
- Go to "DAGs" tab
- Click on `karachi_aqi_pipeline` or `karachi_aqi_hourly_collection`
- View "Graph" to see workflow
- Check "Tree" for execution history

---

## 🤖 Alternative: GitHub Actions

### Setup Instructions

#### 1. Add Repository Secrets
Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:
- `OPENWEATHER_API_KEY`: Your OpenWeather API key

#### 2. Enable Workflows
The workflow file is already created at:
```
.github/workflows/karachi-aqi-pipeline.yml
```

#### 3. Manual Trigger (Optional)
You can manually trigger the pipeline:
- Go to Actions tab in your repository
- Select "Karachi AQI CI/CD Pipeline"
- Click "Run workflow"
- Choose options (data collection, model training, etc.)

### GitHub Actions Schedule

```yaml
schedule:
  - cron: '0 * * * *'    # Every hour (data collection)
  - cron: '0 2 * * *'    # Daily 2 AM (model training)
```

---

## ⏰ Alternative: Cron Jobs (Simple)

### Linux/Mac Setup
```bash
# Edit crontab
crontab -e

# Add these lines:
# Data collection every hour
0 * * * * cd /path/to/your/project && python -c "
import sys
sys.path.append('.')
from karachi_aqi_app import KarachiAQISystem
system = KarachiAQISystem()
weather = system.get_current_weather()
if weather:
    system.store_current_data(weather)
    print('✅ Hourly data collected')
"

# Model training daily at 2 AM
0 2 * * * cd /path/to/your/project && python train_karachi_models.py
```

### Windows Task Scheduler Setup

#### 1. Create Data Collection Task
```powershell
# Create a PowerShell script for data collection
$dataCollectionScript = @"
import sys
sys.path.append('.')
from karachi_aqi_app import KarachiAQISystem
system = KarachiAQISystem()
weather = system.get_current_weather()
if weather:
    system.store_current_data(weather)
    print('✅ Data collected')
"@

$dataCollectionScript | Out-File -FilePath "data_collection.py" -Encoding UTF8
```

#### 2. Schedule Tasks
1. Open Task Scheduler
2. Create new task → "Karachi AQI Data Collection"
3. Triggers → New → Daily → Recur every 1 hour
4. Actions → Start a program → `python.exe`
5. Arguments: `data_collection.py`
6. Start in: `C:\path\to\your\project`

Repeat for model training (daily schedule).

---

## 📊 Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Collection │───▶│  Data Validation │───▶│  Feature Store   │
│    (Hourly)      │    │    (Quality)     │    │   (MongoDB)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Model Training  │◀───│ Training Decision│───▶│ Model Evaluation│
│    (Daily)      │    │   (Conditions)   │    │   (Metrics)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Model Deployment│    │  Notifications   │    │   Monitoring     │
│   (Production)  │    │    (Alerts)      │    │   (Dashboard)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔍 Pipeline Components

### Hourly DAG: `karachi_aqi_hourly_collection`
- **Schedule**: Every hour on the hour
- **Tasks**:
  - Collect current weather data from OpenWeather API
  - Store data in MongoDB feature store
  - Basic data validation

### Daily DAG: `karachi_aqi_pipeline`
- **Schedule**: Daily at midnight (can be configured)
- **Tasks**:
  - Data collection (if not already done)
  - Data quality validation
  - Feature engineering
  - Model training decision (skip if recent models exist)
  - Train all 3 ML models (Linear, Random Forest, XGBoost)
  - Model evaluation and performance comparison
  - Model deployment (production ready)

## 📈 Monitoring & Alerts

### Airflow UI Monitoring
- **DAG Status**: Success/Failed/Running states
- **Task Logs**: Detailed execution logs
- **Metrics**: Execution time, success rate
- **Graph View**: Visual workflow representation

### Data Quality Monitoring
```python
# Automatic checks:
- Data completeness (missing values)
- Data freshness (recent timestamps)
- Data accuracy (reasonable ranges)
- Model performance (R² scores, MAE)
```

### Alert Configuration
```python
# Email alerts on:
- Pipeline failures
- Data quality issues
- Model performance degradation
- Missing data warnings
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Airflow Webserver Won't Start
```bash
# Check logs
docker logs karachi_aqi_airflow_webserver

# Restart services
docker-compose restart
```

#### 2. DAGs Not Appearing
```bash
# Check DAG folder permissions
docker exec -it karachi_aqi_airflow_webserver ls -la /opt/airflow/dags/

# Restart scheduler
docker-compose restart airflow-scheduler
```

#### 3. Database Connection Issues
```bash
# Check PostgreSQL
docker logs karachi_aqi_postgres

# Reset database
docker-compose down -v
docker-compose up -d
```

#### 4. Model Training Failures
```bash
# Check available data
python -c "from karachi_feature_store import KarachiFeatureStore; fs = KarachiFeatureStore(); print(f'Records: {fs.get_statistics()[\"total_records\"]}')"

# Check model directory permissions
ls -la src/models/saved_models/
```

## 🔧 Configuration

### Airflow Variables
```python
# Set in Airflow UI Admin → Variables
MODEL_RETRAINING_THRESHOLD = 0.05  # Retrain if R² drops by 5%
DATA_QUALITY_THRESHOLD = 0.95     # Alert if data completeness < 95%
MAX_RETRIES = 3                    # Task retry attempts
```

### Environment Variables
```bash
# In docker-compose.yml or system environment
OPENWEATHER_API_KEY=your_api_key_here
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/
EMAIL_ALERTS=admin@example.com
```

## 📊 Performance Metrics

### Pipeline Metrics
- **Data Collection**: < 5 minutes
- **Model Training**: < 30 minutes
- **Success Rate**: > 95%
- **Data Freshness**: < 1 hour old

### Model Performance
- **Random Forest**: R² = 0.607 (Best)
- **XGBoost**: R² = 0.576
- **Linear Regression**: R² = 0.212 (Baseline)

## 🔄 Scaling & Production

### Production Deployment
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  airflow-webserver:
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
    scale: 3  # Multiple webserver instances

  airflow-worker:
    image: apache/airflow:2.8.1
    command: celery worker
    scale: 5  # Multiple worker instances
```

### Monitoring Stack
- **Prometheus**: Metrics collection
- **Grafana**: Dashboard visualization
- **AlertManager**: Alert management
- **ELK Stack**: Log aggregation

## 📞 Support

### Documentation
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

### Community Resources
- [Apache Airflow Slack](https://apache-airflow.slack.com/)
- [GitHub Actions Community](https://github.community/t/github-actions/41)

---

## 🎯 Summary

Your Karachi AQI system now has a **production-ready CI/CD pipeline** that:

✅ **Automatically collects data every hour**  
✅ **Trains models daily** with performance monitoring  
✅ **Validates data quality** continuously  
✅ **Provides comprehensive monitoring** and alerts  
✅ **Supports multiple deployment options** (Airflow, GitHub Actions, Cron)  
✅ **Scales to production** with proper architecture  

**Choose your preferred CI/CD tool and start the pipeline!** 🚀
