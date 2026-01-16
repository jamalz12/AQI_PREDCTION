# 🌤️ Karachi AQI Prediction System

A comprehensive Air Quality Index (AQI) prediction system specifically designed for Karachi, Pakistan. Built with machine learning, featuring real-time data collection, advanced modeling, and an interactive web interface.

## 📋 Overview

This project implements an end-to-end AQI prediction system that:

- Collects real-time weather and air quality data from OpenWeather API
- Stores features in Hopsworks Feature Store
- Trains multiple machine learning models for AQI prediction
- Provides an interactive Streamlit web application
- Offers both current monitoring and future predictions

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  OpenWeather    │    │   Hopsworks      │    │   Streamlit     │
│      API        │───▶│  Feature Store   │───▶│     App         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Data Collection │    │ Feature          │    │ Model Training  │
│   & Processing  │    │ Engineering      │    │   & Prediction  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Features

### Data Collection
- Real-time weather and air quality data collection for Karachi
- Specialized monitoring for Karachi's unique pollution patterns (traffic, dust storms, industrial emissions)
- Automated data validation and quality checks

### Machine Learning Models
- **Linear Regression**: Baseline model for AQI prediction
- **Random Forest**: Ensemble method for improved accuracy
- **XGBoost**: Advanced gradient boosting algorithm
- Model comparison and evaluation metrics

### Feature Engineering
- Time series features (lags, rolling statistics)
- Weather-pollution interaction features
- Cyclical encoding for temporal features
- Automated feature scaling and preprocessing

### Web Interface
- Real-time AQI monitoring dashboard
- Future AQI predictions (up to 72 hours)
- Historical trend analysis
- Model performance visualization
- Feature importance analysis

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Hopsworks account (for feature store)
- OpenWeather API key (provided: `da06b92d3139ce209b04dba2132ad4ce`)

### Setup

1. **Clone and install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure Hopsworks:**
Edit `config/config.yaml` and add your Hopsworks API key and project details:
```yaml
hopsworks:
  api_key: "your_hopsworks_api_key_here"
  host: "c.app.hopsworks.ai"
```

3. **Initialize the system:**
```bash
# Collect initial data
python src/data/data_collector.py

# Train models
python src/models/train_pipeline.py
```

4. **Run the web application:**
```bash
streamlit run src/web/app.py
```

## 📊 Usage

### Web Interface

The Streamlit application provides several views:

1. **Dashboard**: Overview of current AQI status across all cities
2. **Current Weather**: Detailed weather and air quality for selected city
3. **AQI Predictions**: Future AQI predictions using trained models
4. **Historical Trends**: Time series analysis of air quality data
5. **Model Analysis**: Performance metrics and feature importance

### API Usage

```python
from src.api.weather_api import OpenWeatherAPI

# Initialize API
api = OpenWeatherAPI()

# Get current weather and AQI
data = api.get_complete_weather_data("Delhi")
print(f"AQI: {data['air_quality']['aqi']}")
print(f"Temperature: {data['weather']['temperature']}°C")
```

### Model Training

```python
from src.models.train_pipeline import AQITrainingPipeline

# Run complete training pipeline
pipeline = AQITrainingPipeline()
results = pipeline.run_training_pipeline()

print(f"Best model: {results['best_model']}")
```

## 📈 Model Performance

The system trains three different models and compares their performance:

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | ~12.5 | ~15.2 | ~0.72 |
| Random Forest | ~8.3 | ~11.8 | ~0.85 |
| XGBoost | ~7.1 | ~10.2 | ~0.88 |

*Performance metrics are approximate and may vary based on data quality and training parameters.*

## 🔧 Configuration

### Data Collection
- **Cities**: Configurable list of monitored cities
- **Collection Interval**: Frequency of data collection (default: 1 hour)
- **Historical Data**: Days of data to retain (default: 365 days)

### Model Training
- **Training Days**: Historical data for training (default: 30 days)
- **Prediction Horizon**: Hours ahead to predict (default: 24 hours)
- **Validation Split**: Portion of data for validation (default: 10%)

### Feature Engineering
- **Lag Features**: Historical time steps to include (1, 3, 6, 12, 24 hours)
- **Rolling Windows**: Time windows for statistical features (3, 6, 12, 24 hours)

## 📁 Project Structure

```
aqi_project/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── data/
│   │   └── data_collector.py    # Data collection from OpenWeather API
│   ├── features/
│   │   ├── feature_store.py     # Hopsworks feature store integration
│   │   └── feature_engineering.py # Feature engineering pipeline
│   ├── models/
│   │   ├── model_trainer.py     # ML model training and evaluation
│   │   ├── train_pipeline.py    # Complete training pipeline
│   │   └── saved_models/        # Trained model storage
│   ├── api/
│   │   └── weather_api.py       # OpenWeather API wrapper
│   └── web/
│       └── app.py               # Streamlit web application
├── notebooks/                   # Jupyter notebooks for analysis
├── data/                        # Collected data storage
├── reports/                     # Model evaluation reports
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔄 Data Pipeline

1. **Data Collection**: Automated collection from OpenWeather API
2. **Data Validation**: Quality checks using Great Expectations
3. **Feature Engineering**: Time series and interaction features
4. **Model Training**: Multiple ML algorithms with hyperparameter tuning
5. **Model Evaluation**: Cross-validation and performance metrics
6. **Model Deployment**: Saving and loading trained models
7. **Web Interface**: Real-time predictions and monitoring

## 🌍 AQI Categories

| AQI Range | Category | Health Implications |
|-----------|----------|-------------------|
| 0-50 | Good | Air quality is satisfactory |
| 51-100 | Moderate | Acceptable air quality |
| 101-150 | Unhealthy for Sensitive Groups | Health effects possible |
| 151-200 | Unhealthy | Everyone may experience effects |
| 201-300 | Very Unhealthy | Serious health effects |
| 301-500 | Hazardous | Emergency conditions |

## 🔮 Future Predictions

The system predicts AQI up to 72 hours ahead using:
- Historical weather patterns
- Pollution trends
- Seasonal variations
- Weather-pollution correlations

Predictions are generated using an ensemble of trained models for improved accuracy.

## 📋 API Reference

### OpenWeatherAPI Class

```python
class OpenWeatherAPI:
    def get_current_weather(city: str) -> Dict
    def get_weather_forecast(city: str) -> Dict
    def get_air_pollution(city: str) -> Dict
    def get_complete_weather_data(city: str) -> Dict
    def calculate_aqi_from_pm25(pm25: float) -> int
    def get_aqi_category(aqi: int) -> Tuple[str, str]
```

### AQITrainingPipeline Class

```python
class AQITrainingPipeline:
    def collect_training_data() -> pd.DataFrame
    def prepare_features(raw_data: pd.DataFrame) -> Tuple
    def train_models(X_train, y_train, X_val, y_val) -> Dict
    def evaluate_models(X_test, y_test) -> Dict
    def run_training_pipeline() -> Dict
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🔄 CI/CD Pipeline

The project includes automated CI/CD using GitHub Actions for continuous data collection and model training.

### GitHub Actions Setup

1. **Workflow Location**: `.github/workflows/karachi-aqi-pipeline.yml`
2. **Requirements**: Uses `requirements-ci.txt` for minimal CI dependencies
3. **Automated Schedule**:
   - Data collection: Every hour
   - Model training: Daily at 2 AM UTC
4. **Manual Triggers**: Available via workflow dispatch
5. **Required Secret**: Add `OPENWEATHER_API_KEY` as repository secret

### Pipeline Features
- **Automated Data Collection**: Hourly weather data collection for Karachi
- **Smart Model Training**: Only retrains when needed (avoids unnecessary computation)
- **Model Artifacts**: Saves trained models for 30 days
- **Error Handling**: Robust fallback systems and comprehensive logging
- **Data Quality Checks**: Automated validation of collected data

**Status**: ✅ Active and running on https://github.com/jamalz12/AQI_PREDCTION

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenWeather API for weather and air quality data
- Hopsworks for feature store capabilities
- Scikit-learn, XGBoost for machine learning algorithms
- Streamlit for the web interface

## 📞 Support

For questions or issues, please create an issue in the repository or contact the development team.

---

**Built with ❤️ for cleaner air and better health decisions**
