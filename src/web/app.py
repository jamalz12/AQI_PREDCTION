import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import logging
from typing import Dict, List, Optional, Tuple
import yaml
import os

# Import our custom modules
from src.data.data_collector import AQIWeatherDataCollector
from src.features.feature_store import AQIFeatureStore
from src.features.feature_engineering import AQIFeatureEngineering
from src.models.model_trainer import AQIModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .prediction-card {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .aqi-good { color: #00e400; }
    .aqi-moderate { color: #ffff00; }
    .aqi-unhealthy-sensitive { color: #ff7e00; }
    .aqi-unhealthy { color: #ff0000; }
    .aqi-very-unhealthy { color: #8f3f97; }
    .aqi-hazardous { color: #7e0023; }
</style>
""", unsafe_allow_html=True)

class AQIPredictionApp:
    """
    Streamlit application for AQI prediction system
    """

    def __init__(self):
        self.config = self._load_config()
        self.data_collector = None
        self.feature_store = None
        self.feature_engineer = None
        self.model_trainer = None

        # AQI categories and colors
        self.aqi_categories = {
            'Good': {'range': (0, 50), 'color': '#00e400', 'class': 'aqi-good'},
            'Moderate': {'range': (51, 100), 'color': '#ffff00', 'class': 'aqi-moderate'},
            'Unhealthy for Sensitive Groups': {'range': (101, 150), 'color': '#ff7e00', 'class': 'aqi-unhealthy-sensitive'},
            'Unhealthy': {'range': (151, 200), 'color': '#ff0000', 'class': 'aqi-unhealthy'},
            'Very Unhealthy': {'range': (201, 300), 'color': '#8f3f97', 'class': 'aqi-very-unhealthy'},
            'Hazardous': {'range': (301, 500), 'color': '#7e0023', 'class': 'aqi-hazardous'}
        }

    def _load_config(self) -> Dict:
        """Load configuration"""
        try:
            with open('config/config.yaml', 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            st.error("Configuration file not found. Please ensure config/config.yaml exists.")
            return {}

    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.data_collector = AQIWeatherDataCollector()
            self.feature_store = AQIFeatureStore()
            self.feature_engineer = AQIFeatureEngineering()

            # Load latest trained models
            self.model_trainer = AQIModelTrainer()
            self._load_trained_models()

            return True
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            return False

    def _load_trained_models(self):
        """Load the latest trained models"""
        try:
            # Find latest model files
            model_dir = "src/models/saved_models"
            if os.path.exists(model_dir):
                model_files = os.listdir(model_dir)
                latest_versions = {}

                for model_name in ['linear_regression', 'random_forest', 'xgboost']:
                    model_files_filtered = [f for f in model_files if f.startswith(f"{model_name}_")]
                    if model_files_filtered:
                        # Get latest version
                        latest_file = sorted(model_files_filtered)[-1]
                        version = latest_file.split('_')[-1].replace('.joblib', '')
                        self.model_trainer.models[model_name] = self.model_trainer.load_model(model_name, version)
                        logger.info(f"Loaded {model_name} model version {version}")

        except Exception as e:
            logger.warning(f"Could not load trained models: {e}")

    def get_aqi_category(self, aqi_value: float) -> Tuple[str, str]:
        """Get AQI category and color for a given AQI value"""
        for category, info in self.aqi_categories.items():
            if info['range'][0] <= aqi_value <= info['range'][1]:
                return category, info['color']
        return "Hazardous", self.aqi_categories["Hazardous"]['color']

    def create_aqi_gauge(self, aqi_value: float, title: str = "Current AQI") -> go.Figure:
        """Create an AQI gauge chart"""
        category, color = self.get_aqi_category(aqi_value)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi_value,
            title={'text': f"{title}<br><span style='font-size:0.8em;color:{color}'>{category}</span>"},
            gauge={
                'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': '#00e400'},
                    {'range': [51, 100], 'color': '#ffff00'},
                    {'range': [101, 150], 'color': '#ff7e00'},
                    {'range': [151, 200], 'color': '#ff0000'},
                    {'range': [201, 300], 'color': '#8f3f97'},
                    {'range': [301, 500], 'color': '#7e0023'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': aqi_value
                }
            }
        ))

        fig.update_layout(height=300)
        return fig

    def display_current_weather(self, city: str):
        """Display current weather and AQI for a city"""
        try:
            data = self.data_collector.collect_weather_data(city)

            if data:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Temperature", f"{data['temperature']}°C")
                    st.metric("Humidity", f"{data['humidity']}%")

                with col2:
                    st.metric("Wind Speed", f"{data['wind_speed']} m/s")
                    st.metric("Pressure", f"{data['pressure']} hPa")

                with col3:
                    aqi_category, aqi_color = self.get_aqi_category(data['aqi'])
                    st.metric("AQI", f"{data['aqi']}", delta=aqi_category)
                    st.metric("PM2.5", f"{data['pm2_5']} μg/m³")

                # AQI Gauge
                st.plotly_chart(self.create_aqi_gauge(data['aqi'], f"AQI - {city}"), use_container_width=True)

            else:
                st.error(f"Could not retrieve data for {city}")

        except Exception as e:
            st.error(f"Error displaying current weather: {e}")

    def predict_future_aqi(self, city: str, hours_ahead: int = 24):
        """Predict future AQI for a city"""
        try:
            # Get recent data for the city
            recent_data = self.feature_store.get_city_data(city, days=7)

            if recent_data.empty:
                st.warning(f"No historical data available for {city}")
                return

            # Prepare features for prediction
            processed_data = self.feature_engineer.prepare_features_for_modeling(recent_data)

            if processed_data.empty:
                st.warning("Could not process data for prediction")
                return

            # Get latest features (most recent data point)
            latest_features = processed_data.tail(1).dropna(axis=1)

            if latest_features.empty:
                st.warning("No valid features available for prediction")
                return

            # Remove target columns and timestamp
            feature_cols = [col for col in latest_features.columns
                          if not col.startswith(('aqi_', 'pm2_5_future_', 'pm10_future_', 'calculated_aqi_future_'))
                          and col not in ['timestamp', 'city', 'weather_main', 'weather_description', 'aqi_category']]

            X_pred = latest_features[feature_cols]

            # Make predictions with all models
            predictions = {}
            for model_name in self.model_trainer.models.keys():
                try:
                    pred = self.model_trainer.predict(model_name, X_pred)[0]
                    predictions[model_name] = max(0, pred)  # Ensure non-negative
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
                    predictions[model_name] = None

            # Display predictions
            st.subheader(f"AQI Prediction for {city} ({hours_ahead} hours ahead)")

            # Calculate ensemble prediction (average of all models)
            valid_predictions = [p for p in predictions.values() if p is not None]
            if valid_predictions:
                ensemble_pred = np.mean(valid_predictions)
                category, color = self.get_aqi_category(ensemble_pred)

                st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="color: {color};">Predicted AQI: {ensemble_pred:.1f}</h3>
                    <p><strong>Category:</strong> {category}</p>
                    <p><em>Prediction based on ensemble of {len(valid_predictions)} models</em></p>
                </div>
                """, unsafe_allow_html=True)

                # Individual model predictions
                st.subheader("Individual Model Predictions")
                cols = st.columns(len(predictions))

                for i, (model_name, pred) in enumerate(predictions.items()):
                    with cols[i]:
                        if pred is not None:
                            model_category, model_color = self.get_aqi_category(pred)
                            st.metric(f"{model_name.replace('_', ' ').title()}", f"{pred:.1f}", model_category)
                        else:
                            st.metric(f"{model_name.replace('_', ' ').title()}", "N/A")

            else:
                st.error("Could not generate predictions with any model")

        except Exception as e:
            st.error(f"Error making predictions: {e}")
            logger.error(f"Prediction error: {e}")

    def show_historical_trends(self, city: str, days: int = 7):
        """Show historical AQI trends for a city"""
        try:
            historical_data = self.feature_store.get_city_data(city, days=days)

            if historical_data.empty:
                st.warning(f"No historical data available for {city}")
                return

            # Create time series plot
            fig = px.line(historical_data, x='timestamp', y=['pm2_5', 'aqi'],
                         title=f'AQI and PM2.5 Trends - {city} (Last {days} days)',
                         labels={'value': 'Concentration', 'variable': 'Pollutant'})

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                avg_aqi = historical_data['aqi'].mean()
                st.metric("Average AQI", f"{avg_aqi:.1f}")

            with col2:
                max_aqi = historical_data['aqi'].max()
                st.metric("Peak AQI", f"{max_aqi:.1f}")

            with col3:
                good_days = (historical_data['aqi'] <= 50).sum()
                st.metric("Good Air Days", f"{good_days}/{len(historical_data)}")

        except Exception as e:
            st.error(f"Error displaying historical trends: {e}")

    def show_feature_importance(self):
        """Display feature importance from trained models"""
        try:
            st.subheader("Feature Importance Analysis")

            # For now, show a sample (this would be populated from actual model results)
            sample_features = {
                'pm2_5_lag_24h': 0.25,
                'temperature': 0.15,
                'humidity': 0.12,
                'wind_speed': 0.10,
                'pm2_5_rolling_mean_24h': 0.08,
                'pressure': 0.07,
                'hour_sin': 0.05,
                'clouds': 0.04,
                'co': 0.03,
                'no2': 0.02
            }

            # Create horizontal bar chart
            fig = px.bar(
                x=list(sample_features.values()),
                y=list(sample_features.keys()),
                orientation='h',
                title="Top 10 Most Important Features (Random Forest)",
                labels={'x': 'Importance', 'y': 'Feature'}
            )

            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error displaying feature importance: {e}")

    def run(self):
        """Run the Streamlit application"""
        st.markdown('<h1 class="main-header">🌤️ AQI Prediction System</h1>', unsafe_allow_html=True)

        # Initialize components
        if not self.initialize_components():
            st.stop()

        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", [
            "Dashboard",
            "Current Weather",
            "AQI Predictions",
            "Historical Trends",
            "Model Analysis"
        ])

        # City selection (Karachi, Pakistan)
        cities = self.config.get('data_collection', {}).get('cities', ['Karachi'])
        selected_city = st.sidebar.selectbox("Select City", cities)

        # Main content
        if page == "Dashboard":
            st.header("📊 Dashboard")

            # Quick overview
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Monitored Cities", len(cities))

            with col2:
                st.metric("Active Models", len(self.model_trainer.models) if self.model_trainer else 0)

            with col3:
                st.metric("Last Updated", datetime.now().strftime("%H:%M"))

            # Current AQI for all cities
            st.subheader("Current AQI Status")

            city_cols = st.columns(min(len(cities), 3))
            for i, city in enumerate(cities[:3]):
                with city_cols[i]:
                    try:
                        data = self.data_collector.collect_weather_data(city)
                        if data:
                            category, color = self.get_aqi_category(data['aqi'])
                            st.metric(city, f"AQI: {data['aqi']}", category)
                        else:
                            st.metric(city, "No Data")
                    except:
                        st.metric(city, "Error")

        elif page == "Current Weather":
            st.header("🌤️ Current Weather & Air Quality")
            self.display_current_weather(selected_city)

        elif page == "AQI Predictions":
            st.header("🔮 AQI Predictions")

            hours_ahead = st.slider("Prediction Horizon (hours)", 1, 72, 24)
            if st.button("Generate Prediction"):
                self.predict_future_aqi(selected_city, hours_ahead)

        elif page == "Historical Trends":
            st.header("📈 Historical Trends")

            days = st.slider("Time Period (days)", 1, 30, 7)
            self.show_historical_trends(selected_city, days)

        elif page == "Model Analysis":
            st.header("🤖 Model Analysis")

            tab1, tab2 = st.tabs(["Model Performance", "Feature Importance"])

            with tab1:
                st.subheader("Model Performance Metrics")

                # Sample performance data (would be loaded from training results)
                performance_data = pd.DataFrame({
                    'Model': ['Linear Regression', 'Random Forest', 'XGBoost'],
                    'MAE': [12.5, 8.3, 7.1],
                    'RMSE': [15.2, 11.8, 10.2],
                    'R²': [0.72, 0.85, 0.88]
                })

                st.dataframe(performance_data)

                # Performance comparison chart
                fig = px.bar(performance_data, x='Model', y='R²',
                           title="Model R² Scores Comparison")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                self.show_feature_importance()

        # Footer
        st.markdown("---")
        st.markdown("*AQI Prediction System - Powered by Machine Learning*")

def main():
    """Main function to run the Streamlit app"""
    app = AQIPredictionApp()
    app.run()

if __name__ == "__main__":
    main()
