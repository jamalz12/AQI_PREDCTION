"""
Advanced Exploratory Data Analysis for AQI Prediction System
===========================================================

This script performs comprehensive advanced EDA including:
- Advanced statistical analysis
- Time series decomposition
- Feature correlation networks
- Anomaly detection
- Clustering analysis
- Causal inference analysis
- Predictive modeling insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import warnings
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append('../src')

# Import custom modules
from src.data.data_collector import AQIWeatherDataCollector
from src.features.feature_store import AQIFeatureStore
from src.features.feature_engineering import AQIFeatureEngineering

# Import advanced libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.stats import normaltest, shapiro, anderson
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import networkx as nx
from networkx.algorithms import community
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class AdvancedAQIEDA:
    """
    Advanced Exploratory Data Analysis for AQI Prediction System
    """

    def __init__(self, config_path="../config/config.yaml"):
        self.config_path = config_path
        self.data_collector = AQIWeatherDataCollector(config_path)
        self.feature_store = AQIFeatureStore(config_path)
        self.feature_engineer = AQIFeatureEngineering(config_path)

        # Initialize data storage
        self.raw_data = None
        self.processed_data = None
        self.city_data = {}

        # Color schemes for consistency
        self.aqi_colors = {
            'Good': '#00e400',
            'Moderate': '#ffff00',
            'Unhealthy for Sensitive Groups': '#ff7e00',
            'Unhealthy': '#ff0000',
            'Very Unhealthy': '#8f3f97',
            'Hazardous': '#7e0023'
        }

        self.city_colors = {
            'Delhi': '#1f77b4',
            'Mumbai': '#ff7f0e',
            'Bangalore': '#2ca02c',
            'Chennai': '#d62728',
            'Kolkata': '#9467bd',
            'Hyderabad': '#8c564b'
        }

    def collect_sample_data(self, days=7, save_to_csv=True):
        """Collect sample data for analysis"""
        print("🔄 Collecting sample data...")

        all_data = []
        cities = self.data_collector.cities

        for city in cities:
            print(f"  📍 Collecting data for {city}...")
            try:
                # Collect multiple days of data (simulate historical data)
                city_data = []
                base_time = datetime.now() - timedelta(days=days)

                for i in range(days * 24):  # Hourly data
                    # Add some realistic variations
                    timestamp = base_time + timedelta(hours=i)
                    data_point = self.data_collector.collect_weather_data(city)

                    if data_point:
                        data_point['timestamp'] = timestamp
                        # Add some realistic noise to create time series patterns
                        noise_factor = 0.1
                        data_point['pm2_5'] *= (1 + np.random.normal(0, noise_factor))
                        data_point['aqi'] = self.data_collector.calculate_aqi_from_pm25(data_point['pm2_5'])
                        city_data.append(data_point)

                if city_data:
                    city_df = pd.DataFrame(city_data)
                    all_data.append(city_df)
                    self.city_data[city] = city_df
                    print(f"    ✅ Collected {len(city_df)} records for {city}")

            except Exception as e:
                print(f"    ❌ Error collecting data for {city}: {e}")

        if all_data:
            self.raw_data = pd.concat(all_data, ignore_index=True)
            self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])

            if save_to_csv:
                os.makedirs('../data', exist_ok=True)
                filename = f"../data/advanced_eda_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.raw_data.to_csv(filename, index=False)
                print(f"💾 Data saved to {filename}")

            print(f"🎯 Total records collected: {len(self.raw_data)}")
            return self.raw_data
        else:
            print("❌ No data collected")
            return None

    def perform_advanced_statistical_analysis(self):
        """Perform advanced statistical analysis"""
        if self.raw_data is None:
            print("❌ No data available. Run collect_sample_data() first.")
            return

        print("\n📊 ADVANCED STATISTICAL ANALYSIS")
        print("=" * 50)

        # Basic statistics
        print("\n1. BASIC STATISTICS BY CITY:")
        numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'pm2_5', 'pm10']

        city_stats = self.raw_data.groupby('city')[numeric_cols].agg(['mean', 'std', 'skew', 'kurtosis'])
        city_stats.columns = ['_'.join(col).strip() for col in city_stats.columns.values]
        print(city_stats.round(3))

        # Normality tests
        print("\n2. NORMALITY TESTS (Shapiro-Wilk):")
        normality_results = {}
        for city in self.raw_data['city'].unique():
            city_data = self.raw_data[self.raw_data['city'] == city]
            normality_results[city] = {}
            for col in ['aqi', 'pm2_5', 'temperature']:
                try:
                    stat, p_value = shapiro(city_data[col].dropna())
                    normality_results[city][col] = {'statistic': stat, 'p_value': p_value, 'normal': p_value > 0.05}
                except:
                    normality_results[city][col] = {'error': 'Insufficient data'}

        for city, tests in normality_results.items():
            print(f"\n{city}:")
            for var, result in tests.items():
                if 'error' not in result:
                    normal_status = "✅ Normal" if result['normal'] else "❌ Not Normal"
                    print(f"  {var}: p={result['p_value']:.4f} ({normal_status})")

        # Outlier analysis using IQR method
        print("\n3. OUTLIER ANALYSIS (IQR Method):")
        outlier_stats = {}
        for city in self.raw_data['city'].unique():
            city_data = self.raw_data[self.raw_data['city'] == city]
            outlier_stats[city] = {}

            for col in ['aqi', 'pm2_5']:
                if col in city_data.columns:
                    Q1 = city_data[col].quantile(0.25)
                    Q3 = city_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = city_data[(city_data[col] < lower_bound) | (city_data[col] > upper_bound)]
                    outlier_stats[city][col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(city_data)) * 100,
                        'bounds': (lower_bound, upper_bound)
                    }

        for city, stats in outlier_stats.items():
            print(f"\n{city}:")
            for var, stat in stats.items():
                print(f"  {var}: {stat['count']} outliers ({stat['percentage']:.1f}%)")

        return {
            'city_stats': city_stats,
            'normality_results': normality_results,
            'outlier_stats': outlier_stats
        }

    def perform_time_series_analysis(self):
        """Advanced time series analysis"""
        if self.raw_data is None:
            print("❌ No data available. Run collect_sample_data() first.")
            return

        print("\n⏰ ADVANCED TIME SERIES ANALYSIS")
        print("=" * 50)

        # Stationarity tests
        print("\n1. STATIONARITY TESTS:")
        stationarity_results = {}

        for city in self.raw_data['city'].unique():
            city_data = self.raw_data[self.raw_data['city'] == city].copy()
            city_data = city_data.sort_values('timestamp')

            stationarity_results[city] = {}

            for col in ['aqi', 'pm2_5']:
                if col in city_data.columns and len(city_data[col].dropna()) > 10:
                    try:
                        # ADF Test
                        adf_result = adfuller(city_data[col].dropna())
                        stationarity_results[city][col] = {
                            'adf_statistic': adf_result[0],
                            'adf_pvalue': adf_result[1],
                            'adf_critical_values': adf_result[4],
                            'stationary': adf_result[1] < 0.05
                        }
                    except:
                        stationarity_results[city][col] = {'error': 'Test failed'}

        for city, tests in stationarity_results.items():
            print(f"\n{city}:")
            for var, result in tests.items():
                if 'error' not in result:
                    stationary_status = "✅ Stationary" if result['stationary'] else "❌ Non-stationary"
                    print(f"  {var}: ADF p-value={result['adf_pvalue']:.4f} ({stationary_status})")

        # Seasonal decomposition
        print("\n2. SEASONAL DECOMPOSITION:")
        fig, axes = plt.subplots(len(self.raw_data['city'].unique()), 4, figsize=(20, 15))
        fig.suptitle('Seasonal Decomposition by City', fontsize=16)

        for i, city in enumerate(self.raw_data['city'].unique()):
            city_data = self.raw_data[self.raw_data['city'] == city].copy()
            city_data = city_data.sort_values('timestamp').set_index('timestamp')

            if len(city_data) > 24 and 'aqi' in city_data.columns:
                try:
                    # Resample to daily for better decomposition
                    daily_data = city_data['aqi'].resample('D').mean().dropna()

                    if len(daily_data) >= 7:  # Need at least 7 points for weekly seasonality
                        decomposition = seasonal_decompose(daily_data, model='additive', period=7)

                        axes[i, 0].plot(decomposition.observed, color=self.city_colors.get(city, 'blue'))
                        axes[i, 0].set_title(f'{city} - Observed')
                        axes[i, 0].tick_params(axis='x', rotation=45)

                        axes[i, 1].plot(decomposition.trend, color='red')
                        axes[i, 1].set_title('Trend')

                        axes[i, 2].plot(decomposition.seasonal, color='green')
                        axes[i, 2].set_title('Seasonal')

                        axes[i, 3].plot(decomposition.resid, color='orange')
                        axes[i, 3].set_title('Residual')
                except Exception as e:
                    print(f"Could not decompose {city} data: {e}")

        plt.tight_layout()
        plt.savefig('../reports/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
        plt.show()

        return stationarity_results

    def perform_correlation_network_analysis(self):
        """Analyze correlation networks between features"""
        if self.raw_data is None:
            print("❌ No data available. Run collect_sample_data() first.")
            return

        print("\n🔗 CORRELATION NETWORK ANALYSIS")
        print("=" * 50)

        # Calculate correlation matrix for each city
        correlation_matrices = {}
        feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'clouds',
                       'visibility', 'aqi', 'pm2_5', 'pm10', 'co', 'no', 'no2', 'o3', 'so2']

        for city in self.raw_data['city'].unique():
            city_data = self.raw_data[self.raw_data['city'] == city]
            available_cols = [col for col in feature_cols if col in city_data.columns]

            if len(available_cols) > 1:
                corr_matrix = city_data[available_cols].corr()
                correlation_matrices[city] = corr_matrix

        # Create correlation networks
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Feature Correlation Networks by City', fontsize=16)

        for i, (city, corr_matrix) in enumerate(correlation_matrices.items()):
            if i < 6:  # Limit to 6 subplots
                ax = axes[i // 3, i % 3]

                # Create network graph
                G = nx.Graph()

                # Add nodes
                for feature in corr_matrix.columns:
                    G.add_node(feature)

                # Add edges for strong correlations (|r| > 0.3)
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.3:
                            G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j],
                                     weight=abs(corr_value), sign='positive' if corr_value > 0 else 'negative')

                # Draw network
                pos = nx.spring_layout(G, k=1, iterations=50)

                # Edge colors based on correlation sign
                edge_colors = ['red' if G[u][v]['sign'] == 'positive' else 'blue'
                             for u, v in G.edges()]

                nx.draw(G, pos, with_labels=True, node_color='lightblue',
                       node_size=800, font_size=8, font_weight='bold',
                       edge_color=edge_colors, width=2, ax=ax)

                ax.set_title(f'{city} Correlation Network')

        plt.tight_layout()
        plt.savefig('../reports/correlation_networks.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Find most correlated feature pairs
        print("\n3. TOP CORRELATION PAIRS:")
        all_correlations = []

        for city, corr_matrix in correlation_matrices.items():
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    all_correlations.append({
                        'city': city,
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    })

        corr_df = pd.DataFrame(all_correlations)
        top_correlations = corr_df.nlargest(20, 'abs_correlation')

        for _, row in top_correlations.iterrows():
            strength = "Strong" if abs(row['correlation']) > 0.7 else "Moderate" if abs(row['correlation']) > 0.5 else "Weak"
            direction = "Positive" if row['correlation'] > 0 else "Negative"
            print(f"  {row['city']}: {row['feature1']} ↔ {row['feature2']} ({direction}, {strength}: {row['correlation']:.3f})")

        return correlation_matrices

    def perform_anomaly_detection(self):
        """Advanced anomaly detection using multiple methods"""
        if self.raw_data is None:
            print("❌ No data available. Run collect_sample_data() first.")
            return

        print("\n🚨 ANOMALY DETECTION ANALYSIS")
        print("=" * 50)

        anomaly_results = {}

        for city in self.raw_data['city'].unique():
            city_data = self.raw_data[self.raw_data['city'] == city].copy()

            # Prepare features for anomaly detection
            feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'pm2_5']
            available_cols = [col for col in feature_cols if col in city_data.columns]

            if len(city_data) > 10 and len(available_cols) > 1:
                X = city_data[available_cols].dropna()

                if len(X) > 10:
                    anomaly_results[city] = {}

                    # Method 1: Isolation Forest
                    try:
                        iso_forest = IsolationForest(contamination=0.1, random_state=42)
                        iso_predictions = iso_forest.fit_predict(X)
                        iso_anomalies = np.sum(iso_predictions == -1)
                        anomaly_results[city]['isolation_forest'] = {
                            'anomalies': iso_anomalies,
                            'percentage': (iso_anomalies / len(X)) * 100
                        }
                    except:
                        anomaly_results[city]['isolation_forest'] = {'error': 'Failed'}

                    # Method 2: Elliptic Envelope (for multivariate normal data)
                    try:
                        elliptic_env = EllipticEnvelope(contamination=0.1, random_state=42)
                        elliptic_predictions = elliptic_env.fit_predict(X)
                        elliptic_anomalies = np.sum(elliptic_predictions == -1)
                        anomaly_results[city]['elliptic_envelope'] = {
                            'anomalies': elliptic_anomalies,
                            'percentage': (elliptic_anomalies / len(X)) * 100
                        }
                    except:
                        anomaly_results[city]['elliptic_envelope'] = {'error': 'Failed'}

                    # Method 3: Statistical (Z-score based)
                    try:
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        z_scores = np.abs(X_scaled)
                        statistical_anomalies = np.sum(np.any(z_scores > 3, axis=1))
                        anomaly_results[city]['z_score'] = {
                            'anomalies': statistical_anomalies,
                            'percentage': (statistical_anomalies / len(X)) * 100
                        }
                    except:
                        anomaly_results[city]['z_score'] = {'error': 'Failed'}

        # Print results
        print("\nANOMALY DETECTION RESULTS:")
        for city, methods in anomaly_results.items():
            print(f"\n{city}:")
            for method, result in methods.items():
                if 'error' not in result:
                    print(f"  {method}: {result['anomalies']} anomalies ({result['percentage']:.1f}%)")

        # Visualize anomalies for one city
        if anomaly_results:
            sample_city = list(anomaly_results.keys())[0]
            city_data = self.raw_data[self.raw_data['city'] == sample_city].copy()

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Anomaly Detection Visualization - {sample_city}', fontsize=16)

            # Time series with anomalies
            axes[0, 0].plot(city_data['timestamp'], city_data['aqi'], alpha=0.7)
            axes[0, 0].set_title('AQI Time Series')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Scatter plot colored by anomaly score
            if len(city_data) > 10:
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    features = ['temperature', 'humidity', 'aqi', 'pm2_5']
                    available_features = [f for f in features if f in city_data.columns]

                    if len(available_features) >= 2:
                        X = city_data[available_features].dropna()
                        if len(X) > 10:
                            anomaly_scores = iso_forest.fit_predict(X)

                            scatter = axes[0, 1].scatter(X['temperature'], X['aqi'],
                                                        c=anomaly_scores, cmap='coolwarm', alpha=0.6)
                            axes[0, 1].set_title('Temperature vs AQI (Anomaly Colored)')
                            axes[0, 1].set_xlabel('Temperature (°C)')
                            axes[0, 1].set_ylabel('AQI')
                            plt.colorbar(scatter, ax=axes[0, 1])
                except:
                    axes[0, 1].text(0.5, 0.5, 'Anomaly detection failed',
                                   transform=axes[0, 1].transAxes, ha='center')

            # Distribution plots
            if 'aqi' in city_data.columns:
                sns.histplot(city_data['aqi'], kde=True, ax=axes[1, 0])
                axes[1, 0].set_title('AQI Distribution')

            if 'pm2_5' in city_data.columns:
                sns.histplot(city_data['pm2_5'], kde=True, ax=axes[1, 1])
                axes[1, 1].set_title('PM2.5 Distribution')

            plt.tight_layout()
            plt.savefig('../reports/anomaly_detection.png', dpi=300, bbox_inches='tight')
            plt.show()

        return anomaly_results

    def perform_clustering_analysis(self):
        """Perform clustering analysis to identify patterns"""
        if self.raw_data is None:
            print("❌ No data available. Run collect_sample_data() first.")
            return

        print("\n🎯 CLUSTERING ANALYSIS")
        print("=" * 50)

        # Prepare data for clustering
        feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'pm2_5']
        available_cols = [col for col in feature_cols if col in self.raw_data.columns]

        if len(available_cols) < 3:
            print("❌ Insufficient features for clustering")
            return

        # Scale the data
        scaler = StandardScaler()
        X = self.raw_data[available_cols].dropna()
        X_scaled = scaler.fit_transform(X)

        if len(X_scaled) < 10:
            print("❌ Insufficient data points for clustering")
            return

        # Determine optimal number of clusters using elbow method
        print("\n1. DETERMINING OPTIMAL CLUSTERS (Elbow Method):")
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(8, len(X_scaled)))

        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)

                if len(X_scaled) > k:
                    silhouette = silhouette_score(X_scaled, kmeans.labels_)
                    silhouette_scores.append(silhouette)
                else:
                    silhouette_scores.append(0)
            except:
                inertias.append(0)
                silhouette_scores.append(0)

        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Elbow Method for Optimal k')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.grid(True, alpha=0.3)

        ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
        ax2.set_title('Silhouette Score vs k')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../reports/clustering_elbow.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Choose optimal k (where silhouette score is highest)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")

        # Perform clustering with optimal k
        print(f"\n2. CLUSTERING WITH K={optimal_k}:")
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Add cluster labels to data
        X_clustered = X.copy()
        X_clustered['cluster'] = clusters

        # Analyze cluster characteristics
        cluster_stats = X_clustered.groupby('cluster')[available_cols].agg(['mean', 'std', 'count'])
        cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]

        print("\nCLUSTER CHARACTERISTICS:")
        for cluster in range(optimal_k):
            cluster_data = cluster_stats.loc[cluster]
            print(f"\nCluster {cluster} (n={cluster_data['aqi_count']:.0f}):")
            for col in available_cols:
                mean_val = cluster_data[f'{col}_mean']
                std_val = cluster_data[f'{col}_std']
                print(f"  {col}: {mean_val:.2f} ± {std_val:.2f}")

        # Visualize clusters
        if len(available_cols) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # 2D scatter plot of first two features
            scatter = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters,
                                    cmap='viridis', alpha=0.6, s=50)
            axes[0].set_xlabel(available_cols[0])
            axes[0].set_ylabel(available_cols[1])
            axes[0].set_title('Clusters (First 2 Features)')
            plt.colorbar(scatter, ax=axes[0])

            # Cluster centers
            centers = kmeans.cluster_centers_
            axes[0].scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X')

            # PCA for dimensionality reduction
            try:
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                scatter_pca = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                                            cmap='viridis', alpha=0.6, s=50)
                axes[1].set_xlabel('PC1')
                axes[1].set_ylabel('PC2')
                axes[1].set_title(f'Clusters (PCA) - Explained Variance: {pca.explained_variance_ratio_[:2].sum():.2f}')
                plt.colorbar(scatter_pca, ax=axes[1])

                # Plot PCA centers
                centers_pca = pca.transform(centers)
                axes[1].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.8, marker='X')

            except Exception as e:
                axes[1].text(0.5, 0.5, f'PCA failed: {e}', transform=axes[1].transAxes, ha='center')

            plt.tight_layout()
            plt.savefig('../reports/clustering_visualization.png', dpi=300, bbox_inches='tight')
            plt.show()

        return {
            'optimal_k': optimal_k,
            'cluster_stats': cluster_stats,
            'cluster_labels': clusters,
            'silhouette_scores': silhouette_scores
        }

    def perform_predictive_modeling_insights(self):
        """Analyze data from predictive modeling perspective"""
        if self.raw_data is None:
            print("❌ No data available. Run collect_sample_data() first.")
            return

        print("\n🤖 PREDICTIVE MODELING INSIGHTS")
        print("=" * 50)

        # Feature importance analysis using correlation
        print("\n1. FEATURE IMPORTANCE ANALYSIS:")

        feature_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'clouds',
                       'visibility', 'pm2_5', 'pm10', 'co', 'no', 'no2', 'o3', 'so2']
        target_vars = ['aqi', 'pm2_5']

        feature_importance = {}

        for target in target_vars:
            if target in self.raw_data.columns:
                correlations = {}
                available_features = [col for col in feature_cols if col in self.raw_data.columns and col != target]

                for feature in available_features:
                    try:
                        corr = abs(self.raw_data[feature].corr(self.raw_data[target]))
                        correlations[feature] = corr
                    except:
                        correlations[feature] = 0

                # Sort by correlation strength
                feature_importance[target] = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))

        for target, features in feature_importance.items():
            print(f"\nTop features for predicting {target}:")
            for i, (feature, importance) in enumerate(list(features.items())[:10]):
                print(f"  {i+1}. {feature}: {importance:.4f}")

        # Feature engineering suggestions
        print("\n2. FEATURE ENGINEERING SUGGESTIONS:")

        # Check for non-linear relationships
        print("\nNon-linear relationships detected:")
        for target in target_vars:
            if target in self.raw_data.columns:
                for feature in feature_cols[:5]:  # Check first 5 features
                    if feature in self.raw_data.columns and feature != target:
                        try:
                            # Simple quadratic relationship check
                            corr_linear = abs(self.raw_data[feature].corr(self.raw_data[target]))
                            quadratic_corr = abs((self.raw_data[feature] ** 2).corr(self.raw_data[target]))

                            if quadratic_corr > corr_linear * 1.2:
                                print(f"  {feature} → {target}: Quadratic relationship suggested (linear: {corr_linear:.3f}, quadratic: {quadratic_corr:.3f})")
                        except:
                            pass

        # Interaction features
        print("\nPotential interaction features:")
        strong_corr_pairs = []
        corr_matrix = self.raw_data[feature_cols].corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.5:
                    strong_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

        for feat1, feat2, corr in strong_corr_pairs[:5]:
            print(f"  {feat1} × {feat2} (correlation: {corr:.3f})")

        # Time-based features
        print("\nTime-based feature suggestions:")
        if 'timestamp' in self.raw_data.columns:
            print("  ✅ Hour of day (cyclical encoding)")
            print("  ✅ Day of week (categorical)")
            print("  ✅ Month/season (cyclical encoding)")
            print("  ✅ Lag features (1h, 3h, 6h, 12h, 24h)")
            print("  ✅ Rolling statistics (mean, std, min, max)")

        return feature_importance

    def create_comprehensive_report(self):
        """Create a comprehensive EDA report"""
        print("\n📋 GENERATING COMPREHENSIVE EDA REPORT")
        print("=" * 50)

        if self.raw_data is None:
            print("❌ No data available. Run collect_sample_data() first.")
            return

        report = {
            'data_overview': {
                'total_records': len(self.raw_data),
                'cities_covered': len(self.raw_data['city'].unique()),
                'features_available': len(self.raw_data.columns),
                'date_range': {
                    'start': self.raw_data['timestamp'].min(),
                    'end': self.raw_data['timestamp'].max()
                }
            },
            'quality_metrics': {
                'missing_data_percentage': (self.raw_data.isnull().sum().sum() / (len(self.raw_data) * len(self.raw_data.columns))) * 100,
                'duplicate_records': self.raw_data.duplicated().sum()
            }
        }

        # AQI Statistics
        if 'aqi' in self.raw_data.columns:
            aqi_stats = self.raw_data['aqi'].describe()
            report['aqi_statistics'] = {
                'mean': aqi_stats['mean'],
                'std': aqi_stats['std'],
                'min': aqi_stats['min'],
                'max': aqi_stats['max'],
                'median': aqi_stats['50%']
            }

            # AQI Categories distribution
            def categorize_aqi(aqi):
                if aqi <= 50: return 'Good'
                elif aqi <= 100: return 'Moderate'
                elif aqi <= 150: return 'Unhealthy for Sensitive Groups'
                elif aqi <= 200: return 'Unhealthy'
                elif aqi <= 300: return 'Very Unhealthy'
                else: return 'Hazardous'

            aqi_categories = self.raw_data['aqi'].apply(categorize_aqi).value_counts()
            report['aqi_categories'] = aqi_categories.to_dict()

        # City-wise statistics
        city_stats = {}
        for city in self.raw_data['city'].unique():
            city_data = self.raw_data[self.raw_data['city'] == city]
            city_stats[city] = {
                'records': len(city_data),
                'avg_aqi': city_data['aqi'].mean() if 'aqi' in city_data.columns else None,
                'avg_pm25': city_data['pm2_5'].mean() if 'pm2_5' in city_data.columns else None,
                'avg_temp': city_data['temperature'].mean() if 'temperature' in city_data.columns else None
            }
        report['city_statistics'] = city_stats

        # Save report
        os.makedirs('../reports', exist_ok=True)
        report_file = f"../reports/advanced_eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        import json
        with open(report_file, 'w') as f:
            # Convert timestamps to strings for JSON serialization
            json_report = json.dumps(report, default=str, indent=2)
            f.write(json_report)

        print(f"📄 Report saved to {report_file}")

        # Print summary
        print("\n📊 EDA SUMMARY:")
        print(f"• Total Records: {report['data_overview']['total_records']}")
        print(f"• Cities Covered: {report['data_overview']['cities_covered']}")
        print(f"• Date Range: {report['data_overview']['date_range']['start']} to {report['data_overview']['date_range']['end']}")
        print(f"• Missing Data: {report['quality_metrics']['missing_data_percentage']:.2f}%")
        print(f"• Duplicate Records: {report['quality_metrics']['duplicate_records']}")

        if 'aqi_statistics' in report:
            print(f"• Average AQI: {report['aqi_statistics']['mean']:.1f}")
            print(f"• AQI Range: {report['aqi_statistics']['min']:.1f} - {report['aqi_statistics']['max']:.1f}")

        print("\n🏙️  CITY RANKING BY AIR QUALITY:")
        city_aqi = [(city, stats['avg_aqi']) for city, stats in city_stats.items() if stats['avg_aqi'] is not None]
        city_aqi.sort(key=lambda x: x[1])

        for i, (city, aqi) in enumerate(city_aqi, 1):
            status = "🟢 Good" if aqi <= 50 else "🟡 Moderate" if aqi <= 100 else "🟠 Unhealthy" if aqi <= 150 else "🔴 Very Unhealthy"
            print(f"  {i}. {city}: {aqi:.1f} ({status})")

        return report

    def run_complete_advanced_eda(self):
        """Run the complete advanced EDA pipeline"""
        print("🚀 STARTING ADVANCED EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        # Step 1: Collect data
        self.collect_sample_data(days=7)

        # Step 2: Statistical analysis
        stat_results = self.perform_advanced_statistical_analysis()

        # Step 3: Time series analysis
        ts_results = self.perform_time_series_analysis()

        # Step 4: Correlation network analysis
        corr_results = self.perform_correlation_network_analysis()

        # Step 5: Anomaly detection
        anomaly_results = self.perform_anomaly_detection()

        # Step 6: Clustering analysis
        cluster_results = self.perform_clustering_analysis()

        # Step 7: Predictive modeling insights
        modeling_insights = self.perform_predictive_modeling_insights()

        # Step 8: Comprehensive report
        report = self.create_comprehensive_report()

        print("\n🎉 ADVANCED EDA COMPLETED!")
        print("=" * 60)
        print("Generated insights:")
        print("• Statistical distributions and normality tests")
        print("• Time series stationarity and seasonal decomposition")
        print("• Feature correlation networks")
        print("• Multi-method anomaly detection")
        print("• Optimal clustering analysis")
        print("• Predictive modeling feature importance")
        print("• Comprehensive quality report")
        print("\n📁 Check the 'reports/' directory for visualizations and detailed reports.")

        return {
            'statistical': stat_results,
            'time_series': ts_results,
            'correlations': corr_results,
            'anomalies': anomaly_results,
            'clusters': cluster_results,
            'modeling': modeling_insights,
            'report': report
        }

# Main execution
if __name__ == "__main__":
    # Initialize advanced EDA
    eda = AdvancedAQIEDA()

    # Run complete analysis
    results = eda.run_complete_advanced_eda()

    print("\n✅ Advanced EDA analysis completed successfully!")
    print("📊 All results and visualizations saved to reports/ directory")


