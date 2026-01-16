"""
Simplified Advanced EDA for AQI Prediction System
================================================

This script performs advanced EDA without complex dependencies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append('..')

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

class SimplifiedAQIEDA:
    """
    Simplified Advanced EDA for AQI data
    """

    def __init__(self):
        self.raw_data = None
        self.aqi_colors = {
            'Good': '#00e400',
            'Moderate': '#ffff00',
            'Unhealthy for Sensitive Groups': '#ff7e00',
            'Unhealthy': '#ff0000',
            'Very Unhealthy': '#8f3f97',
            'Hazardous': '#7e0023'
        }

    def generate_sample_data(self):
        """Generate realistic sample AQI data for analysis"""
        print("🔄 Generating sample AQI data...")

        cities = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
        records_per_city = 168  # 7 days * 24 hours

        all_data = []

        # Base parameters for each city (realistic AQI patterns)
        city_params = {
            'Delhi': {'base_aqi': 120, 'seasonal_amp': 30, 'daily_amp': 40, 'trend': 0.1},
            'Mumbai': {'base_aqi': 85, 'seasonal_amp': 25, 'daily_amp': 35, 'trend': 0.05},
            'Bangalore': {'base_aqi': 65, 'seasonal_amp': 20, 'daily_amp': 25, 'trend': 0.02},
            'Chennai': {'base_aqi': 55, 'seasonal_amp': 15, 'daily_amp': 20, 'trend': -0.01},
            'Kolkata': {'base_aqi': 95, 'seasonal_amp': 35, 'daily_amp': 45, 'trend': 0.08},
            'Hyderabad': {'base_aqi': 75, 'seasonal_amp': 22, 'daily_amp': 30, 'trend': 0.03}
        }

        base_time = datetime.now() - timedelta(days=7)

        for city in cities:
            params = city_params[city]

            for hour in range(records_per_city):
                timestamp = base_time + timedelta(hours=hour)

                # Generate realistic AQI patterns
                # Seasonal component (daily cycle)
                daily_cycle = params['daily_amp'] * np.sin(2 * np.pi * (timestamp.hour - 6) / 24)

                # Weekly pattern (higher on weekdays)
                weekly_factor = 1.2 if timestamp.weekday() < 5 else 0.8

                # Random noise
                noise = np.random.normal(0, 10)

                # Trend component
                trend = params['trend'] * hour

                # Calculate AQI
                aqi = params['base_aqi'] + daily_cycle + trend + noise * weekly_factor
                aqi = max(20, min(400, aqi))  # Clamp to realistic range

                # Calculate PM2.5 from AQI (simplified relationship)
                pm25 = (aqi - 20) * 0.5 + np.random.normal(0, 5)

                # Generate weather data
                temp_base = 25 if city in ['Chennai', 'Hyderabad'] else 20
                temperature = temp_base + 5 * np.sin(2 * np.pi * timestamp.hour / 24) + np.random.normal(0, 2)

                humidity = 65 + 15 * np.sin(2 * np.pi * (timestamp.hour - 12) / 24) + np.random.normal(0, 5)
                humidity = max(20, min(90, humidity))

                pressure = 1013 + np.random.normal(0, 5)
                wind_speed = 8 + 3 * np.random.normal(0, 1)
                wind_speed = max(0, wind_speed)

                # Generate pollutant data
                co = 0.5 + 0.3 * (aqi / 100) + np.random.normal(0, 0.1)
                no2 = 20 + 15 * (aqi / 100) + np.random.normal(0, 3)
                so2 = 5 + 8 * (aqi / 100) + np.random.normal(0, 1)
                o3 = 25 + 20 * (aqi / 100) + np.random.normal(0, 5)

                record = {
                    'city': city,
                    'timestamp': timestamp,
                    'temperature': round(temperature, 1),
                    'humidity': round(humidity, 1),
                    'pressure': round(pressure, 1),
                    'wind_speed': round(wind_speed, 1),
                    'wind_direction': round(np.random.uniform(0, 360), 1),
                    'clouds': round(np.random.uniform(0, 100), 1),
                    'visibility': round(np.random.uniform(5, 15), 1),
                    'aqi': round(aqi, 1),
                    'pm2_5': round(max(5, pm25), 1),
                    'pm10': round(max(10, pm25 * 1.5), 1),
                    'co': round(max(0.1, co), 2),
                    'no': round(np.random.uniform(1, 10), 2),
                    'no2': round(max(5, no2), 2),
                    'o3': round(max(10, o3), 2),
                    'so2': round(max(1, so2), 2),
                    'nh3': round(np.random.uniform(1, 8), 2)
                }

                all_data.append(record)

        self.raw_data = pd.DataFrame(all_data)
        print(f"✅ Generated {len(self.raw_data)} records for {len(cities)} cities")

        # Save data
        os.makedirs('../data', exist_ok=True)
        filename = f"../data/advanced_eda_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.raw_data.to_csv(filename, index=False)
        print(f"💾 Data saved to {filename}")

        return self.raw_data

    def perform_statistical_analysis(self):
        """Perform advanced statistical analysis"""
        if self.raw_data is None:
            print("❌ No data available")
            return

        print("\n📊 STATISTICAL ANALYSIS")
        print("=" * 50)

        # Basic statistics by city
        print("\n🏙️  CITY-WISE AQI STATISTICS:")
        city_stats = self.raw_data.groupby('city')['aqi'].agg(['mean', 'std', 'min', 'max', 'median'])
        print(city_stats.round(2))

        # Overall statistics
        print(f"\n🌍 OVERALL STATISTICS:")
        print(f"• Total Records: {len(self.raw_data)}")
        print(f"• Cities: {len(self.raw_data['city'].unique())}")
        print(f"• Date Range: {self.raw_data['timestamp'].min()} to {self.raw_data['timestamp'].max()}")
        print(f"• Average AQI: {self.raw_data['aqi'].mean():.1f}")
        print(f"• AQI Range: {self.raw_data['aqi'].min():.1f} - {self.raw_data['aqi'].max():.1f}")

        # AQI Categories
        def categorize_aqi(aqi):
            if aqi <= 50: return 'Good'
            elif aqi <= 100: return 'Moderate'
            elif aqi <= 150: return 'Unhealthy for Sensitive Groups'
            elif aqi <= 200: return 'Unhealthy'
            elif aqi <= 300: return 'Very Unhealthy'
            else: return 'Hazardous'

        aqi_categories = self.raw_data['aqi'].apply(categorize_aqi).value_counts()
        print("\n🏷️  AQI CATEGORIES:")
        for category, count in aqi_categories.items():
            percentage = (count / len(self.raw_data)) * 100
            print(f"• {category}: {count} ({percentage:.1f}%)")

        # Normality tests
        print("\n🧪 NORMALITY TESTS:")
        from scipy.stats import shapiro

        for city in self.raw_data['city'].unique():
            city_data = self.raw_data[self.raw_data['city'] == city]['aqi']
            try:
                stat, p_value = shapiro(city_data)
                normal = "✅ Normal" if p_value > 0.05 else "❌ Not Normal"
                print(f"• {city}: p-value = {p_value:.4f} ({normal})")
            except:
                print(f"• {city}: Test failed (insufficient data)")

        return city_stats

    def create_visualizations(self):
        """Create comprehensive visualizations"""
        if self.raw_data is None:
            print("❌ No data available")
            return

        print("\n📈 CREATING VISUALIZATIONS")
        print("=" * 50)

        # Create reports directory
        os.makedirs('../reports', exist_ok=True)

        # 1. AQI Distribution by City
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.raw_data, x='city', y='aqi', palette='Set3')
        plt.title('AQI Distribution by City', fontsize=16, fontweight='bold')
        plt.xlabel('City', fontsize=12)
        plt.ylabel('AQI Value', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../reports/aqi_distribution_by_city.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Time Series Analysis - AQI over time
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('AQI Time Series Analysis by City', fontsize=16, fontweight='bold')

        cities = self.raw_data['city'].unique()
        for i, city in enumerate(cities):
            ax = axes[i // 3, i % 3]
            city_data = self.raw_data[self.raw_data['city'] == city].copy()
            city_data = city_data.sort_values('timestamp')

            ax.plot(city_data['timestamp'], city_data['aqi'], linewidth=2, alpha=0.8)
            ax.set_title(f'{city}', fontsize=12, fontweight='bold')
            ax.set_ylabel('AQI', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('../reports/aqi_time_series.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Correlation Heatmap
        numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']
        available_cols = [col for col in numeric_cols if col in self.raw_data.columns]

        if len(available_cols) > 1:
            plt.figure(figsize=(14, 10))
            correlation_matrix = self.raw_data[available_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig('../reports/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 4. Weather vs AQI Relationships
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Weather Factors vs AQI Relationships', fontsize=16, fontweight='bold')

        # Temperature vs AQI
        sns.scatterplot(data=self.raw_data, x='temperature', y='aqi', hue='city',
                       alpha=0.6, ax=axes[0, 0])
        axes[0, 0].set_title('Temperature vs AQI')
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('AQI')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Humidity vs AQI
        sns.scatterplot(data=self.raw_data, x='humidity', y='aqi', hue='city',
                       alpha=0.6, ax=axes[0, 1])
        axes[0, 1].set_title('Humidity vs AQI')
        axes[0, 1].set_xlabel('Humidity (%)')
        axes[0, 1].set_ylabel('AQI')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Wind Speed vs AQI
        sns.scatterplot(data=self.raw_data, x='wind_speed', y='aqi', hue='city',
                       alpha=0.6, ax=axes[1, 0])
        axes[1, 0].set_title('Wind Speed vs AQI')
        axes[1, 0].set_xlabel('Wind Speed (m/s)')
        axes[1, 0].set_ylabel('AQI')

        # PM2.5 vs AQI
        sns.scatterplot(data=self.raw_data, x='pm2_5', y='aqi', hue='city',
                       alpha=0.6, ax=axes[1, 1])
        axes[1, 1].set_title('PM2.5 vs AQI')
        axes[1, 1].set_xlabel('PM2.5 (μg/m³)')
        axes[1, 1].set_ylabel('AQI')

        plt.tight_layout()
        plt.savefig('../reports/weather_vs_aqi.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 5. Hourly Patterns
        self.raw_data['hour'] = self.raw_data['timestamp'].dt.hour
        hourly_patterns = self.raw_data.groupby('hour')['aqi'].mean()

        plt.figure(figsize=(12, 6))
        plt.plot(hourly_patterns.index, hourly_patterns.values, marker='o',
                linewidth=3, markersize=8, color='darkblue')
        plt.title('Average AQI by Hour of Day', fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Average AQI', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24))
        plt.axhline(y=self.raw_data['aqi'].mean(), color='red', linestyle='--',
                   alpha=0.7, label=f'Overall Mean: {self.raw_data["aqi"].mean():.1f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('../reports/hourly_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 6. Pollutant Analysis
        pollutants = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']
        available_pollutants = [p for p in pollutants if p in self.raw_data.columns]

        if available_pollutants:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Pollutant Distributions by City', fontsize=16, fontweight='bold')

            for i, pollutant in enumerate(available_pollutants):
                if i < 6:
                    ax = axes[i // 3, i % 3]
                    sns.boxplot(data=self.raw_data, x='city', y=pollutant, ax=ax)
                    ax.set_title(f'{pollutant.upper()} Distribution')
                    ax.set_xlabel('City')
                    ax.set_ylabel(f'{pollutant.upper()} Concentration')
                    ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig('../reports/pollutant_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()

    def perform_clustering_analysis(self):
        """Perform clustering analysis"""
        if self.raw_data is None:
            print("❌ No data available")
            return

        print("\n🎯 CLUSTERING ANALYSIS")
        print("=" * 50)

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Prepare data for clustering
        feature_cols = ['temperature', 'humidity', 'wind_speed', 'aqi', 'pm2_5']
        available_cols = [col for col in feature_cols if col in self.raw_data.columns]

        if len(available_cols) < 3:
            print("❌ Insufficient features for clustering")
            return

        # Scale the data
        scaler = StandardScaler()
        X = self.raw_data[available_cols].dropna()
        X_scaled = scaler.fit_transform(X)

        if len(X_scaled) < 10:
            print("❌ Insufficient data for clustering")
            return

        # Find optimal number of clusters
        silhouette_scores = []
        k_range = range(2, min(7, len(X_scaled)))

        print("Finding optimal number of clusters...")
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                silhouette = silhouette_score(X_scaled, kmeans.labels_)
                silhouette_scores.append(silhouette)
                print(f"  k={k}: Silhouette Score = {silhouette:.3f}")
            except:
                silhouette_scores.append(0)

        # Choose optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n✅ Optimal number of clusters: {optimal_k}")

        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Add cluster labels
        X_clustered = X.copy()
        X_clustered['cluster'] = clusters

        # Analyze clusters
        print("\n📊 CLUSTER ANALYSIS:")
        cluster_stats = X_clustered.groupby('cluster')[available_cols].mean()
        cluster_sizes = X_clustered['cluster'].value_counts()

        for cluster in range(optimal_k):
            print(f"\nCluster {cluster} (n={cluster_sizes[cluster]}):")
            for feature in available_cols:
                mean_val = cluster_stats.loc[cluster, feature]
                print(f"  {feature}: {mean_val:.2f}")

        # Visualize clusters
        if len(available_cols) >= 2:
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters,
                                cmap='viridis', alpha=0.6, s=50)
            plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                       c='red', s=200, alpha=0.8, marker='X', label='Centroids')
            plt.xlabel(available_cols[0])
            plt.ylabel(available_cols[1])
            plt.title(f'AQI Data Clusters (k={optimal_k})', fontsize=14, fontweight='bold')
            plt.colorbar(scatter)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('../reports/clustering_results.png', dpi=300, bbox_inches='tight')
            plt.show()

        return clusters

    def create_comprehensive_report(self):
        """Create a comprehensive EDA report"""
        if self.raw_data is None:
            print("❌ No data available")
            return

        print("\n📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 50)

        # Calculate key metrics
        report = {
            'data_overview': {
                'total_records': len(self.raw_data),
                'cities_covered': len(self.raw_data['city'].unique()),
                'features_available': len(self.raw_data.columns),
                'date_range': {
                    'start': self.raw_data['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
                    'end': self.raw_data['timestamp'].max().strftime('%Y-%m-%d %H:%M')
                }
            },
            'aqi_statistics': {
                'overall_mean': float(self.raw_data['aqi'].mean()),
                'overall_std': float(self.raw_data['aqi'].std()),
                'overall_min': float(self.raw_data['aqi'].min()),
                'overall_max': float(self.raw_data['aqi'].max()),
                'overall_median': float(self.raw_data['aqi'].median())
            }
        }

        # City-wise statistics
        city_stats = {}
        for city in self.raw_data['city'].unique():
            city_data = self.raw_data[self.raw_data['city'] == city]
            city_stats[city] = {
                'records': int(len(city_data)),
                'avg_aqi': float(city_data['aqi'].mean()),
                'avg_pm25': float(city_data['pm2_5'].mean()) if 'pm2_5' in city_data.columns else None,
                'avg_temp': float(city_data['temperature'].mean()) if 'temperature' in city_data.columns else None,
                'aqi_std': float(city_data['aqi'].std())
            }

        report['city_statistics'] = city_stats

        # AQI Categories
        def categorize_aqi(aqi):
            if aqi <= 50: return 'Good'
            elif aqi <= 100: return 'Moderate'
            elif aqi <= 150: return 'Unhealthy for Sensitive Groups'
            elif aqi <= 200: return 'Unhealthy'
            elif aqi <= 300: return 'Very Unhealthy'
            else: return 'Hazardous'

        aqi_categories = self.raw_data['aqi'].apply(categorize_aqi).value_counts()
        report['aqi_categories'] = aqi_categories.to_dict()

        # Top insights
        city_aqi_ranking = sorted(city_stats.items(), key=lambda x: x[1]['avg_aqi'])

        report['key_insights'] = {
            'best_city': city_aqi_ranking[0][0],
            'worst_city': city_aqi_ranking[-1][0],
            'best_aqi': city_aqi_ranking[0][1]['avg_aqi'],
            'worst_aqi': city_aqi_ranking[-1][1]['avg_aqi'],
            'most_variable_city': max(city_stats.items(), key=lambda x: x[1]['aqi_std'])[0],
            'highest_variability': max(city_stats.items(), key=lambda x: x[1]['aqi_std'])[1]['aqi_std']
        }

        # Save report
        import json
        os.makedirs('../reports', exist_ok=True)
        report_file = f"../reports/advanced_eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Detailed report saved to {report_file}")

        # Print summary
        print("\n📊 ADVANCED EDA SUMMARY:")
        print(f"• Total Records: {report['data_overview']['total_records']}")
        print(f"• Cities Covered: {report['data_overview']['cities_covered']}")
        print(f"• Date Range: {report['data_overview']['date_range']['start']} to {report['data_overview']['date_range']['end']}")
        print(f"• Average AQI: {report['aqi_statistics']['overall_mean']:.1f}")
        print(f"• AQI Range: {report['aqi_statistics']['overall_min']:.1f} - {report['aqi_statistics']['overall_max']:.1f}")

        print("\n🏷️  AQI CATEGORIES:")
        for category, count in aqi_categories.items():
            percentage = (count / len(self.raw_data)) * 100
            print(f"• {category}: {count} ({percentage:.1f}%)")

        print("\n🏙️  CITY RANKING BY AIR QUALITY:")
        for i, (city, stats) in enumerate(city_aqi_ranking, 1):
            status = "🟢 Good" if stats['avg_aqi'] <= 50 else "🟡 Moderate" if stats['avg_aqi'] <= 100 else "🟠 Unhealthy" if stats['avg_aqi'] <= 150 else "🔴 Very Unhealthy"
            print(f"  {i}. {city}: {stats['avg_aqi']:.1f} ({status})")

        print("\n🔑 KEY INSIGHTS:")
        insights = report['key_insights']
        print(f"• Best air quality: {insights['best_city']} (AQI: {insights['best_aqi']:.1f})")
        print(f"• Worst air quality: {insights['worst_city']} (AQI: {insights['worst_aqi']:.1f})")
        print(f"• Most variable city: {insights['most_variable_city']} (Std: {insights['highest_variability']:.1f})")

        return report

    def run_complete_advanced_eda(self):
        """Run the complete advanced EDA pipeline"""
        print("🚀 STARTING ADVANCED EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        # Generate sample data
        self.generate_sample_data()

        # Statistical analysis
        stats_results = self.perform_statistical_analysis()

        # Create visualizations
        self.create_visualizations()

        # Clustering analysis
        clusters = self.perform_clustering_analysis()

        # Comprehensive report
        report = self.create_comprehensive_report()

        print("\n🎉 ADVANCED EDA COMPLETED!")
        print("=" * 60)
        print("Generated insights:")
        print("• Statistical analysis by city")
        print("• AQI distribution patterns")
        print("• Time series analysis")
        print("• Feature correlations")
        print("• Weather vs AQI relationships")
        print("• Hourly and daily patterns")
        print("• Pollutant distributions")
        print("• Clustering analysis")
        print("• Comprehensive quality report")
        print("\n📁 All visualizations saved to 'reports/' directory")

        return {
            'statistics': stats_results,
            'clusters': clusters,
            'report': report
        }

# Main execution
if __name__ == "__main__":
    # Initialize and run advanced EDA
    eda = SimplifiedAQIEDA()
    results = eda.run_complete_advanced_eda()

    print("\n✅ Advanced EDA analysis completed successfully!")
    print("📊 Check the 'reports/' directory for all visualizations and reports.")
