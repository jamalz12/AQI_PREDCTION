"""
Advanced Exploratory Data Analysis for AQI Prediction System
===========================================================

This script performs advanced EDA with realistic data generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
import os
import json

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

class AdvancedEDA:
    """Advanced EDA for AQI data"""

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

    def generate_realistic_data(self):
        """Generate realistic AQI data for Karachi, Pakistan"""
        print("🔄 Generating realistic Karachi AQI dataset...")

        cities = ['Karachi']
        base_time = datetime.now() - timedelta(days=7)
        all_data = []

        # Realistic Karachi parameters based on Pakistan's air quality patterns
        # Karachi typically has moderate to poor air quality with dust storms and traffic
        city_params = {
            'Karachi': {
                'base_aqi': 95,  # Moderate to unhealthy levels typical for Karachi
                'daily_amp': 35,  # Daily variation due to traffic and weather
                'trend': 0.05,    # Slight upward trend due to urbanization
                'seasonal_factor': 1.2,  # Higher in winter due to temperature inversions
                'dust_storm_factor': 1.4,  # Occasional dust storms increase AQI significantly
                'traffic_factor': 1.3      # Heavy traffic contributes to pollution
            }
        }

        for city in cities:
            params = city_params[city]

            for hour in range(168):  # 7 days * 24 hours
                timestamp = base_time + timedelta(hours=hour)

                # Karachi-specific diurnal cycle (traffic patterns)
                # Morning rush hour (7-9 AM), Evening rush hour (5-7 PM)
                if 7 <= timestamp.hour <= 9 or 17 <= timestamp.hour <= 19:
                    traffic_factor = params['traffic_factor']  # Heavy traffic
                elif 10 <= timestamp.hour <= 16:
                    traffic_factor = 1.1  # Moderate traffic
                else:
                    traffic_factor = 0.9  # Light traffic

                # Weekly pattern (higher on weekdays, lower on weekends)
                weekday_factor = 1.2 if timestamp.weekday() < 5 else 0.85

                # Karachi seasonal factor (higher in winter due to inversions)
                month = timestamp.month
                if month in [12, 1, 2]:  # Winter months
                    seasonal_factor = params['seasonal_factor']
                elif month in [5, 6, 7]:  # Summer months
                    seasonal_factor = 0.9  # Slightly better in summer
                else:
                    seasonal_factor = 1.0

                # Dust storm events (occasional in Karachi)
                dust_storm = np.random.choice([1, params['dust_storm_factor']],
                                            p=[0.85, 0.15])  # 15% chance of dust storm

                # Random weather impact
                weather_impact = np.random.normal(1, 0.1)

                # Calculate AQI with Karachi-specific patterns
                aqi = (params['base_aqi'] * traffic_factor * weekday_factor *
                      seasonal_factor * dust_storm * weather_impact + params['trend'] * hour)

                # Add realistic noise (Karachi has more variable pollution)
                aqi += np.random.normal(0, 12)
                aqi = max(25, min(450, aqi))  # Karachi can have very high AQI during dust storms

                # Generate correlated pollutants (Karachi has high particulate matter)
                pm25 = (aqi - 25) * 0.5 + np.random.normal(0, 8)  # Higher PM2.5 in Karachi
                pm10 = pm25 * 1.6 + np.random.normal(0, 5)  # Even higher PM10 due to dust
                co = 0.5 + 0.3 * (aqi / 100) + np.random.normal(0, 0.08)
                no2 = 18 + 15 * (aqi / 100) + np.random.normal(0, 3)  # Higher NO2 from traffic
                so2 = 6 + 8 * (aqi / 100) + np.random.normal(0, 1.2)  # Industrial sources
                o3 = 15 + 12 * (aqi / 100) + np.random.normal(0, 4)

                # Karachi weather data (hot and humid coastal city)
                temp_base = 32  # Karachi is typically hot
                temperature = temp_base + 8 * np.sin(2 * np.pi * (timestamp.hour - 6) / 24) + np.random.normal(0, 3)
                temperature = max(20, min(45, temperature))  # Karachi temperature range

                humidity = 65 + 25 * np.sin(2 * np.pi * (timestamp.hour - 12) / 24) + np.random.normal(0, 8)
                humidity = max(30, min(90, humidity))  # Karachi is humid

                pressure = 1008 + np.random.normal(0, 3)  # Slightly lower pressure due to coastal location
                wind_speed = 8 + 6 * np.random.normal(0, 0.7)  # Karachi has moderate winds
                wind_speed = max(1, wind_speed)

                record = {
                    'city': city,
                    'timestamp': timestamp,
                    'temperature': round(temperature, 1),
                    'humidity': round(humidity, 1),
                    'pressure': round(pressure, 1),
                    'wind_speed': round(wind_speed, 1),
                    'aqi': round(aqi, 1),
                    'pm2_5': round(max(5, pm25), 1),
                    'pm10': round(max(8, pm10), 1),
                    'co': round(max(0.1, co), 2),
                    'no2': round(max(2, no2), 2),
                    'o3': round(max(5, o3), 2),
                    'so2': round(max(0.5, so2), 2)
                }

                all_data.append(record)

        self.raw_data = pd.DataFrame(all_data)
        print(f"✅ Generated {len(self.raw_data)} records across {len(cities)} cities")

        # Save data
        os.makedirs('../data', exist_ok=True)
        filename = f"../data/advanced_eda_realistic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.raw_data.to_csv(filename, index=False)
        print(f"💾 Data saved to {filename}")

        return self.raw_data

    def perform_advanced_analysis(self):
        """Perform comprehensive advanced analysis"""
        if self.raw_data is None:
            print("❌ No data available")
            return

        print("\n📊 ADVANCED STATISTICAL ANALYSIS")
        print("=" * 50)

        # City-wise AQI analysis
        city_stats = self.raw_data.groupby('city')['aqi'].agg(['mean', 'std', 'min', 'max', 'median'])
        print("\n🏙️ CITY-WISE AQI STATISTICS:")
        print(city_stats.round(2))

        # AQI Categories
        def get_aqi_category(aqi):
            if aqi <= 50: return 'Good'
            elif aqi <= 100: return 'Moderate'
            elif aqi <= 150: return 'Unhealthy for Sensitive Groups'
            elif aqi <= 200: return 'Unhealthy'
            elif aqi <= 300: return 'Very Unhealthy'
            else: return 'Hazardous'

        self.raw_data['aqi_category'] = self.raw_data['aqi'].apply(get_aqi_category)
        category_counts = self.raw_data['aqi_category'].value_counts()

        print("\n🏷️ AQI CATEGORIES DISTRIBUTION:")
        for category, count in category_counts.items():
            percentage = (count / len(self.raw_data)) * 100
            print(f"• {category}: {count} ({percentage:.1f}%)")

        # Correlation analysis
        numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'aqi', 'pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']
        correlation_matrix = self.raw_data[numeric_cols].corr()

        print("\n🔗 TOP CORRELATIONS WITH AQI:")
        aqi_correlations = correlation_matrix['aqi'].drop('aqi').abs().sort_values(ascending=False)
        for feature, corr in aqi_correlations.head(5).items():
            direction = "positive" if correlation_matrix.loc[feature, 'aqi'] > 0 else "negative"
            print(f"• {feature}: {correlation_matrix.loc[feature, 'aqi']:.3f} ({direction})")

        # Time series patterns
        self.raw_data['hour'] = self.raw_data['timestamp'].dt.hour
        self.raw_data['day_of_week'] = self.raw_data['timestamp'].dt.day_name()

        hourly_aqi = self.raw_data.groupby('hour')['aqi'].mean()
        print("\n⏰ HOURLY AQI PATTERNS:")
        print("Peak hours:", hourly_aqi.idxmax(), f"(AQI: {hourly_aqi.max():.1f})")
        print("Best hours:", hourly_aqi.idxmin(), f"(AQI: {hourly_aqi.min():.1f})")

        # Day of week patterns
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_aqi = self.raw_data.groupby('day_of_week')['aqi'].mean().reindex(day_order)
        print("\n📅 WEEKLY PATTERNS:")
        print("Highest AQI day:", daily_aqi.idxmax(), f"(AQI: {daily_aqi.max():.1f})")
        print("Lowest AQI day:", daily_aqi.idxmin(), f"(AQI: {daily_aqi.min():.1f})")

        return {
            'city_stats': city_stats,
            'category_counts': category_counts,
            'correlations': correlation_matrix,
            'hourly_patterns': hourly_aqi,
            'daily_patterns': daily_aqi
        }

    def create_advanced_visualizations(self):
        """Create comprehensive visualizations"""
        if self.raw_data is None:
            return

        print("\n📈 CREATING ADVANCED VISUALIZATIONS")
        print("=" * 50)

        os.makedirs('../reports', exist_ok=True)

        # 1. AQI Distribution Comparison
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=self.raw_data, x='city', y='aqi', palette='Set3')
        plt.title('AQI Distribution Across Cities', fontsize=16, fontweight='bold')
        plt.xlabel('City', fontsize=12)
        plt.ylabel('AQI Value', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('../reports/aqi_city_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Correlation Heatmap
        numeric_cols = ['temperature', 'humidity', 'wind_speed', 'aqi', 'pm2_5', 'pm10']
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.raw_data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('../reports/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Time Series with Trends
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('AQI Time Series Analysis by City', fontsize=16, fontweight='bold')

        cities = self.raw_data['city'].unique()
        for i, city in enumerate(cities):
            ax = axes[i // 2, i % 2]
            city_data = self.raw_data[self.raw_data['city'] == city].copy()
            city_data = city_data.sort_values('timestamp')

            ax.plot(city_data['timestamp'], city_data['aqi'], linewidth=2, alpha=0.8)
            ax.set_title(f'{city}', fontsize=12, fontweight='bold')
            ax.set_ylabel('AQI', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('../reports/time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Hourly Patterns
        plt.figure(figsize=(14, 8))
        hourly_aqi = self.raw_data.groupby('hour')['aqi'].mean()
        plt.plot(hourly_aqi.index, hourly_aqi.values, marker='o', linewidth=3,
                markersize=8, color='darkblue', markerfacecolor='lightblue')
        plt.title('Average AQI by Hour of Day', fontsize=16, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Average AQI', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24))
        plt.axhline(y=self.raw_data['aqi'].mean(), color='red', linestyle='--',
                   alpha=0.7, label=f'Overall Mean: {self.raw_data["aqi"].mean():.1f}')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('../reports/hourly_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 5. Weather vs AQI Relationships
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Weather Factors vs AQI Relationships', fontsize=16, fontweight='bold')

        # Temperature vs AQI
        sns.scatterplot(data=self.raw_data, x='temperature', y='aqi', hue='city',
                       alpha=0.6, ax=axes[0, 0])
        axes[0, 0].set_title('Temperature vs AQI')
        axes[0, 0].set_xlabel('Temperature (°C)')
        axes[0, 0].set_ylabel('AQI')

        # Humidity vs AQI
        sns.scatterplot(data=self.raw_data, x='humidity', y='aqi', hue='city',
                       alpha=0.6, ax=axes[0, 1])
        axes[0, 1].set_title('Humidity vs AQI')
        axes[0, 1].set_xlabel('Humidity (%)')
        axes[0, 1].set_ylabel('AQI')

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
        plt.savefig('../reports/weather_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 6. Pollutant Analysis
        pollutants = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']
        available_pollutants = [p for p in pollutants if p in self.raw_data.columns]

        if len(available_pollutants) >= 4:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Pollutant Distributions by City', fontsize=16, fontweight='bold')

            for i, pollutant in enumerate(available_pollutants[:6]):
                ax = axes[i // 3, i % 3]
                sns.boxplot(data=self.raw_data, x='city', y=pollutant, ax=ax)
                ax.set_title(f'{pollutant.upper()} Distribution')
                ax.set_xlabel('City')
                ax.set_ylabel(f'{pollutant.upper()} Concentration')
                ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig('../reports/pollutant_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

    def perform_clustering_analysis(self):
        """Perform clustering analysis"""
        if self.raw_data is None:
            return

        print("\n🎯 CLUSTERING ANALYSIS")
        print("=" * 50)

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        # Prepare features
        feature_cols = ['temperature', 'humidity', 'wind_speed', 'aqi', 'pm2_5']
        available_cols = [col for col in feature_cols if col in self.raw_data.columns]

        if len(available_cols) < 3:
            print("❌ Insufficient features for clustering")
            return

        # Scale data
        scaler = StandardScaler()
        X = self.raw_data[available_cols].dropna()
        X_scaled = scaler.fit_transform(X)

        if len(X_scaled) < 10:
            print("❌ Insufficient data for clustering")
            return

        # Find optimal clusters
        silhouette_scores = []
        k_range = range(2, min(6, len(X_scaled)))

        print("Finding optimal number of clusters...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            silhouette = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(silhouette)
            print(f"  k={k}: Silhouette Score = {silhouette:.3f}")

        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n✅ Optimal clusters: {optimal_k}")

        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Analyze clusters
        X_clustered = X.copy()
        X_clustered['cluster'] = clusters

        cluster_stats = X_clustered.groupby('cluster')[available_cols].mean()
        cluster_sizes = X_clustered['cluster'].value_counts()

        print("\n📊 CLUSTER CHARACTERISTICS:")
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

    def generate_comprehensive_report(self):
        """Generate comprehensive EDA report"""
        if self.raw_data is None:
            return

        print("\n📋 GENERATING COMPREHENSIVE REPORT")
        print("=" * 50)

        # Calculate key metrics
        report = {
            'data_summary': {
                'total_records': int(len(self.raw_data)),
                'cities_covered': len(self.raw_data['city'].unique()),
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

        # City rankings
        city_aqi = self.raw_data.groupby('city')['aqi'].mean().sort_values()
        report['city_rankings'] = {
            'best_city': city_aqi.index[0],
            'worst_city': city_aqi.index[-1],
            'best_aqi': float(city_aqi.iloc[0]),
            'worst_aqi': float(city_aqi.iloc[-1])
        }

        # Key insights
        hourly_aqi = self.raw_data.groupby(self.raw_data['timestamp'].dt.hour)['aqi'].mean()
        report['temporal_patterns'] = {
            'peak_hour': int(hourly_aqi.idxmax()),
            'best_hour': int(hourly_aqi.idxmin()),
            'peak_hour_aqi': float(hourly_aqi.max()),
            'best_hour_aqi': float(hourly_aqi.min())
        }

        # Save report
        os.makedirs('../reports', exist_ok=True)
        report_file = f"../reports/advanced_eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Detailed report saved to {report_file}")

        # Print summary
        print("\n📊 ADVANCED EDA SUMMARY:")
        print(f"• Total Records: {report['data_summary']['total_records']}")
        print(f"• Cities Covered: {report['data_summary']['cities_covered']}")
        print(f"• Average AQI: {report['aqi_statistics']['overall_mean']:.1f}")
        print(f"• AQI Range: {report['aqi_statistics']['overall_min']:.1f} - {report['aqi_statistics']['overall_max']:.1f}")

        print(f"\n🏆 AIR QUALITY RANKINGS:")
        for i, (city, aqi) in enumerate(city_aqi.items(), 1):
            status = "🟢 Good" if aqi <= 50 else "🟡 Moderate" if aqi <= 100 else "🟠 Unhealthy" if aqi <= 150 else "🔴 Very Unhealthy"
            print(f"  {i}. {city}: {aqi:.1f} ({status})")

        print(f"\n⏰ TEMPORAL PATTERNS:")
        print(f"• Peak pollution hour: {report['temporal_patterns']['peak_hour']}:00 ({report['temporal_patterns']['peak_hour_aqi']:.1f} AQI)")
        print(f"• Cleanest hour: {report['temporal_patterns']['best_hour']}:00 ({report['temporal_patterns']['best_hour_aqi']:.1f} AQI)")

        return report

    def run_complete_advanced_eda(self):
        """Run the complete advanced EDA pipeline"""
        print("🚀 STARTING ADVANCED EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        # Generate data
        self.generate_realistic_data()

        # Perform analysis
        analysis_results = self.perform_advanced_analysis()

        # Create visualizations
        self.create_advanced_visualizations()

        # Clustering
        clusters = self.perform_clustering_analysis()

        # Generate report
        report = self.generate_comprehensive_report()

        print("\n🎉 ADVANCED EDA COMPLETED!")
        print("=" * 60)
        print("📊 Generated comprehensive insights:")
        print("• Statistical analysis across cities")
        print("• AQI distribution and category analysis")
        print("• Correlation analysis between features")
        print("• Time series patterns (hourly, daily)")
        print("• Weather-pollution relationships")
        print("• Pollutant distribution analysis")
        print("• Clustering analysis for pattern discovery")
        print("• Comprehensive quality report")
        print("\n📁 All visualizations saved to 'reports/' directory")

        return {
            'analysis': analysis_results,
            'clusters': clusters,
            'report': report
        }

# Run the analysis
if __name__ == "__main__":
    eda = AdvancedEDA()
    results = eda.run_complete_advanced_eda()

    print("\n✅ Advanced EDA analysis completed successfully!")
    print("📊 Check the 'reports/' and 'data/' directories for all outputs.")
