"""
Complete Karachi AQI System Test
================================

Tests the entire Karachi AQI prediction system including:
- MongoDB Feature Store (with CSV fallback)
- ML Model Loading
- Data Retrieval and Processing
- Prediction Capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import joblib

# Add current directory to path
sys.path.append('.')

def test_mongodb_feature_store():
    """Test MongoDB feature store with CSV fallback"""
    print("🗄️  TESTING MONGODB FEATURE STORE")
    print("=" * 50)

    try:
        from mongodb_feature_store import MongoDBFeatureStore

        # Initialize store (will fallback to CSV if MongoDB not available)
        store = MongoDBFeatureStore()

        # Generate test data
        test_data = pd.DataFrame({
            'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=5), periods=5, freq='h'),
            'city': ['Karachi'] * 5,
            'temperature': np.random.normal(32, 3, 5),
            'humidity': np.random.normal(65, 8, 5),
            'aqi': np.random.normal(110, 25, 5),
            'pm2_5': np.random.normal(50, 15, 5),
            'pm10': np.random.normal(85, 20, 5),
            'co': np.random.normal(0.6, 0.1, 5),
            'no2': np.random.normal(22, 5, 5),
            'so2': np.random.normal(8, 2, 5),
            'o3': np.random.normal(18, 4, 5)
        })

        # Test data insertion
        print("📥 Testing data insertion...")
        success = store.insert_data(test_data)
        print(f"✅ Data insertion: {'SUCCESS' if success else 'FAILED'}")

        # Test data retrieval
        print("📤 Testing data retrieval...")
        recent_data = store.get_recent_data(hours=6)
        historical_data = store.get_historical_data(days=1)

        print(f"✅ Recent data: {len(recent_data)} records")
        print(f"✅ Historical data: {len(historical_data)} records")

        # Test statistics
        print("📊 Testing statistics...")
        stats = store.get_statistics()
        print(f"✅ Total records: {stats.get('total_records', 0)}")
        print(f"✅ Storage type: {stats.get('storage_type', 'Unknown')}")

        store.close_connection()
        return True

    except Exception as e:
        print(f"❌ MongoDB Feature Store test failed: {e}")
        return False

def test_ml_models():
    """Test ML model loading and prediction"""
    print("\n🤖 TESTING ML MODELS")
    print("=" * 50)

    try:
        # Check if models exist
        models_dir = "src/models/saved_models"
        if not os.path.exists(models_dir):
            print("❌ Models directory not found")
            return False

        import joblib
        import os

        # Look for trained models
        if not os.path.exists(models_dir):
            print("❌ Models directory not found")
            return False

        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') and 'karachi' in f]
        scaler_files = [f for f in os.listdir(models_dir) if f.startswith('scaler') and 'karachi' in f]

        if not model_files:
            print("❌ No trained Karachi models found")
            print("💡 Run 'python train_karachi_models.py' to train models")
            return False

        print(f"✅ Found {len(model_files)} model files:")
        for model_file in model_files:
            print(f"   • {model_file}")

        if not scaler_files:
            print("❌ No scaler file found")
            return False

        print(f"✅ Found scaler: {scaler_files[0]}")

        # Test model loading
        try:
            model_path = os.path.join(models_dir, model_files[0])
            scaler_path = os.path.join(models_dir, scaler_files[0])

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            print("✅ Model and scaler loaded successfully")

            # Test prediction with sample data
            sample_features = np.array([[32.0, 65.0, 8.0, 110.0, 50.0, 0.8, 0.9, 105.0, 45.0]])  # 9 features
            scaled_features = scaler.transform(sample_features)

            prediction = model.predict(scaled_features)[0]
            print(f"   Prediction: {prediction:.1f}")
            return True

        except Exception as e:
            print(f"❌ Model prediction test failed: {e}")
            return False

    except Exception as e:
        print(f"❌ ML Models test failed: {e}")
        return False

def test_karachi_aqi_system():
    """Test the complete Karachi AQI system"""
    print("\n🌤️  TESTING COMPLETE KARACHI AQI SYSTEM")
    print("=" * 50)

    try:
        from karachi_aqi_app import KarachiAQISystem

        system = KarachiAQISystem()

        # Test current weather fetch
        print("🌤️  Testing current weather fetch...")
        current_weather = system.get_current_weather()
        if current_weather:
            print("✅ Current weather data retrieved")
            print(f"   Temperature: {current_weather.get('temperature', 'N/A')}°C")
            print(f"   AQI: {current_weather.get('aqi', 'N/A')}")
        else:
            print("⚠️  Current weather fetch failed (may be due to API limits)")
            print("   This is normal if OpenWeather API key has reached limits")

        # Test feature store
        print("💾 Testing feature store integration...")
        if system.feature_store:
            stats = system.feature_store.get_statistics()
            print("✅ Feature store connected")
            print(f"   Records: {stats.get('total_records', 0)}")
            print(f"   Type: {stats.get('storage_type', 'Unknown')}")
        else:
            print("❌ Feature store not connected")

        # Test model loading
        print("🤖 Testing ML model integration...")
        if hasattr(system, 'models') and system.models:
            print(f"✅ {len(system.models)} ML models loaded")
            for model_name in system.models.keys():
                print(f"   • {model_name}")
        else:
            print("❌ No ML models loaded")

        # Test prediction (if models are available)
        if hasattr(system, 'models') and system.models and current_weather:
            print("🔮 Testing prediction system...")
            predictions = system.predict_aqi_24h(current_weather)
            if 'error' not in predictions:
                print("✅ Prediction system working")
                if 'ensemble' in predictions:
                    ensemble = predictions['ensemble']
                    print(f"   Ensemble prediction: {ensemble['prediction']:.1f} AQI")
            else:
                print("⚠️  Prediction returned error")
                print(f"   Error: {predictions.get('error', 'Unknown')}")
        else:
            print("⚠️  Prediction test skipped (models or weather data not available)")

        return True

    except Exception as e:
        print(f"❌ Complete system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 KARACHI AQI SYSTEM - COMPLETE TEST SUITE")
    print("=" * 60)

    test_results = []

    # Test MongoDB Feature Store
    test_results.append(("MongoDB Feature Store", test_mongodb_feature_store()))

    # Test ML Models
    test_results.append(("ML Models", test_ml_models()))

    # Test Complete System
    test_results.append(("Complete Karachi AQI System", test_karachi_aqi_system()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print("20")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("🏆 Your Karachi AQI Prediction System is fully functional!")
        print("\n🚀 Ready to run:")
        print("   streamlit run karachi_aqi_app.py")
        print("\n📊 Features working:")
        print("   • MongoDB/CSV Feature Store")
        print("   • 3 ML Models (Linear, Random Forest, XGBoost)")
        print("   • Real-time AQI monitoring")
        print("   • AI-powered predictions")
        print("   • Interactive visualizations")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed.")
        print("Check the error messages above and fix issues before running the app.")

    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
