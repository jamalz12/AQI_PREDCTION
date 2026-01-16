"""
Test script for Karachi AQI System
=================================

Tests the fixed methods to ensure they work properly.
"""

import sys
import os
sys.path.append('.')

from karachi_aqi_app import KarachiAQISystem
import pandas as pd

def test_karachi_system():
    """Test the Karachi AQI system methods"""

    print("🧪 Testing Karachi AQI System")
    print("=" * 50)

    # Initialize system
    system = KarachiAQISystem()

    # Test 1: Check if get_historical_data method exists
    print("✅ Test 1: get_historical_data method exists")
    if hasattr(system, 'get_historical_data'):
        print("   ✓ Method exists")

        # Test calling the method
        try:
            data = system.get_historical_data(days=1)
            print(f"   ✓ Method callable, returned {len(data)} records")
        except Exception as e:
            print(f"   ❌ Method failed: {e}")
    else:
        print("   ❌ Method missing")

    # Test 2: Check model loading
    print("\n✅ Test 2: Model loading")
    if hasattr(system, 'models') and system.models:
        print(f"   ✓ {len(system.models)} models loaded")
        for model_name in system.models.keys():
            print(f"     - {model_name}")
    else:
        print("   ❌ No models loaded")

    # Test 3: Check feature preparation
    print("\n✅ Test 3: Feature preparation")
    test_data = {
        'timestamp': pd.Timestamp.now(),
        'temperature': 32,
        'humidity': 70,
        'aqi': 120,
        'pm2_5': 55,
        'pm10': 85,
        'co': 0.8,
        'no2': 25,
        'o3': 30,
        'so2': 10
    }

    if hasattr(system, '_prepare_features_for_prediction'):
        try:
            features = system._prepare_features_for_prediction(test_data)
            if features is not None:
                print(f"   ✓ Feature preparation successful, shape: {features.shape}")
                print(f"     Features: {list(features.columns)}")
            else:
                print("   ❌ Feature preparation returned None")
        except Exception as e:
            print(f"   ❌ Feature preparation failed: {e}")
    else:
        print("   ❌ Feature preparation method missing")

    # Test 4: Check prediction method
    print("\n✅ Test 4: Prediction method")
    if hasattr(system, 'predict_aqi_24h'):
        try:
            predictions = system.predict_aqi_24h(test_data)
            if 'error' not in predictions:
                print("   ✓ Prediction successful")
                if 'ensemble' in predictions:
                    ensemble = predictions['ensemble']
                    print(f"     Ensemble prediction: {ensemble['prediction']:.1f} AQI")
                    print(f"     Category: {ensemble['category']}")
            else:
                print(f"   ⚠️ Prediction returned error: {predictions['error']}")
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
    else:
        print("   ❌ Prediction method missing")

    # Test 5: Check feature store
    print("\n✅ Test 5: Feature store")
    if hasattr(system, 'feature_store') and system.feature_store:
        try:
            stats = system.feature_store.get_statistics()
            print(f"   ✓ Feature store connected, {stats.get('total_records', 0)} records")
        except Exception as e:
            print(f"   ❌ Feature store error: {e}")
    else:
        print("   ❌ Feature store not available")

    print("\n🎉 Testing completed!")
    return True

if __name__ == "__main__":
    test_karachi_system()


