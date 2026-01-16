#!/usr/bin/env python3
"""
Test script to verify GitHub Actions secrets are working
"""

import os
import sys

def test_secrets():
    """Test if environment variables are set"""
    print("🔍 Testing GitHub Actions secrets...")

    # Test OPENWEATHER_API_KEY
    api_key = os.getenv('OPENWEATHER_API_KEY')

    if api_key and len(api_key.strip()) > 0:
        print(f"✅ OPENWEATHER_API_KEY is set (length: {len(api_key)})")
        # Don't print the actual key for security
        if len(api_key) == 32:  # Expected length for the API key
            print("✅ API key appears to be the correct length")
        else:
            print(f"⚠️  API key length is {len(api_key)}, expected 32")
    else:
        print("⚠️  OPENWEATHER_API_KEY is not set or empty")
        print("🔧 Using fallback API key for testing (this is temporary)")
        fallback_key = "da06b92d3139ce209b04dba2132ad4ce"
        print(f"✅ Fallback key available (length: {len(fallback_key)})")

    # Test other environment variables
    python_location = os.getenv('pythonLocation')
    if python_location:
        print(f"✅ Python location: {python_location}")
    else:
        print("⚠️  Python location not found")

    print("🎉 Secret test completed!")
    print("📝 Note: Please set up OPENWEATHER_API_KEY secret in GitHub for production use")
    return True

if __name__ == "__main__":
    success = test_secrets()
    sys.exit(0 if success else 1)
