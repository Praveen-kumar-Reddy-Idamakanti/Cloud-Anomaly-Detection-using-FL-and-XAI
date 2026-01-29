"""
Test script for Phase 1 Backend-Frontend Integration
Tests the enhanced two-stage prediction API endpoints
"""

import requests
import json
import numpy as np
import time

# API Configuration
API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n🔍 Testing model info...")
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved")
            print(f"   Model: {data.get('model_path', 'unknown')}")
            print(f"   Input dimensions: {data.get('input_dim', 'unknown')}")
            print(f"   Two-stage enabled: {data.get('two_stage_enabled', False)}")
            print(f"   Attack types: {data.get('attack_types', [])}")
            return True
        elif response.status_code == 404:
            print("⚠️  No model loaded (expected in some cases)")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_standard_detection():
    """Test standard anomaly detection endpoint"""
    print("\n🔍 Testing standard anomaly detection...")
    try:
        # Generate test data (78 features for the loaded model)
        test_features = np.random.random((5, 78)).tolist()
        
        payload = {
            "features": test_features,
            "threshold": 0.4
        }
        
        response = requests.post(
            f"{API_BASE_URL}/model/detect",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Standard detection successful")
            print(f"   Predictions: {data.get('predictions', [])}")
            print(f"   Scores: {[round(s, 4) for s in data.get('scores', [])]}")
            return True
        else:
            print(f"❌ Standard detection failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Standard detection error: {e}")
        return False

def test_enhanced_detection():
    """Test enhanced two-stage anomaly detection endpoint"""
    print("\n🔍 Testing enhanced two-stage anomaly detection...")
    try:
        # Generate test data (78 features for enhanced detection)
        test_features = np.random.random((3, 78)).tolist()
        
        payload = {
            "features": test_features,
            "threshold": 0.22610116
        }
        
        response = requests.post(
            f"{API_BASE_URL}/model/detect-enhanced",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Enhanced detection successful")
            print(f"   Anomaly predictions: {data.get('anomaly_predictions', [])}")
            print(f"   Reconstruction errors: {[round(e, 4) for e in data.get('reconstruction_errors', [])]}")
            print(f"   Attack type predictions: {data.get('attack_type_predictions', [])}")
            print(f"   Attack confidences: {[round(c, 4) for c in data.get('attack_confidences', [])]}")
            print(f"   Attack types available: {data.get('attack_types', [])}")
            return True
        else:
            print(f"❌ Enhanced detection failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Enhanced detection error: {e}")
        return False

def test_stats_endpoint():
    """Test the stats endpoint"""
    print("\n🔍 Testing stats endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            print("✅ Stats retrieved")
            print(f"   Total logs: {data.get('total_logs', 0)}")
            print(f"   Total anomalies: {data.get('total_anomalies', 0)}")
            print(f"   Alert rate: {data.get('alert_rate', 0)}%")
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🚀 Starting Phase 1 Backend Integration Tests")
    print("=" * 50)
    
    # Wait a moment for server to be ready
    time.sleep(2)
    
    tests = [
        test_health_check,
        test_model_info,
        test_standard_detection,
        test_enhanced_detection,
        test_stats_endpoint
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 1 integration successful.")
    else:
        print("⚠️  Some tests failed. Check the server and model availability.")
    
    print("\n📝 Next steps:")
    print("1. Start the backend server: python -m uvicorn main:app --reload")
    print("2. Run this test script: python test_integration.py")
    print("3. Check frontend integration in browser")

if __name__ == "__main__":
    main()
