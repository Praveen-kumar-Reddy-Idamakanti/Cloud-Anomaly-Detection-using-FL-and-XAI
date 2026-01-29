/**
 * Frontend Integration Test Script
 * Tests the enhanced detection API endpoints from the browser console
 */

// Test configuration
const API_BASE_URL = 'http://localhost:8000';

// Test data (78 features)
const generateTestData = () => {
  return Array.from({ length: 3 }, () =>
    Array.from({ length: 78 }, () => (Math.random() * 2 - 1).toFixed(6))
  );
};

// Test functions
const testModelInfo = async () => {
  console.log('🔍 Testing model info endpoint...');
  try {
    const response = await fetch(`${API_BASE_URL}/model/info`);
    const data = await response.json();
    
    console.log('✅ Model info retrieved:');
    console.log('   - Model path:', data.model_path);
    console.log('   - Input dimensions:', data.input_dim);
    console.log('   - Two-stage enabled:', data.two_stage_enabled);
    console.log('   - Attack types:', data.attack_types);
    
    return data;
  } catch (error) {
    console.error('❌ Model info test failed:', error);
    return null;
  }
};

const testStandardDetection = async () => {
  console.log('\\n🔍 Testing standard detection endpoint...');
  try {
    const features = generateTestData();
    const payload = {
      features: features,
      threshold: 0.4
    };
    
    const response = await fetch(`${API_BASE_URL}/model/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await response.json();
    
    console.log('✅ Standard detection successful:');
    console.log('   - Predictions:', data.predictions);
    console.log('   - Scores:', data.scores.map(s => s.toFixed(4)));
    console.log('   - Threshold:', data.threshold);
    
    return data;
  } catch (error) {
    console.error('❌ Standard detection test failed:', error);
    return null;
  }
};

const testEnhancedDetection = async () => {
  console.log('\\n🔍 Testing enhanced detection endpoint...');
  try {
    const features = generateTestData();
    const payload = {
      features: features,
      threshold: 0.22610116
    };
    
    const response = await fetch(`${API_BASE_URL}/model/detect-enhanced`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const data = await response.json();
    
    console.log('✅ Enhanced detection successful:');
    console.log('   - Anomaly predictions:', data.anomaly_predictions);
    console.log('   - Reconstruction errors:', data.reconstruction_errors.map(e => e.toFixed(4)));
    console.log('   - Attack type predictions:', data.attack_type_predictions);
    console.log('   - Attack confidences:', data.attack_confidences.map(c => c.toFixed(4)));
    console.log('   - Attack types available:', data.attack_types);
    
    return data;
  } catch (error) {
    console.error('❌ Enhanced detection test failed:', error);
    return null;
  }
};

const testRealtimeStream = () => {
  console.log('\\n🔍 Testing real-time stream...');
  try {
    const eventSource = new EventSource(`${API_BASE_URL}/realtime/stream`);
    
    eventSource.onmessage = (event) => {
      console.log('✅ Real-time update received:', JSON.parse(event.data));
    };
    
    eventSource.onerror = (error) => {
      console.error('❌ Real-time stream error:', error);
    };
    
    // Close after 10 seconds
    setTimeout(() => {
      eventSource.close();
      console.log('🔌 Real-time stream test completed');
    }, 10000);
    
    console.log('✅ Real-time stream connected (testing for 10 seconds)');
    return eventSource;
  } catch (error) {
    console.error('❌ Real-time stream test failed:', error);
    return null;
  }
};

// Run all tests
const runAllTests = async () => {
  console.log('🚀 Starting Frontend Integration Tests');
  console.log('=' .repeat(50));
  
  const modelInfo = await testModelInfo();
  
  if (modelInfo && modelInfo.two_stage_enabled) {
    await testEnhancedDetection();
    await testStandardDetection();
  } else {
    await testStandardDetection();
    console.log('⚠️  Enhanced detection skipped (two-stage not enabled)');
  }
  
  testRealtimeStream();
  
  console.log('\\n' + '=' .repeat(50));
  console.log('📊 Frontend integration tests completed!');
  console.log('\\n📝 Next steps:');
  console.log('1. Open browser to http://localhost:5173/detect');
  console.log('2. Test the UI components with sample data');
  console.log('3. Verify real-time monitoring functionality');
  console.log('4. Check attack type visualization');
};

// Export for browser console use
window.testFrontendIntegration = {
  runAllTests,
  testModelInfo,
  testStandardDetection,
  testEnhancedDetection,
  testRealtimeStream,
  generateTestData
};

console.log('🧪 Frontend test functions loaded!');
console.log('Run: testFrontendIntegration.runAllTests()');
