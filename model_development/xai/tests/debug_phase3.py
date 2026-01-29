"""
Debug script for Phase 3 classifier explainer
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def create_simple_classifier(input_dim=78, num_classes=6):
    """Create a simple classifier for testing"""
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(SimpleClassifier, self).__init__()
            
            self.network = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, num_classes)
            )
        
        def forward(self, x):
            return self.network(x)
    
    return SimpleClassifier(input_dim, num_classes)

def create_test_data():
    """Create simple test data"""
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 100
    n_features = 78
    n_classes = 6
    
    # Generate simple data
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Normalize data
    X = (X - X.min()) / (X.max() - X.min())
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    return X, y, X_tensor, y_tensor

def test_classifier_explainer_simple():
    """Test classifier explainer with simple data"""
    print("🔧 Testing ClassifierExplainer with simple data...")
    
    try:
        from classifier_explainer import ClassifierExplainer
        
        # Create simple data and model
        X, y, X_tensor, y_tensor = create_test_data()
        
        # Create simple model
        model = create_simple_classifier(78, 6)
        
        # Initialize explainer
        attack_type_names = ['Normal', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        explainer = ClassifierExplainer(model, attack_type_names, device='cpu')
        
        print("✅ ClassifierExplainer initialized")
        
        # Test prediction with confidence
        sample_data = X_tensor[0:1]  # Single sample
        prediction_result = explainer.predict_with_confidence(sample_data)
        print(f"✅ Predictions computed: {len(prediction_result['predictions'])} samples")
        print(f"   - Mean confidence: {np.mean(prediction_result['confidence_scores']):.3f}")
        print(f"   - Predicted classes: {prediction_result['predicted_classes']}")
        
        # Test comprehensive explanation
        print("   Testing comprehensive explanation...")
        explanation = explainer.explain_attack_type_prediction(sample_data, include_lime=False)
        print(f"✅ Comprehensive explanation generated")
        print(f"   - Predicted: {explanation['prediction_result']['predicted_classes'][0]}")
        print(f"   - Confidence: {explanation['prediction_result']['confidence_scores'][0]:.3f}")
        
        # Test explanation report
        report = explainer.generate_attack_type_explanation_report(sample_data)
        print(f"✅ Explanation report generated: {len(report)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_classifier_explainer_simple()
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
