"""
Simple test for Phase 4 without the problematic integrated_plots.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

def test_phase4_simple():
    """Test Phase 4 with minimal components"""
    print("🧪 Testing Phase 4 (Simple Version)...")
    
    try:
        # Import only the integrated explainer (avoiding visualization for now)
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from integrated_explainer import IntegratedExplainer
        
        # Create simple models
        class SimpleAutoencoder(nn.Module):
            def __init__(self, input_dim=78):
                super(SimpleAutoencoder, self).__init__()
                
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 4),
                    nn.ReLU()
                )
                
                self.decoder = nn.Sequential(
                    nn.Linear(4, 16),
                    nn.ReLU(),
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, input_dim),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        class SimpleClassifier(nn.Module):
            def __init__(self, input_dim=78, num_classes=6):
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
        
        # Create test data
        np.random.seed(42)
        torch.manual_seed(42)
        
        X = np.random.randn(100, 78)
        y = np.random.randint(0, 6, 100)
        
        X = (X - X.min()) / (X.max() - X.min())
        
        # Initialize explainer
        autoencoder = SimpleAutoencoder()
        classifier = SimpleClassifier()
        attack_type_names = ['Normal', 'Anomaly', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        
        explainer = IntegratedExplainer(autoencoder, classifier, attack_type_names[:6])
        print("✅ IntegratedExplainer initialized")
        
        # Test with a sample
        sample_data = torch.FloatTensor(X[0])
        explanation = explainer.explain_two_stage_prediction(sample_data)
        print("✅ Two-stage explanation generated")
        print(f"   - Status: {explanation['unified_analysis']['overall_status']['status']}")
        print(f"   - Confidence: {explanation['unified_analysis']['confidence_analysis']['overall_confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase4_simple()
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
