"""
Minimal test for Phase 3 to isolate the issue
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def create_minimal_test():
    """Create minimal test case"""
    print("🔧 Creating minimal test...")
    
    # Create simple data
    X = np.random.randn(10, 78)
    y = np.random.randint(0, 6, 10)
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(78, 6)
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    
    # Test prediction
    X_tensor = torch.FloatTensor(X)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        confidence_scores = torch.max(probabilities, dim=1)[0]
    
    print(f"✅ Minimal test passed")
    print(f"   - Predictions shape: {predictions.shape}")
    print(f"   - Probabilities shape: {probabilities.shape}")
    print(f"   - Confidence scores shape: {confidence_scores.shape}")
    
    return True

if __name__ == "__main__":
    success = create_minimal_test()
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
