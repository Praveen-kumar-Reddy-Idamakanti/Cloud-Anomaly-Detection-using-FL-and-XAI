"""
Test only the explanation report generation
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def test_explanation_report_only():
    """Test only the explanation report generation"""
    print("🔧 Testing explanation report generation...")
    
    try:
        from classifier_explainer import ClassifierExplainer
        
        # Create simple data and model
        X = np.random.randn(10, 78)
        y = np.random.randint(0, 6, 10)
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(78, 6)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # Initialize explainer
        attack_type_names = ['Normal', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        explainer = ClassifierExplainer(model, attack_type_names, device='cpu')
        
        # Test only the report generation
        print("   Testing explanation report generation...")
        sample_data = torch.FloatTensor(X[0])
        report = explainer.generate_attack_type_explanation_report(sample_data)
        print(f"✅ Explanation report generated: {len(report)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_explanation_report_only()
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
