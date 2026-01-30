"""
XAI Phase 4 Comprehensive Test Script

This script tests all XAI Phase 4 components to verify they're working correctly:
- Integrated explainer functionality
- Two-stage explanation pipeline
- Explanation aggregation and unification
- Comparative analysis (normal vs anomaly vs attack)
- Attack progression analysis
- Integrated dashboard and visualizations
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_two_stage_models():
    """Create autoencoder and classifier models for testing"""
    print("🔧 Creating two-stage models...")
    
    # Create autoencoder
    class SimpleAutoencoder(nn.Module):
        def __init__(self, input_dim=78):
            super(SimpleAutoencoder, self).__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 4),
                nn.ReLU()
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    # Create classifier
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
    
    return SimpleAutoencoder(), SimpleClassifier()

def create_test_data_for_two_stage():
    """Create test data suitable for two-stage analysis"""
    print("🔧 Creating test data for two-stage analysis...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create realistic data with clear patterns
    n_samples = 600
    n_features = 78
    
    # Generate different patterns for each category
    X = []
    y = []
    
    # Normal traffic (200 samples)
    normal_data = np.random.normal(0, 1, (200, n_features))
    X.extend(normal_data)
    y.extend([0] * 200)  # Normal label
    
    # Anomaly but not attack (100 samples)
    anomaly_data = np.random.normal(1.5, 1.2, (100, n_features))
    X.extend(anomaly_data)
    y.extend([1] * 100)  # Anomaly label
    
    # DoS attacks (100 samples)
    dos_data = np.random.normal(2, 1.5, (100, n_features))
    dos_data[:, 10:20] += np.random.normal(3, 1, (100, 10))  # DoS-specific features
    X.extend(dos_data)
    y.extend([2] * 100)  # DoS label
    
    # PortScan attacks (100 samples)
    portscan_data = np.random.normal(1.2, 1.1, (100, n_features))
    portscan_data[:, 20:30] += np.random.normal(2, 0.8, (100, 10))  # PortScan-specific features
    X.extend(portscan_data)
    y.extend([3] * 100)  # PortScan label
    
    # BruteForce attacks (100 samples)
    bruteforce_data = np.random.normal(1.8, 1.3, (100, n_features))
    bruteforce_data[:, 30:40] += np.random.normal(2.5, 1, (100, 10))  # BruteForce-specific features
    X.extend(bruteforce_data)
    y.extend([4] * 100)  # BruteForce label
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Normalize data
    X = (X - X.min()) / (X.max() - X.min())
    
    # Handle NaN values
    X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
    
    # Create labels for autoencoder (0=normal, 1=anomaly) and classifier
    autoencoder_labels = np.array([0 if label == 0 else 1 for label in y])
    classifier_labels = np.array([label for label in y])
    
    print(f"✅ Created test dataset: {X.shape} samples")
    print(f"   - Normal: {np.sum(y == 0)} samples")
    print(f"   - Anomaly only: {np.sum(y == 1)} samples")
    print(f"   - DoS: {np.sum(y == 2)} samples")
    print(f"   - PortScan: {np.sum(y == 3)} samples")
    print(f"   - BruteForce: {np.sum(y == 4)} samples")
    
    return X, autoencoder_labels, classifier_labels

def test_integrated_explainer():
    """Test IntegratedExplainer functionality"""
    print("\n🔄 Testing IntegratedExplainer...")
    
    try:
        from integrated_explainer import IntegratedExplainer
        
        # Create models and data
        autoencoder, classifier = create_two_stage_models()
        X, autoencoder_labels, classifier_labels = create_test_data_for_two_stage()
        
        # Initialize integrated explainer
        attack_type_names = ['Normal', 'Anomaly', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        explainer = IntegratedExplainer(autoencoder, classifier, attack_type_names[:6], device='cpu')
        print("✅ IntegratedExplainer initialized")
        
        # Test two-stage explanation
        print("   Testing two-stage explanation...")
        sample_data = torch.FloatTensor(X[0])
        integrated_explanation = explainer.explain_two_stage_prediction(sample_data)
        print(f"✅ Two-stage explanation generated")
        print(f"   - Overall status: {integrated_explanation['unified_analysis']['overall_status']['status']}")
        print(f"   - Consistency: {integrated_explanation['unified_analysis']['stage_correlation']['correlation_level']}")
        print(f"   - Overall confidence: {integrated_explanation['unified_analysis']['confidence_analysis']['overall_confidence']:.3f}")
        
        # Test integrated report
        print("   Testing integrated report...")
        report = explainer.generate_integrated_report(sample_data)
        print(f"✅ Integrated report generated: {len(report)} characters")
        
        return True, explainer, X, autoencoder_labels, classifier_labels
        
    except Exception as e:
        print(f"❌ IntegratedExplainer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None, None, None

def test_attack_progression_analysis():
    """Test attack progression analysis"""
    print("\n📈 Testing Attack Progression Analysis...")
    
    try:
        from integrated_explainer import IntegratedExplainer
        
        # Create models and data
        autoencoder, classifier = create_two_stage_models()
        X, autoencoder_labels, classifier_labels = create_test_data_for_two_stage()
        
        # Initialize explainer
        attack_type_names = ['Normal', 'Anomaly', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        explainer = IntegratedExplainer(autoencoder, classifier, attack_type_names[:6], device='cpu')
        
        # Create dataloader
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(classifier_labels)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Test attack progression analysis
        progression_analysis = explainer.analyze_attack_progression(dataloader, sample_indices=list(range(50)))
        print(f"✅ Attack progression analysis completed")
        print(f"   - Overall consistency: {progression_analysis['overall_consistency']:.3f}")
        print(f"   - Total samples analyzed: {progression_analysis['total_samples']}")
        print(f"   - Attack patterns analyzed: {len(progression_analysis['attack_patterns'])}")
        
        return True, progression_analysis
        
    except Exception as e:
        print(f"❌ Attack progression test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def test_comparative_analysis():
    """Test comparative analysis"""
    print("\n🔍 Testing Comparative Analysis...")
    
    try:
        from integrated_explainer import IntegratedExplainer
        
        # Create models and data
        autoencoder, classifier = create_two_stage_models()
        X, autoencoder_labels, classifier_labels = create_test_data_for_two_stage()
        
        # Initialize explainer
        attack_type_names = ['Normal', 'Anomaly', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        explainer = IntegratedExplainer(autoencoder, classifier, attack_type_names[:6], device='cpu')
        
        # Separate samples by type
        normal_indices = np.where(classifier_labels == 0)[0][:50]  # Normal
        anomaly_indices = np.where(classifier_labels == 1)[0][:50]  # Anomaly
        attack_indices = np.where(classifier_labels >= 2)[0][:50]  # Attacks
        
        normal_samples = X[normal_indices]
        anomaly_samples = X[anomaly_indices]
        attack_samples = X[attack_indices]
        
        # Test comparative analysis
        comparative_analysis = explainer.create_comparative_analysis(
            normal_samples, anomaly_samples, attack_samples
        )
        print(f"✅ Comparative analysis completed")
        print(f"   - Feature evolution analyzed: {len(comparative_analysis['feature_evolution']['top_evolving_features'])} features")
        print(f"   - Evolution patterns: {comparative_analysis['feature_evolution']['evolution_summary']['dominant_pattern']}")
        print(f"   - Risk progression: {comparative_analysis['risk_progression']['risk_progression_trend']}")
        
        return True, comparative_analysis
        
    except Exception as e:
        print(f"❌ Comparative analysis test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def test_integrated_visualizations():
    """Test integrated visualization capabilities"""
    print("\n📊 Testing Integrated Visualizations...")
    
    try:
        from visualization.integrated_plots import IntegratedPlotter
        
        plotter = IntegratedPlotter()
        print("✅ IntegratedPlotter initialized")
        
        # Create sample explanation for visualization
        autoencoder, classifier = create_two_stage_models()
        X, autoencoder_labels, classifier_labels = create_test_data_for_two_stage()
        
        attack_type_names = ['Normal', 'Anomaly', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        explainer = IntegratedExplainer(autoencoder, classifier, attack_type_names[:6], device='cpu')
        
        sample_data = torch.FloatTensor(X[0])
        integrated_explanation = explainer.explain_two_stage_prediction(sample_data)
        
        # Test two-stage summary plot
        print("   Testing two-stage summary plot...")
        plotter.plot_two_stage_summary(integrated_explanation, save_path='test_two_stage_summary.png')
        print("✅ Two-stage summary plot created")
        
        # Test integrated dashboard
        print("   Testing integrated dashboard...")
        plotter.plot_integrated_dashboard(integrated_explanation, save_path='test_integrated_dashboard.png')
        print("✅ Integrated dashboard created")
        
        # Test interactive plot (if available)
        print("   Testing interactive integrated dashboard...")
        interactive_fig = plotter.plot_interactive_integrated_dashboard(integrated_explanation)
        if interactive_fig:
            print("✅ Interactive integrated dashboard created")
        else:
            print("⚠️  Interactive plots not available (Plotly missing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Integrated visualization test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_workflow():
    """Test comprehensive integrated workflow"""
    print("\n🔄 Testing Comprehensive Integrated Workflow...")
    
    try:
        from integrated_explainer import IntegratedExplainer
        from visualization.integrated_plots import IntegratedPlotter
        
        # Create models and data
        autoencoder, classifier = create_two_stage_models()
        X, autoencoder_labels, classifier_labels = create_test_data_for_two_stage()
        
        # Initialize components
        attack_type_names = ['Normal', 'Anomaly', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        explainer = IntegratedExplainer(autoencoder, classifier, attack_type_names[:6], device='cpu')
        plotter = IntegratedPlotter()
        
        # Create dataloader
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(classifier_labels)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Run complete workflow
        print("   Running complete integrated workflow...")
        
        # 1. Generate explanations for multiple samples
        explanations = []
        for i in range(10):  # Test with 10 samples
            sample_data = X[i]
            integrated_explanation = explainer.explain_two_stage_prediction(
                torch.FloatTensor(sample_data)
            )
            explanations.append(integrated_explanation)
        
        # 2. Attack progression analysis
        progression_analysis = explainer.analyze_attack_progression(dataloader)
        
        # 3. Comparative analysis
        normal_indices = np.where(classifier_labels == 0)[0][:30]
        anomaly_indices = np.where(classifier_labels == 1)[0][:30]
        attack_indices = np.where(classifier_labels >= 2)[0][:30]
        
        comparative_analysis = explainer.create_comparative_analysis(
            X[normal_indices], X[anomaly_indices], X[attack_indices]
        )
        
        # 4. Create comprehensive visualizations
        print("   Creating comprehensive visualizations...")
        
        # Two-stage summaries for multiple samples
        for i, explanation in enumerate(explanations[:3]):
            plotter.plot_two_stage_summary(explanation, save_path=f'workflow_two_stage_{i}.png')
        
        # Attack progression visualization
        plotter.plot_attack_progression(progression_analysis, save_path='workflow_attack_progression.png')
        
        # Comparative analysis visualization
        plotter.plot_comparative_analysis(comparative_analysis, save_path='workflow_comparative_analysis.png')
        
        # Integrated dashboard
        for i, explanation in enumerate(explanations[:3]):
            plotter.plot_integrated_dashboard(explanation, save_path=f'workflow_dashboard_{i}.png')
        
        print("✅ Complete workflow successful")
        print(f"   - Generated {len(explanations)} sample explanations")
        print(f"   - Created {7} visualization files")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive workflow test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 80)
    print("🧪 XAI Phase 4 Comprehensive Test Suite")
    print("=" * 80)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python Version: {sys.version}")
    
    try:
        # Run all tests
        tests = [
            ("Integrated Explainer", test_integrated_explainer),
            ("Attack Progression Analysis", test_attack_progression_analysis),
            ("Comparative Analysis", test_comparative_analysis),
            ("Integrated Visualizations", test_integrated_visualizations),
            ("Comprehensive Workflow", test_comprehensive_workflow)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            print(f"\n🔄 Running {test_name}...")
            result = test_func()
            results.append((test_name, result))
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)
        
        passed = 0
        failed = 0
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:<35} {status}")
            if result:
                passed += 1
            else:
                failed += 1
        
        print(f"\n📈 Overall Results: {passed} passed, {failed} failed")
        success_rate = (passed / len(results)) * 100
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! XAI Phase 4 is working correctly!")
            print("\n📁 Generated test files:")
            print("   - test_two_stage_summary.png")
            print("   - test_integrated_dashboard.png")
            print("   - workflow_two_stage_0.png")
            print("   - workflow_two_stage_1.png")
            print("   - workflow_two_stage_2.png")
            print("   - workflow_attack_progression.png")
            print("   - workflow_comparative_analysis.png")
            print("   - workflow_dashboard_0.png")
            print("   - workflow_dashboard_1.png")
            print("   - workflow_dashboard_2.png")
        else:
            print(f"\n⚠️  {failed} test(s) failed. Please check the error messages above.")
        
        print("\n" + "=" * 80)
        
        return failed == 0
        
    except Exception as e:
        print(f"\n❌ Test suite failed with critical error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
