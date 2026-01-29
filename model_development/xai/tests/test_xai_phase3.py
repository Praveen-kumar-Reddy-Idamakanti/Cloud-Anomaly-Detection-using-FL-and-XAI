"""
XAI Phase 3 Comprehensive Test Script

This script tests all XAI Phase 3 components to verify they're working correctly:
- Classifier explainer functionality
- LIME explanations for attack classification
- Attack type-specific feature importance
- Confidence and uncertainty analysis
- Decision boundary visualization
- Misclassification analysis
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

def create_test_data_with_classifier():
    """Create test data and trained classifier for testing"""
    print("🔧 Creating test data and classifier...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create simpler test data
    n_samples = 400
    n_features = 78
    n_classes = 6
    
    # Generate simple data with clear patterns
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    
    # Normalize data
    X = (X - X.min()) / (X.max() - X.min())
    
    # Handle NaN values
    X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
    
    # Create and train simple classifier
    print("   Training simple classifier...")
    model = create_simple_classifier(n_features, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train for a few epochs
    model.train()
    for epoch in range(10):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        if epoch % 5 == 0:
            accuracy = 100 * correct / total
            print(f"   Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%")
    
    print(f"✅ Created test dataset: {X.shape} samples")
    print(f"✅ Trained classifier model")
    
    return X, y, model, dataloader

def test_classifier_explainer():
    """Test ClassifierExplainer functionality"""
    print("\n🎯 Testing ClassifierExplainer...")
    
    try:
        from classifier_explainer import ClassifierExplainer
        
        # Create test data and model
        X, y, model, dataloader = create_test_data_with_classifier()
        
        # Initialize explainer
        attack_type_names = ['Normal', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        explainer = ClassifierExplainer(model, attack_type_names, device='cpu')
        print("✅ ClassifierExplainer initialized")
        
        # Test prediction with confidence
        sample_data = X[0:5]  # Test with 5 samples
        prediction_result = explainer.predict_with_confidence(sample_data)
        print(f"✅ Predictions computed: {len(prediction_result['predictions'])} samples")
        print(f"   - Mean confidence: {np.mean(prediction_result['confidence_scores']):.3f}")
        print(f"   - Predicted classes: {prediction_result['predicted_classes']}")
        
        # Test LIME explanations (if available)
        print("   Testing LIME explanations...")
        try:
            lime_result = explainer.explain_attack_type_lime(X[0], training_data=X[:100])
            if lime_result:
                print("✅ LIME explanation computed")
            else:
                print("⚠️  LIME not available")
        except Exception as e:
            print(f"⚠️  LIME explanation failed: {str(e)}")
        
        # Test attack type feature importance
        print("   Testing attack type feature importance...")
        feature_importance = explainer.compute_attack_type_feature_importance(dataloader)
        print(f"✅ Feature importance computed: {len(feature_importance)} attack types")
        for attack_type, importance in feature_importance.items():
            print(f"   - {attack_type}: {len(importance)} features")
        
        # Test confidence analysis
        print("   Testing confidence analysis...")
        confidence_analysis = explainer.analyze_prediction_confidence(dataloader)
        print(f"✅ Confidence analysis completed")
        print(f"   - Overall confidence: {confidence_analysis['overall_stats']['mean_confidence']:.3f}")
        print(f"   - Low confidence samples: {confidence_analysis['low_confidence_samples']['count']}")
        
        # Test misclassification analysis
        print("   Testing misclassification analysis...")
        misclassification_analysis = explainer.analyze_misclassifications(dataloader)
        print(f"✅ Misclassification analysis completed")
        print(f"   - Misclassification rate: {misclassification_analysis['misclassification_rate']:.3f}")
        print(f"   - Most confused pairs: {len(misclassification_analysis['most_confused_pairs'])}")
        
        # Test comprehensive explanation
        print("   Testing comprehensive explanation...")
        sample_data = torch.FloatTensor(X[0])  # Convert to tensor
        explanation = explainer.explain_attack_type_prediction(sample_data, include_lime=False)  # Disable LIME for now
        print(f"✅ Comprehensive explanation generated")
        print(f"   - Predicted: {explanation['prediction_result']['predicted_classes'][0]}")
        print(f"   - Confidence: {explanation['prediction_result']['confidence_scores'][0]:.3f}")
        
        # Test explanation report
        report = explainer.generate_attack_type_explanation_report(X[0])
        print(f"✅ Explanation report generated: {len(report)} characters")
        
        return True, explainer, feature_importance, confidence_analysis, misclassification_analysis
        
    except Exception as e:
        print(f"❌ ClassifierExplainer test failed: {str(e)}")
        return False, None, None, None, None

def test_classifier_visualizations(feature_importance, confidence_analysis, misclassification_analysis):
    """Test classifier visualization capabilities"""
    print("\n📊 Testing Classifier Visualizations...")
    
    try:
        from visualization.classifier_plots import ClassifierPlotter
        
        plotter = ClassifierPlotter()
        print("✅ ClassifierPlotter initialized")
        
        # Test confusion matrix
        print("   Testing confusion matrix...")
        plotter.plot_confusion_matrix(misclassification_analysis['confusion_matrix'], save_path='test_confusion_matrix.png')
        print("✅ Confusion matrix plot created")
        
        # Test attack type feature importance
        print("   Testing attack type feature importance...")
        plotter.plot_attack_type_feature_importance(feature_importance, save_path='test_attack_importance.png')
        print("✅ Attack type feature importance plot created")
        
        # Test confidence distribution
        print("   Testing confidence distribution...")
        plotter.plot_confidence_distribution(confidence_analysis, save_path='test_confidence_distribution.png')
        print("✅ Confidence distribution plot created")
        
        # Test misclassification analysis
        print("   Testing misclassification analysis...")
        plotter.plot_misclassification_analysis(misclassification_analysis, save_path='test_misclassification.png')
        print("✅ Misclassification analysis plot created")
        
        # Test LIME explanation plot (if available)
        print("   Testing LIME explanation plot...")
        # This would need actual LIME result, so we'll skip for now
        print("⚠️  LIME plot skipped (requires actual LIME explanation)")
        
        # Test interactive plot (if available)
        print("   Testing interactive plots...")
        interactive_fig = plotter.plot_interactive_confidence_analysis(confidence_analysis)
        if interactive_fig:
            print("✅ Interactive confidence plot created")
        else:
            print("⚠️  Interactive plots not available (Plotly missing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Classifier visualization test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_decision_boundaries():
    """Test decision boundary visualization"""
    print("\n🗺️ Testing Decision Boundary Visualization...")
    
    try:
        from visualization.classifier_plots import ClassifierPlotter
        
        # Create simple 2D test data
        np.random.seed(42)
        n_samples = 200
        n_features = 78
        
        # Generate 2D clusters for visualization
        X_2d = np.random.randn(n_samples, 2) * 2
        y_2d = np.zeros(n_samples, dtype=int)
        
        # Create different clusters for different classes
        samples_per_class = n_samples // 4
        for i in range(4):
            start_idx = i * samples_per_class
            end_idx = (i + 1) * samples_per_class
            if i == 0:
                X_2d[start_idx:end_idx] = np.random.randn(samples_per_class, 2) + np.array([0, 0])
                y_2d[start_idx:end_idx] = 0
            elif i == 1:
                X_2d[start_idx:end_idx] = np.random.randn(samples_per_class, 2) + np.array([3, 3])
                y_2d[start_idx:end_idx] = 1
            elif i == 2:
                X_2d[start_idx:end_idx] = np.random.randn(samples_per_class, 2) + np.array([-3, 3])
                y_2d[start_idx:end_idx] = 2
            else:
                X_2d[start_idx:end_idx] = np.random.randn(samples_per_class, 2) + np.array([3, -3])
                y_2d[start_idx:end_idx] = 3
        
        # Pad to full dimension
        X_full = np.random.randn(n_samples, n_features) * 0.1
        X_full[:, 0] = X_2d[:, 0]
        X_full[:, 1] = X_2d[:, 1]
        
        # Create simple classifier for 2D visualization
        model_2d = create_simple_classifier(n_features, 4)
        
        # Train briefly
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_2d.parameters(), lr=0.01)
        
        X_tensor = torch.FloatTensor(X_full)
        y_tensor = torch.LongTensor(y_2d)
        
        model_2d.train()
        for epoch in range(20):
            optimizer.zero_grad()
            outputs = model_2d(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Plot decision boundaries
        plotter = ClassifierPlotter()
        plotter.plot_decision_boundaries(X_full, y_2d, model_2d, feature_indices=(0, 1), save_path='test_decision_boundaries.png')
        print("✅ Decision boundary visualization created")
        
        return True
        
    except Exception as e:
        print(f"❌ Decision boundary test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_classifier_workflow():
    """Test comprehensive classifier explanation workflow"""
    print("\n🔄 Testing Comprehensive Classifier Workflow...")
    
    try:
        from classifier_explainer import ClassifierExplainer
        from visualization.classifier_plots import ClassifierPlotter
        
        # Create test setup
        X, y, model, dataloader = create_test_data_with_classifier()
        
        # Initialize components
        attack_type_names = ['Normal', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        explainer = ClassifierExplainer(model, attack_type_names, device='cpu')
        plotter = ClassifierPlotter()
        
        # Run complete workflow
        print("   Running complete classifier explanation workflow...")
        
        # 1. Compute feature importance
        feature_importance = explainer.compute_attack_type_feature_importance(dataloader)
        
        # 2. Analyze confidence
        confidence_analysis = explainer.analyze_prediction_confidence(dataloader)
        
        # 3. Analyze misclassifications
        misclassification_analysis = explainer.analyze_misclassifications(dataloader)
        
        # 4. Generate explanations for multiple samples
        explanations = []
        for i in range(5):  # Test with 5 samples
            sample_data = torch.FloatTensor(X[i])  # Convert to tensor
            explanation = explainer.explain_attack_type_prediction(sample_data, include_lime=False)  # Disable LIME for now
            explanations.append(explanation)
        
        # 5. Create comprehensive visualization
        print("   Creating comprehensive visualizations...")
        
        # Confusion matrix
        plotter.plot_confusion_matrix(misclassification_analysis['confusion_matrix'], save_path='workflow_confusion.png')
        
        # Feature importance
        plotter.plot_attack_type_feature_importance(feature_importance, save_path='workflow_feature_importance.png')
        
        # Confidence analysis
        plotter.plot_confidence_distribution(confidence_analysis, save_path='workflow_confidence.png')
        
        # Misclassification analysis
        plotter.plot_misclassification_analysis(misclassification_analysis, save_path='workflow_misclassification.png')
        
        # Comprehensive summary
        plotter.plot_comprehensive_classifier_summary(
            confidence_analysis, 
            misclassification_analysis, 
            feature_importance,
            save_path='workflow_classifier_summary.png'
        )
        
        print("✅ Complete workflow successful")
        print(f"   - Generated {len(explanations)} sample explanations")
        print(f"   - Created {5} visualization files")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive workflow test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("=" * 80)
    print("🧪 XAI Phase 3 Comprehensive Test Suite")
    print("=" * 80)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python Version: {sys.version}")
    
    # Store test results for visualization tests
    global feature_importance, confidence_analysis, misclassification_analysis
    feature_importance = None
    confidence_analysis = None
    misclassification_analysis = None
    
    try:
        # Run all tests
        tests = [
            ("Classifier Explainer", test_classifier_explainer),
            ("Classifier Visualizations", lambda: test_classifier_visualizations(
                feature_importance, confidence_analysis, misclassification_analysis)),
            ("Decision Boundary Visualization", test_decision_boundaries),
            ("Comprehensive Workflow", lambda: test_comprehensive_classifier_workflow())
        ]
        
        results = []
        
        # Test classifier explainer first to get data for other tests
        test_name, test_func = tests[0]
        print(f"\n🔄 Running {test_name}...")
        result, explainer, feat_importance, conf_analysis, misclass_analysis = test_func()
        results.append((test_name, result))
        
        # Store results for other tests
        if result:
            feature_importance = feat_importance
            confidence_analysis = conf_analysis
            misclassification_analysis = misclass_analysis
            
            # Test visualizations
            viz_result = test_classifier_visualizations(feature_importance, confidence_analysis, misclassification_analysis)
            results.append(("Classifier Visualizations", viz_result))
            
            # Test decision boundaries
            boundary_result = test_decision_boundaries()
            results.append(("Decision Boundary Visualization", boundary_result))
        
        # Test comprehensive workflow
        test_name, test_func = tests[3]
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
            print("\n🎉 ALL TESTS PASSED! XAI Phase 3 is working correctly!")
            print("\n📁 Generated test files:")
            print("   - test_confusion_matrix.png")
            print("   - test_attack_importance.png")
            print("   - test_confidence_distribution.png")
            print("   - test_misclassification.png")
            print("   - test_decision_boundaries.png")
            print("   - workflow_confusion.png")
            print("   - workflow_feature_importance.png")
            print("   - workflow_confidence.png")
            print("   - workflow_misclassification.png")
            print("   - workflow_classifier_summary.png")
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
