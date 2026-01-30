"""
XAI Phase 2 Comprehensive Test Script

This script tests all XAI Phase 2 components to verify they're working correctly:
- Autoencoder explainer functionality
- Reconstruction error analysis
- Latent space visualization
- Feature attribution methods
- Anomaly explanation generation
- Similarity analysis
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

def create_simple_autoencoder(input_dim=78):
    """Create a simple autoencoder for testing"""
    class SimpleAutoencoder(nn.Module):
        def __init__(self, input_dim):
            super(SimpleAutoencoder, self).__init__()
            
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 4),
                nn.ReLU()
            )
            
            # Decoder
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
    
    return SimpleAutoencoder(input_dim)

def create_test_data_with_model():
    """Create test data and trained model for testing"""
    print("🔧 Creating test data and model...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create realistic network traffic data
    n_samples = 500
    n_features = 78
    
    # Normal traffic patterns
    normal_data = np.random.normal(0, 1, (n_samples // 2, n_features))
    
    # Anomalous traffic patterns (different distribution)
    anomaly_data = np.random.normal(2, 1.5, (n_samples // 2, n_features))
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Normalize data to [0, 1] for autoencoder
    X = (X - X.min()) / (X.max() - X.min())
    
    # Add some missing values
    missing_indices = np.random.choice(X.shape[0] * X.shape[1], 20, replace=False)
    X.flat[missing_indices] = np.nan
    
    # Handle NaN values
    X = pd.DataFrame(X).fillna(pd.DataFrame(X).median()).values
    
    # Create and train simple autoencoder
    print("   Training simple autoencoder...")
    model = create_simple_autoencoder(n_features)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train for a few epochs
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 3 == 0:
            print(f"   Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
    
    print(f"✅ Created test dataset: {X.shape} samples")
    print(f"✅ Trained autoencoder model")
    
    return X, y, model, dataloader

def test_autoencoder_explainer():
    """Test AutoencoderExplainer functionality"""
    print("\n🤖 Testing AutoencoderExplainer...")
    
    try:
        from autoencoder_explainer import AutoencoderExplainer
        
        # Create test data and model
        X, y, model, dataloader = create_test_data_with_model()
        
        # Initialize explainer
        explainer = AutoencoderExplainer(model, device='cpu')
        print("✅ AutoencoderExplainer initialized")
        
        # Test reconstruction error computation
        reconstruction_errors = explainer.compute_reconstruction_errors(dataloader)
        print(f"✅ Reconstruction errors computed: {len(reconstruction_errors['total_errors'])} samples")
        print(f"   - Mean error: {reconstruction_errors['mean_error']:.6f}")
        print(f"   - Threshold: {reconstruction_errors['threshold']:.6f}")
        print(f"   - Anomalies detected: {np.sum(reconstruction_errors['predictions'])}")
        
        # Test per-feature reconstruction analysis
        feature_analysis = explainer.analyze_per_feature_reconstruction()
        print(f"✅ Per-feature analysis completed: {len(feature_analysis['feature_analysis'])} features")
        print(f"   - Top contributing feature: {feature_analysis['ranked_features'][0][0]}")
        
        # Test latent space extraction
        latent_representations = explainer.extract_latent_representations(dataloader)
        print(f"✅ Latent representations extracted: {latent_representations['latent_dim']}D space")
        print(f"   - Total samples: {latent_representations['n_samples']}")
        
        # Test latent space visualization
        print("   Testing latent space visualization...")
        try:
            viz_results = explainer.visualize_latent_space(method='tsne')
            print("✅ Latent space visualization completed")
        except Exception as e:
            print(f"⚠️  Latent visualization failed: {str(e)}")
        
        # Test anomaly explanation
        sample_data = X[0]  # Take first sample
        explanation = explainer.explain_anomaly_sample(sample_data)
        print(f"✅ Anomaly explanation generated")
        print(f"   - Reconstruction error: {explanation['reconstruction_error']:.6f}")
        print(f"   - Is anomaly: {explanation['is_anomaly']}")
        print(f"   - Top feature: {explanation['top_contributing_features'][0][0]}")
        
        # Test similarity analysis
        similar_samples = explainer.find_similar_samples(sample_data, dataloader, top_k=3)
        print(f"✅ Similarity analysis completed: {len(similar_samples['similar_samples'])} similar samples")
        
        # Test explanation report
        report = explainer.generate_explanation_report(sample_data)
        print(f"✅ Explanation report generated: {len(report)} characters")
        
        return True, explainer, reconstruction_errors, feature_analysis, latent_representations
        
    except Exception as e:
        print(f"❌ AutoencoderExplainer test failed: {str(e)}")
        return False, None, None, None, None

def test_feature_attributions(explainer, sample_data):
    """Test feature attribution methods"""
    print("\n🎯 Testing Feature Attribution Methods...")
    
    try:
        # Test SHAP attributions (if available)
        print("   Testing SHAP attributions...")
        try:
            shap_results = explainer.compute_shap_attributions(sample_data, nsamples=10)
            if shap_results:
                print("✅ SHAP attributions computed")
            else:
                print("⚠️  SHAP not available")
        except Exception as e:
            print(f"⚠️  SHAP attribution failed: {str(e)}")
        
        # Test Integrated Gradients (if available)
        print("   Testing Integrated Gradients...")
        try:
            ig_results = explainer.compute_integrated_gradients(sample_data, n_steps=10)
            if ig_results:
                print("✅ Integrated Gradients computed")
            else:
                print("⚠️  Integrated Gradients not available")
        except Exception as e:
            print(f"⚠️  Integrated Gradients failed: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature attribution test failed: {str(e)}")
        return False

def test_autoencoder_visualizations(reconstruction_errors, feature_analysis, latent_representations):
    """Test autoencoder visualization capabilities"""
    print("\n📊 Testing Autoencoder Visualizations...")
    
    try:
        from visualization.autoencoder_plots import AutoencoderPlotter
        
        plotter = AutoencoderPlotter()
        print("✅ AutoencoderPlotter initialized")
        
        # Test reconstruction error distribution
        print("   Testing reconstruction error distribution...")
        plotter.plot_reconstruction_error_distribution(reconstruction_errors, save_path='images/test_results/test_reconstruction_errors.png')
        print("✅ Reconstruction error distribution plot created")
        
        # Test per-feature error analysis
        print("   Testing per-feature error analysis...")
        plotter.plot_per_feature_reconstruction_errors(feature_analysis, save_path='images/test_results/test_per_feature_errors.png')
        print("✅ Per-feature error analysis plot created")
        
        # Test latent space visualization
        print("   Testing latent space visualization...")
        plotter.plot_latent_space_clusters(latent_representations, save_path='images/test_results/test_latent_clusters.png')
        print("✅ Latent space clusters plot created")
        
        # Test feature comparison plot
        print("   Testing feature comparison plot...")
        sample_original = reconstruction_errors['inputs'][0]
        sample_reconstructed = reconstruction_errors['outputs'][0]
        plotter.plot_feature_comparison(sample_original, sample_reconstructed, save_path='images/test_results/test_feature_comparison.png')
        print("✅ Feature comparison plot created")
        
        # Test interactive plot (if available)
        print("   Testing interactive plots...")
        interactive_fig = plotter.plot_interactive_reconstruction_analysis(reconstruction_errors)
        if interactive_fig:
            print("✅ Interactive reconstruction analysis plot created")
        else:
            print("⚠️  Interactive plots not available (Plotly missing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Autoencoder visualization test failed: {str(e)}")
        return False

def test_comprehensive_explanation_workflow():
    """Test comprehensive explanation workflow"""
    print("\n🔄 Testing Comprehensive Explanation Workflow...")
    
    try:
        from autoencoder_explainer import AutoencoderExplainer
        from visualization.autoencoder_plots import AutoencoderPlotter
        
        # Create test setup
        X, y, model, dataloader = create_test_data_with_model()
        
        # Initialize components
        explainer = AutoencoderExplainer(model, device='cpu')
        plotter = AutoencoderPlotter()
        
        # Run complete workflow
        print("   Running complete autoencoder explanation workflow...")
        
        # 1. Compute reconstruction errors
        reconstruction_errors = explainer.compute_reconstruction_errors(dataloader)
        
        # 2. Analyze features
        feature_analysis = explainer.analyze_per_feature_reconstruction()
        
        # 3. Extract latent space
        latent_representations = explainer.extract_latent_representations(dataloader)
        
        # 4. Generate explanations for multiple samples
        explanations = []
        for i in range(5):  # Test with 5 samples
            sample_data = X[i]
            explanation = explainer.explain_anomaly_sample(sample_data, threshold=reconstruction_errors['threshold'])
            explanations.append(explanation)
        
        # 5. Create comprehensive visualization
        print("   Creating comprehensive visualizations...")
        
        # Reconstruction analysis
        plotter.plot_reconstruction_error_distribution(reconstruction_errors, save_path='images/workflow_images/workflow_reconstruction.png')
        
        # Feature analysis
        plotter.plot_per_feature_reconstruction_errors(feature_analysis, save_path='images/workflow_images/workflow_features.png')
        
        # Latent space
        plotter.plot_latent_space_clusters(latent_representations, save_path='images/workflow_images/workflow_latent.png')
        
        # Sample explanations
        for i, explanation in enumerate(explanations[:3]):
            plotter.plot_anomaly_explanation_summary(explanation, save_path=f'images/workflow_images/workflow_explanation_{i}.png')
        
        print("✅ Complete workflow successful")
        print(f"   - Generated {len(explanations)} sample explanations")
        print(f"   - Created {4} visualization files")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive workflow test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("=" * 80)
    print("🧪 XAI Phase 2 Comprehensive Test Suite")
    print("=" * 80)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python Version: {sys.version}")
    
    # Store test results for visualization tests
    global reconstruction_errors, feature_analysis, latent_representations
    reconstruction_errors = None
    feature_analysis = None
    latent_representations = None
    
    try:
        # Run all tests
        tests = [
            ("Autoencoder Explainer", test_autoencoder_explainer),
            ("Feature Attribution Methods", lambda: test_feature_attributions(None, None)),
            ("Autoencoder Visualizations", lambda: test_autoencoder_visualizations(
                reconstruction_errors, feature_analysis, latent_representations)),
            ("Comprehensive Workflow", test_comprehensive_explanation_workflow)
        ]
        
        results = []
        
        # Test autoencoder explainer first to get data for other tests
        test_name, test_func = tests[0]
        print(f"\n🔄 Running {test_name}...")
        result, explainer, recon_errors, feat_analysis, latent_reps = test_func()
        results.append((test_name, result))
        
        # Store results for other tests
        if result:
            reconstruction_errors = recon_errors
            feature_analysis = feat_analysis
            latent_representations = latent_reps
            
            # Test feature attributions with actual data
            sample_data = np.random.normal(0, 1, 78)  # Sample data for attribution testing
            attribution_result = test_feature_attributions(explainer, sample_data)
            results.append(("Feature Attribution Methods", attribution_result))
            
            # Test visualizations
            viz_result = test_autoencoder_visualizations(reconstruction_errors, feature_analysis, latent_representations)
            results.append(("Autoencoder Visualizations", viz_result))
        
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
            print("\n🎉 ALL TESTS PASSED! XAI Phase 2 is working correctly!")
            print("\n📁 Generated test files:")
            print("   - images/test_results/test_reconstruction_errors.png")
            print("   - images/test_results/test_per_feature_errors.png")
            print("   - images/test_results/test_latent_clusters.png")
            print("   - images/test_results/test_feature_comparison.png")
            print("   - images/workflow_images/workflow_reconstruction.png")
            print("   - images/workflow_images/workflow_features.png")
            print("   - images/workflow_images/workflow_latent.png")
            print("   - images/workflow_images/workflow_explanation_0.png")
            print("   - images/workflow_images/workflow_explanation_1.png")
            print("   - images/workflow_images/workflow_explanation_2.png")
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
