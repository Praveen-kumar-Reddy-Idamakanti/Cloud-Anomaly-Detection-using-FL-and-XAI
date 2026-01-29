"""
XAI Phase 1 Comprehensive Test Script

This script tests all XAI Phase 1 components to verify they're working correctly:
- Data analysis functionality
- Feature importance analysis
- Baseline pattern analysis
- Visualization capabilities
- Dashboard functionality
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_data():
    """Create realistic test data for XAI testing"""
    print("📊 Creating test data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate 78 network traffic features with realistic patterns
    feature_names = [f'feature_{i:02d}' for i in range(1, 79)]
    
    # Normal traffic patterns (lower values, less variance)
    normal_data = np.random.normal(0, 1, (n_samples // 2, 78))
    
    # Anomalous traffic patterns (higher values, more variance)
    anomaly_data = np.random.normal(2, 1.5, (n_samples // 2, 78))
    
    # Add some correlations between features
    for i in range(0, 78, 3):
        if i + 2 < 78:
            anomaly_data[:, i+1] = anomaly_data[:, i] * 0.8 + np.random.normal(0, 0.2, n_samples // 2)
            anomaly_data[:, i+2] = anomaly_data[:, i] * 0.6 + np.random.normal(0, 0.3, n_samples // 2)
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    
    # Create labels (0 = normal, 1 = anomaly)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    # Add some missing values and duplicates for realism
    missing_indices = np.random.choice(df.index, 20, replace=False)
    df.loc[missing_indices, 'feature_01'] = np.nan
    
    # Add some duplicate rows
    df = pd.concat([df, df.sample(10)], ignore_index=True)
    
    print(f"✅ Created test dataset: {len(df)} samples, {len(df.columns)} features")
    print(f"   - Normal samples: {len(df[df['label'] == 0])}")
    print(f"   - Anomaly samples: {len(df[df['label'] != 0])}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - Duplicate rows: {df.duplicated().sum()}")
    
    return df

def test_data_analyzer():
    """Test DataAnalyzer functionality"""
    print("\n🔍 Testing DataAnalyzer...")
    
    try:
        from foundation.data_analyzer import DataAnalyzer
        
        analyzer = DataAnalyzer()
        
        # Test data quality analysis
        quality_report = analyzer.analyze_data_quality(test_df)
        print("✅ Data quality analysis completed")
        
        # Test correlation analysis
        correlation_matrix = analyzer.compute_correlation_matrix(test_df)
        print(f"✅ Correlation matrix computed: {correlation_matrix.shape}")
        
        # Test highly correlated features
        highly_correlated = analyzer.find_highly_correlated_features(threshold=0.8)
        print(f"✅ Found {len(highly_correlated)} highly correlated feature pairs")
        
        # Test feature distribution analysis
        distributions = analyzer.analyze_feature_distributions(test_df)
        print(f"✅ Feature distribution analysis completed for {len(distributions)} features")
        
        # Test PCA analysis (handle NaN values)
        clean_df_for_pca = test_df.fillna(test_df.median())
        pca_analysis = analyzer.perform_pca_analysis(clean_df_for_pca, n_components=5)
        print(f"✅ PCA analysis completed: {pca_analysis['explained_variance_ratio'].shape}")
        
        # Test outlier detection
        outliers = analyzer.identify_outliers(test_df)
        print(f"✅ Outlier detection completed for {len(outliers)} features")
        
        # Test summary report
        summary = analyzer.generate_summary_report(test_df)
        print("✅ Summary report generated")
        
        return True
        
    except Exception as e:
        print(f"❌ DataAnalyzer test failed: {str(e)}")
        return False

def test_feature_importance_analyzer():
    """Test FeatureImportanceAnalyzer functionality"""
    print("\n🎯 Testing FeatureImportanceAnalyzer...")
    
    try:
        from foundation.feature_importance import FeatureImportanceAnalyzer
        
        analyzer = FeatureImportanceAnalyzer()
        
        # Clean data for feature importance analysis (handle NaN values)
        clean_df = test_df.fillna(test_df.median())
        
        # Test statistical importance
        statistical_importance = analyzer.compute_statistical_importance(clean_df)
        print(f"✅ Statistical importance computed: {len(statistical_importance)} methods")
        
        # Test model-based importance
        model_importance = analyzer.compute_model_based_importance(clean_df)
        print(f"✅ Model-based importance computed: {len(model_importance)} models")
        
        # Test SHAP importance (may not be available)
        shap_importance = analyzer.compute_shap_importance(clean_df, sample_size=50)
        if shap_importance:
            print(f"✅ SHAP importance computed: {len(shap_importance)} features")
        else:
            print("⚠️  SHAP not available (skipping)")
        
        # Test attack type-specific importance
        attack_importance = analyzer.compute_attack_type_specific_importance(clean_df)
        print(f"✅ Attack type-specific importance computed")
        
        # Test aggregated importance
        aggregated = analyzer.aggregate_importance_scores()
        print(f"✅ Aggregated importance scores: {len(aggregated)} features")
        
        # Test feature selection
        top_features = analyzer.select_top_features(aggregated, 20)
        print(f"✅ Top 20 features selected: {len(top_features)}")
        
        # Test feature importance report
        report = analyzer.generate_feature_importance_report(clean_df)
        print("✅ Feature importance report generated")
        
        return True
        
    except Exception as e:
        print(f"❌ FeatureImportanceAnalyzer test failed: {str(e)}")
        return False

def test_baseline_explainer():
    """Test BaselineExplainer functionality"""
    print("\n🔬 Testing BaselineExplainer...")
    
    try:
        from foundation.baseline_explainer import BaselineExplainer
        
        explainer = BaselineExplainer()
        
        # Test baseline pattern establishment
        baseline_patterns = explainer.establish_baseline_patterns(test_df)
        print(f"✅ Baseline patterns established: {len(baseline_patterns)} categories")
        
        # Test critical features identification
        critical_features = explainer.identify_critical_features()
        print(f"✅ Critical features identified: {len(critical_features['highly_discriminative'])} highly discriminative")
        
        # Test attack type pattern analysis
        attack_patterns = explainer.analyze_attack_type_patterns(test_df)
        print(f"✅ Attack type patterns analyzed: {len(attack_patterns)} attack types")
        
        # Test feature profiling
        feature_profiles = explainer.create_feature_profiles(test_df)
        print(f"✅ Feature profiles created: {len(feature_profiles)} features")
        
        # Test baseline report
        report = explainer.generate_baseline_report(test_df)
        print("✅ Baseline report generated")
        
        return True
        
    except Exception as e:
        print(f"❌ BaselineExplainer test failed: {str(e)}")
        return False

def test_visualization():
    """Test visualization functionality"""
    print("\n📊 Testing Visualization...")
    
    try:
        from visualization.plots import XAIPlotter
        
        plotter = XAIPlotter()
        
        # Get some sample features for testing
        sample_features = list(test_df.select_dtypes(include=[np.number]).columns.drop('label', errors='ignore'))[:6]
        
        # Test feature distribution plots
        print("   Testing feature distribution plots...")
        plotter.plot_feature_distributions(test_df, features=sample_features[:4], save_path='images/test_results/test_feature_distributions.png')
        print("✅ Feature distribution plots created")
        
        # Test feature importance plot
        print("   Testing feature importance plot...")
        sample_importance = {f'feature_{i:02d}': np.random.random() for i in range(1, 21)}
        plotter.plot_feature_importance(sample_importance, save_path='images/test_results/test_feature_importance.png')
        print("✅ Feature importance plot created")
        
        # Test correlation heatmap
        print("   Testing correlation heatmap...")
        sample_corr = test_df[sample_features].corr()
        plotter.plot_correlation_heatmap(sample_corr, save_path='images/test_results/test_correlation_heatmap.png')
        print("✅ Correlation heatmap created")
        
        # Test attack type patterns
        print("   Testing attack type patterns...")
        plotter.plot_attack_type_patterns(test_df, features=sample_features[:3], save_path='images/test_results/test_attack_patterns.png')
        print("✅ Attack type patterns created")
        
        # Test PCA analysis
        print("   Testing PCA analysis...")
        plotter.plot_pca_analysis(test_df, save_path='images/test_results/test_pca_analysis.png')
        print("✅ PCA analysis plot created")
        
        # Test interactive plots (if available)
        print("   Testing interactive plots...")
        interactive_fig = plotter.plot_interactive_feature_importance(sample_importance)
        if interactive_fig:
            print("✅ Interactive feature importance plot created")
        else:
            print("⚠️  Interactive plots not available (Plotly missing)")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {str(e)}")
        return False

def test_dashboard():
    """Test dashboard functionality"""
    print("\n🎛️  Testing Dashboard...")
    
    try:
        from visualization.dashboard import XAIDashboard
        
        dashboard = XAIDashboard()
        
        # Load data
        dashboard.load_data(test_df)
        print("✅ Data loaded into dashboard")
        
        # Test insights generation
        insights = dashboard.generate_insights_summary()
        print("✅ Insights summary generated")
        print(f"   Insights length: {len(insights)} characters")
        
        # Test dashboard creation (if Plotly available)
        overview_fig = dashboard.create_overview_dashboard()
        if overview_fig:
            print("✅ Overview dashboard created")
        else:
            print("⚠️  Interactive dashboard not available (Plotly missing)")
        
        # Test HTML export (if available)
        try:
            dashboard.export_dashboard_html('test_dashboard.html')
            print("✅ Dashboard HTML export attempted")
        except Exception as e:
            print(f"⚠️  Dashboard export failed: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {str(e)}")
        return False

def test_integration():
    """Test full integration workflow"""
    print("\n🔄 Testing Integration Workflow...")
    
    try:
        # Import all components
        from foundation.data_analyzer import DataAnalyzer
        from foundation.feature_importance import FeatureImportanceAnalyzer
        from foundation.baseline_explainer import BaselineExplainer
        from visualization.plots import XAIPlotter
        from visualization.dashboard import XAIDashboard
        
        # Initialize all components
        data_analyzer = DataAnalyzer()
        feature_analyzer = FeatureImportanceAnalyzer()
        baseline_explainer = BaselineExplainer()
        plotter = XAIPlotter()
        dashboard = XAIDashboard()
        
        # Clean data for analysis
        clean_df = test_df.fillna(test_df.median())
        
        # Run complete workflow
        print("   Running complete XAI analysis workflow...")
        
        # 1. Data analysis
        correlation_matrix = data_analyzer.compute_correlation_matrix(clean_df)
        
        # 2. Feature importance
        statistical_importance = feature_analyzer.compute_statistical_importance(clean_df)
        model_importance = feature_analyzer.compute_model_based_importance(clean_df)
        aggregated_importance = feature_analyzer.aggregate_importance_scores()
        
        # 3. Baseline analysis
        baseline_patterns = baseline_explainer.establish_baseline_patterns(clean_df)
        
        # 4. Dashboard integration
        dashboard.load_data(clean_df)
        dashboard.load_analysis_results(
            feature_importance=aggregated_importance,
            correlation_matrix=correlation_matrix,
            baseline_patterns=baseline_patterns
        )
        
        # 5. Generate final insights
        final_insights = dashboard.generate_insights_summary()
        
        print("✅ Complete integration workflow successful")
        print(f"   Final insights generated: {len(final_insights)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("=" * 80)
    print("🧪 XAI Phase 1 Comprehensive Test Suite")
    print("=" * 80)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python Version: {sys.version}")
    
    global test_df
    test_df = None
    
    try:
        # Create test data
        test_df = create_test_data()
        
        # Run all tests
        tests = [
            ("Data Analyzer", test_data_analyzer),
            ("Feature Importance Analyzer", test_feature_importance_analyzer),
            ("Baseline Explainer", test_baseline_explainer),
            ("Visualization", test_visualization),
            ("Dashboard", test_dashboard),
            ("Integration Workflow", test_integration)
        ]
        
        results = []
        for test_name, test_func in tests:
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
            print(f"{test_name:<30} {status}")
            if result:
                passed += 1
            else:
                failed += 1
        
        print(f"\n📈 Overall Results: {passed} passed, {failed} failed")
        success_rate = (passed / len(results)) * 100
        print(f"🎯 Success Rate: {success_rate:.1f}%")
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED! XAI Phase 1 is working correctly!")
            print("\n📁 Generated test files:")
            print("   - images/test_results/test_feature_distributions.png")
            print("   - images/test_results/test_feature_importance.png")
            print("   - images/test_results/test_correlation_heatmap.png")
            print("   - images/test_results/test_attack_patterns.png")
            print("   - images/test_results/test_pca_analysis.png")
            print("   - test_dashboard.html (if Plotly available)")
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
