"""
XAI Phase 1 Demonstration Script

This script demonstrates the Phase 1 XAI foundation capabilities:
- Data quality analysis
- Feature importance analysis
- Baseline pattern analysis
- Visualization suite
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path to import XAI modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xai.foundation.data_analyzer import DataAnalyzer
from xai.foundation.feature_importance import FeatureImportanceAnalyzer
from xai.foundation.baseline_explainer import BaselineExplainer
from xai.visualization.plots import XAIPlotter
from xai.visualization.dashboard import XAIDashboard

def create_sample_data(n_samples: int = 1000):
    """
    Create sample network traffic data for demonstration
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Sample DataFrame with network traffic features
    """
    np.random.seed(42)
    
    # Generate 78 network traffic features
    feature_names = [f'feature_{i:02d}' for i in range(1, 79)]
    
    # Create normal traffic patterns
    normal_data = np.random.normal(0, 1, (n_samples // 2, 78))
    
    # Create anomalous traffic patterns (different means/variances)
    anomaly_data = np.random.normal(2, 1.5, (n_samples // 2, 78))
    
    # Combine data
    X = np.vstack([normal_data, anomaly_data])
    
    # Create labels (0 = normal, 1 = anomaly)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y
    
    # Add some missing values and duplicates for realism
    df.loc[np.random.choice(df.index, 20), 'feature_01'] = np.nan
    df = pd.concat([df, df.sample(10)], ignore_index=True)
    
    return df

def demonstrate_phase1():
    """
    Demonstrate Phase 1 XAI capabilities
    """
    print("=" * 60)
    print("XAI Phase 1 Foundation Demonstration")
    print("=" * 60)
    
    # Create sample data
    print("\n📊 Creating sample network traffic data...")
    df = create_sample_data(1000)
    print(f"Created dataset with {len(df)} samples and {len(df.columns)} features")
    
    # Initialize XAI components
    print("\n🔧 Initializing XAI components...")
    data_analyzer = DataAnalyzer()
    feature_analyzer = FeatureImportanceAnalyzer()
    baseline_explainer = BaselineExplainer()
    plotter = XAIPlotter()
    dashboard = XAIDashboard()
    
    # Data Quality Analysis
    print("\n📈 Performing data quality analysis...")
    quality_report = data_analyzer.analyze_data_quality(df)
    print(f"✅ Data quality analysis completed")
    print(f"   - Missing values: {sum(quality_report['missing_values'].values())}")
    print(f"   - Duplicate rows: {quality_report['duplicate_rows']}")
    
    # Correlation Analysis
    print("\n🔗 Computing correlation matrix...")
    correlation_matrix = data_analyzer.compute_correlation_matrix(df)
    highly_correlated = data_analyzer.find_highly_correlated_features(threshold=0.8)
    print(f"✅ Correlation analysis completed")
    print(f"   - Highly correlated pairs: {len(highly_correlated)}")
    
    # Feature Importance Analysis
    print("\n🎯 Computing feature importance...")
    statistical_importance = feature_analyzer.compute_statistical_importance(df)
    model_importance = feature_analyzer.compute_model_based_importance(df)
    aggregated_importance = feature_analyzer.aggregate_importance_scores()
    top_features = feature_analyzer.select_top_features(aggregated_importance, 10)
    print(f"✅ Feature importance analysis completed")
    print(f"   - Top 5 features: {', '.join(top_features[:5])}")
    
    # Baseline Pattern Analysis
    print("\n🔍 Establishing baseline patterns...")
    baseline_patterns = baseline_explainer.establish_baseline_patterns(df)
    critical_features = baseline_explainer.identify_critical_features()
    print(f"✅ Baseline pattern analysis completed")
    print(f"   - Highly discriminative features: {len(critical_features['highly_discriminative'])}")
    
    # Generate Reports
    print("\n📋 Generating analysis reports...")
    
    # Data analysis summary
    data_summary = data_analyzer.generate_summary_report(df)
    print("\n" + "="*50)
    print("DATA ANALYSIS SUMMARY")
    print("="*50)
    print(data_summary)
    
    # Feature importance summary
    feature_summary = feature_analyzer.generate_feature_importance_report(df)
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE SUMMARY")
    print("="*50)
    print(feature_summary)
    
    # Baseline analysis summary
    baseline_summary = baseline_explainer.generate_baseline_report(df)
    print("\n" + "="*50)
    print("BASELINE PATTERN SUMMARY")
    print("="*50)
    print(baseline_summary)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Feature distributions
    plotter.plot_feature_distributions(df, features=top_features[:6], save_path='feature_distributions.png')
    print("✅ Feature distribution plots saved")
    
    # Feature importance plot
    plotter.plot_feature_importance(aggregated_importance, save_path='feature_importance.png')
    print("✅ Feature importance plot saved")
    
    # Correlation heatmap
    plotter.plot_correlation_heatmap(correlation_matrix.iloc[:15, :15], save_path='correlation_heatmap.png')
    print("✅ Correlation heatmap saved")
    
    # Attack type patterns
    plotter.plot_attack_type_patterns(df, features=top_features[:4], save_path='attack_patterns.png')
    print("✅ Attack type pattern plots saved")
    
    # Create dashboard
    print("\n🎛️  Creating interactive dashboard...")
    dashboard.load_data(df)
    dashboard.load_analysis_results(
        feature_importance=aggregated_importance,
        correlation_matrix=correlation_matrix,
        baseline_patterns=baseline_patterns
    )
    
    # Export dashboard
    dashboard.export_dashboard_html('xai_phase1_dashboard.html')
    print("✅ Interactive dashboard exported")
    
    # Generate insights
    print("\n💡 Key Insights:")
    insights = dashboard.generate_insights_summary()
    print(insights)
    
    print("\n" + "="*60)
    print("XAI Phase 1 Demonstration Completed Successfully!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("   - feature_distributions.png")
    print("   - feature_importance.png") 
    print("   - correlation_heatmap.png")
    print("   - attack_patterns.png")
    print("   - xai_phase1_dashboard.html")
    print("\n🎯 Phase 1 Deliverables:")
    print("   ✅ XAI environment setup")
    print("   ✅ Feature importance analysis")
    print("   ✅ Data analysis dashboard")
    print("   ✅ Feature correlation matrix")
    print("   ✅ Baseline statistics document")

if __name__ == "__main__":
    demonstrate_phase1()
