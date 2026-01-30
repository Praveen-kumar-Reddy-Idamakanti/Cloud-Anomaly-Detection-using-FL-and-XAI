# How to Test XAI Phase 1 Implementation

## 🧪 **Quick Testing Guide**

This guide shows you how to test the XAI Phase 1 implementation to verify it's working correctly.

---

## 🚀 **Method 1: Run Comprehensive Test Suite**

### **Step 1: Navigate to XAI Directory**
```bash
cd "model_development/xai"
```

### **Step 2: Run the Test Suite**
```bash
python test_xai_phase1.py
```

### **Expected Results:**
```
🧪 XAI Phase 1 Comprehensive Test Suite
==================================================================
📊 Creating test data...
✅ Created test dataset: 1010 samples, 79 features

🔍 Testing DataAnalyzer...
✅ Data quality analysis completed
✅ Correlation matrix computed: (79, 79)
✅ Found 0 highly correlated feature pairs
✅ Feature distribution analysis completed for 78 features
✅ PCA analysis completed: (5,)
✅ Outlier detection completed for 79 features
✅ Summary report generated

🎯 Testing FeatureImportanceAnalyzer...
✅ Statistical importance computed: 3 methods
✅ Model-based importance computed: 3 models
✅ Attack type-specific importance computed
✅ Aggregated importance scores: 78 features
✅ Top 20 features selected: 20
✅ Feature importance report generated

🔬 Testing BaselineExplainer...
✅ Baseline patterns established: 3 categories
✅ Critical features identified: 78 highly discriminative
✅ Attack type patterns analyzed: 1 attack types
✅ Feature profiles created: 78 features
✅ Baseline report generated

📊 Testing Visualization...
✅ Feature distribution plots created
✅ Feature importance plot created
✅ Correlation heatmap created
✅ Attack type patterns created
✅ PCA analysis plot created

🎛️  Testing Dashboard...
✅ Data loaded into dashboard
✅ Insights summary generated
✅ Dashboard HTML export attempted

🔄 Testing Integration Workflow...
✅ Complete integration workflow successful
✅ Final insights generated: 421 characters

📈 Overall Results: 6 passed, 0 failed
🎯 Success Rate: 100.0%

🎉 ALL TESTS PASSED! XAI Phase 1 is working correctly!
```

---

## 🔧 **Method 2: Test Individual Components**

### **Test Data Analysis**
```python
from foundation.data_analyzer import DataAnalyzer
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'feature_01': np.random.normal(0, 1, 100),
    'feature_02': np.random.normal(1, 2, 100),
    'label': np.random.choice([0, 1], 100)
})

# Test analyzer
analyzer = DataAnalyzer()
quality_report = analyzer.analyze_data_quality(df)
correlation_matrix = analyzer.compute_correlation_matrix(df)
summary = analyzer.generate_summary_report(df)

print("✅ DataAnalyzer working correctly!")
```

### **Test Feature Importance**
```python
from foundation.feature_importance import FeatureImportanceAnalyzer

# Clean data (handle NaN values)
clean_df = df.fillna(df.median())

# Test feature importance
analyzer = FeatureImportanceAnalyzer()
statistical_importance = analyzer.compute_statistical_importance(clean_df)
model_importance = analyzer.compute_model_based_importance(clean_df)
aggregated = analyzer.aggregate_importance_scores()

print("✅ FeatureImportanceAnalyzer working correctly!")
```

### **Test Baseline Analysis**
```python
from foundation.baseline_explainer import BaselineExplainer

explainer = BaselineExplainer()
baseline_patterns = explainer.establish_baseline_patterns(clean_df)
critical_features = explainer.identify_critical_features()
report = explainer.generate_baseline_report(clean_df)

print("✅ BaselineExplainer working correctly!")
```

### **Test Visualizations**
```python
from visualization.plots import XAIPlotter

plotter = XAIPlotter()

# Test static plots
plotter.plot_feature_distributions(clean_df, features=['feature_01', 'feature_02'])
plotter.plot_feature_importance(aggregated)
plotter.plot_correlation_heatmap(clean_df[['feature_01', 'feature_02']].corr())

print("✅ Visualizations working correctly!")
```

### **Test Dashboard**
```python
from visualization.dashboard import XAIDashboard

dashboard = XAIDashboard()
dashboard.load_data(clean_df)
dashboard.load_analysis_results(
    feature_importance=aggregated,
    correlation_matrix=clean_df.corr()
)
insights = dashboard.generate_insights_summary()

print("✅ Dashboard working correctly!")
print(f"Insights generated: {len(insights)} characters")
```

---

## 📊 **Method 3: Test with Real Data**

### **Using Your Own Dataset**
```python
import pandas as pd
from foundation.data_analyzer import DataAnalyzer
from foundation.feature_importance import FeatureImportanceAnalyzer
from visualization.plots import XAIPlotter

# Load your data (replace with your actual data path)
# df = pd.read_csv('your_data.csv')

# For testing, create sample data with 78 features
np.random.seed(42)
feature_names = [f'feature_{i:02d}' for i in range(1, 79)]
X = np.random.normal(0, 1, (1000, 78))
y = np.random.choice([0, 1], 1000)
df = pd.DataFrame(X, columns=feature_names)
df['label'] = y

# Clean data
clean_df = df.fillna(df.median())

# Run analysis
analyzer = DataAnalyzer()
feature_analyzer = FeatureImportanceAnalyzer()
plotter = XAIPlotter()

# Data analysis
correlation_matrix = analyzer.compute_correlation_matrix(clean_df)
quality_report = analyzer.analyze_data_quality(clean_df)

# Feature importance
importance = feature_analyzer.compute_statistical_importance(clean_df)
aggregated = feature_analyzer.aggregate_importance_scores()

# Visualizations
plotter.plot_feature_importance(aggregated, save_path='images/test_results/my_feature_importance.png')
plotter.plot_correlation_heatmap(correlation_matrix.iloc[:10, :10], save_path='images/test_results/my_correlation.png')

print("✅ Real data test completed!")
```

---

## 🔍 **Verification Checklist**

### **✅ What to Look For:**

1. **No Import Errors**: All modules should import without errors
2. **No Runtime Errors**: Tests should run without exceptions
3. **Generated Files**: Test should create PNG files for visualizations
4. **Meaningful Output**: Reports and insights should be generated
5. **Performance**: Tests should complete in reasonable time (< 30 seconds)

### **✅ Expected Files Generated:**
- `images/test_results/test_feature_distributions.png`
- `images/test_results/test_feature_importance.png`
- `images/test_results/test_correlation_heatmap.png`
- `images/test_results/test_attack_patterns.png`
- `images/test_results/test_pca_analysis.png`

### **✅ Expected Outputs:**
- Data quality reports
- Feature importance rankings
- Correlation matrices
- Baseline pattern analysis
- Insights summaries

---

## ⚠️ **Troubleshooting**

### **Common Issues and Solutions:**

#### **Import Errors:**
```bash
# If you get import errors, check your Python path
python -c "import sys; print(sys.path)"

# Make sure you're in the right directory
cd "model_development/xai"
```

#### **Missing Dependencies:**
```bash
# Install basic dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# For interactive features (optional)
pip install plotly shap
```

#### **SHAP Not Available:**
```
⚠️  SHAP not available due to NumPy compatibility. Skipping SHAP analysis.
```
**Solution**: This is expected with NumPy 2.3. The system works fine without SHAP.

#### **Plotly Not Available:**
```
⚠️  Interactive plots not available (Plotly missing)
```
**Solution**: Install Plotly for interactive features, or use static plots.

#### **NaN Value Errors:**
```
Input X contains NaN
```
**Solution**: The test script automatically handles this with `fillna()`.

---

## 🎯 **Success Indicators**

### **✅ Test Success Indicators:**
- All 6 test categories pass
- 100% success rate
- PNG files are generated
- No error messages
- Insights are generated with meaningful content

### **✅ Manual Verification:**
1. Check that PNG files are created and can be opened
2. Review the insights summary for meaningful content
3. Verify that feature importance rankings make sense
4. Confirm correlation matrices are computed correctly

---

## 📞 **Getting Help**

If you encounter issues:

1. **Check the test output** for specific error messages
2. **Verify dependencies** are installed correctly
3. **Ensure you're in the correct directory**
4. **Check data format** (should be pandas DataFrame with numeric features and label column)

---

## 🚀 **Next Steps**

Once testing is successful:

1. **Try with your real data**
2. **Explore different visualization options**
3. **Customize analysis parameters**
4. **Integrate with your existing pipeline**
5. **Proceed to Phase 2: Autoencoder Explainability**

---

**🎉 Happy Testing! Your XAI Phase 1 implementation is ready to use!**
