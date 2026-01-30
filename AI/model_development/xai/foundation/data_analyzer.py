"""
Data Analyzer for XAI Foundation

Provides comprehensive data analysis capabilities including:
- Feature correlation analysis
- Statistical summaries
- Data quality assessment
- Normal vs anomaly pattern analysis
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """
    Comprehensive data analyzer for network traffic features
    """
    
    def __init__(self):
        self.feature_stats = {}
        self.correlation_matrix = None
        self.pca_model = None
        self.scaler = StandardScaler()
        
    def analyze_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Analyze data quality metrics
        
        Args:
            df: Input DataFrame with network traffic data
            
        Returns:
            Dictionary containing data quality metrics
        """
        quality_report = {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Feature-wise statistics
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                quality_report[f'{col}_stats'] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'q25': df[col].quantile(0.25),
                    'q50': df[col].median(),
                    'q75': df[col].quantile(0.75),
                    'skewness': stats.skew(df[col].dropna()),
                    'kurtosis': stats.kurtosis(df[col].dropna())
                }
        
        return quality_report
    
    def compute_correlation_matrix(self, df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlation matrix for features
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix DataFrame
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if method == 'pearson':
            self.correlation_matrix = numeric_df.corr(method='pearson')
        elif method == 'spearman':
            self.correlation_matrix = numeric_df.corr(method='spearman')
        elif method == 'kendall':
            self.correlation_matrix = numeric_df.corr(method='kendall')
        else:
            raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
            
        return self.correlation_matrix
    
    def find_highly_correlated_features(self, threshold: float = 0.8) -> dict:
        """
        Identify highly correlated feature pairs
        
        Args:
            threshold: Correlation threshold for considering features highly correlated
            
        Returns:
            Dictionary of highly correlated feature pairs
        """
        if self.correlation_matrix is None:
            raise ValueError("Correlation matrix not computed. Call compute_correlation_matrix first.")
            
        highly_correlated = {}
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = abs(self.correlation_matrix.iloc[i, j])
                if corr_value > threshold:
                    feat1 = self.correlation_matrix.columns[i]
                    feat2 = self.correlation_matrix.columns[j]
                    highly_correlated[f"{feat1}_vs_{feat2}"] = corr_value
                    
        return highly_correlated
    
    def analyze_feature_distributions(self, df: pd.DataFrame, label_col: str = 'label') -> dict:
        """
        Analyze feature distributions by class (normal vs anomaly)
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            Dictionary containing distribution analysis
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")
            
        distribution_analysis = {}
        
        # Separate normal and anomaly samples
        normal_samples = df[df[label_col] == 0]
        anomaly_samples = df[df[label_col] != 0]
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == label_col:
                continue
                
            normal_stats = {
                'mean': normal_samples[col].mean(),
                'std': normal_samples[col].std(),
                'median': normal_samples[col].median(),
                'q25': normal_samples[col].quantile(0.25),
                'q75': normal_samples[col].quantile(0.75)
            }
            
            anomaly_stats = {
                'mean': anomaly_samples[col].mean(),
                'std': anomaly_samples[col].std(),
                'median': anomaly_samples[col].median(),
                'q25': anomaly_samples[col].quantile(0.25),
                'q75': anomaly_samples[col].quantile(0.75)
            }
            
            # Statistical test for difference
            statistic, p_value = stats.ks_2samp(
                normal_samples[col].dropna(), 
                anomaly_samples[col].dropna()
            )
            
            distribution_analysis[col] = {
                'normal_stats': normal_stats,
                'anomaly_stats': anomaly_stats,
                'ks_statistic': statistic,
                'p_value': p_value,
                'significant_difference': p_value < 0.05
            }
            
        return distribution_analysis
    
    def perform_pca_analysis(self, df: pd.DataFrame, n_components: int = 10) -> dict:
        """
        Perform PCA analysis for dimensionality reduction understanding
        
        Args:
            df: Input DataFrame (numeric features only)
            n_components: Number of PCA components to compute
            
        Returns:
            Dictionary containing PCA results
        """
        # Select numeric columns and scale data
        numeric_df = df.select_dtypes(include=[np.number])
        scaled_data = self.scaler.fit_transform(numeric_df)
        
        # Perform PCA
        self.pca_model = PCA(n_components=n_components)
        pca_result = self.pca_model.fit_transform(scaled_data)
        
        pca_analysis = {
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(self.pca_model.explained_variance_ratio_),
            'components': self.pca_model.components_,
            'feature_names': numeric_df.columns.tolist(),
            'pca_result': pca_result
        }
        
        return pca_analysis
    
    def identify_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> dict:
        """
        Identify outliers in the dataset
        
        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary containing outlier information
        """
        outlier_analysis = {}
        numeric_df = df.select_dtypes(include=[np.number])
        
        for col in numeric_df.columns:
            if method == 'iqr':
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(numeric_df[col].dropna()))
                outliers = numeric_df[z_scores > threshold]
                
            outlier_analysis[col] = {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(numeric_df)) * 100,
                'outlier_indices': outliers.index.tolist()
            }
            
        return outlier_analysis
    
    def generate_summary_report(self, df: pd.DataFrame, label_col: str = 'label') -> str:
        """
        Generate a comprehensive summary report
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            Formatted summary report string
        """
        quality_report = self.analyze_data_quality(df)
        correlation_matrix = self.compute_correlation_matrix(df)
        highly_correlated = self.find_highly_correlated_features()
        distributions = self.analyze_feature_distributions(df, label_col)
        outliers = self.identify_outliers(df)
        
        report = f"""
=== DATA ANALYSIS SUMMARY REPORT ===

Dataset Overview:
- Total Samples: {quality_report['total_samples']:,}
- Total Features: {quality_report['total_features']}
- Missing Values: {sum(quality_report['missing_values'].values())}
- Duplicate Rows: {quality_report['duplicate_rows']}

Correlation Analysis:
- Highly Correlated Feature Pairs (|r| > 0.8): {len(highly_correlated)}

Feature Distribution Analysis:
- Features with Significant Normal vs Anomaly Differences: 
  {sum(1 for feat, stats in distributions.items() if stats['significant_difference'])} / {len(distributions)}

Outlier Analysis:
- Average Outlier Percentage per Feature: 
  {np.mean([stats['outlier_percentage'] for stats in outliers.values()]):.2f}%

Top Highly Correlated Feature Pairs:
"""
        
        # Add top 5 highly correlated pairs
        sorted_correlations = sorted(highly_correlated.items(), key=lambda x: x[1], reverse=True)[:5]
        for pair, corr in sorted_correlations:
            report += f"- {pair}: {corr:.3f}\n"
            
        return report
