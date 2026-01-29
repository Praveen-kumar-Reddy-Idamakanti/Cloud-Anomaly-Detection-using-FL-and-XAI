"""
Feature Importance Analyzer for XAI Foundation

Provides comprehensive feature importance analysis including:
- Statistical feature importance
- Mutual information analysis
- Feature selection based on importance
- Attack type-specific feature importance
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif, 
    f_classif, 
    chi2,
    SelectKBest,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Optional SHAP import (may have NumPy compatibility issues)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analyzer for network traffic features
    """
    
    def __init__(self):
        self.feature_importance_scores = {}
        self.selected_features = {}
        self.scaler = StandardScaler()
        
    def compute_statistical_importance(self, df: pd.DataFrame, label_col: str = 'label') -> dict:
        """
        Compute statistical feature importance scores
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            Dictionary containing statistical importance scores
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")
            
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Select only numeric features
        numeric_features = X.select_dtypes(include=[np.number])
        
        importance_scores = {}
        
        # ANOVA F-test scores
        f_scores, p_values = f_classif(numeric_features, y)
        importance_scores['anova_f'] = dict(zip(numeric_features.columns, f_scores))
        importance_scores['anova_p'] = dict(zip(numeric_features.columns, p_values))
        
        # Mutual Information
        mi_scores = mutual_info_classif(numeric_features, y)
        importance_scores['mutual_info'] = dict(zip(numeric_features.columns, mi_scores))
        
        # Chi-square test (for non-negative features)
        non_negative_features = numeric_features.loc[:, (numeric_features >= 0).all()]
        if len(non_negative_features.columns) > 0:
            chi2_scores, chi2_p_values = chi2(non_negative_features, y)
            importance_scores['chi2'] = dict(zip(non_negative_features.columns, chi2_scores))
            importance_scores['chi2_p'] = dict(zip(non_negative_features.columns, chi2_p_values))
        
        self.feature_importance_scores['statistical'] = importance_scores
        return importance_scores
    
    def compute_model_based_importance(self, df: pd.DataFrame, label_col: str = 'label') -> dict:
        """
        Compute model-based feature importance
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            Dictionary containing model-based importance scores
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")
            
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Select only numeric features
        numeric_features = X.select_dtypes(include=[np.number])
        X_scaled = self.scaler.fit_transform(numeric_features)
        
        importance_scores = {}
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        importance_scores['random_forest'] = dict(zip(numeric_features.columns, rf.feature_importances_))
        
        # Extra Trees importance
        et = ExtraTreesClassifier(n_estimators=100, random_state=42)
        et.fit(X_scaled, y)
        importance_scores['extra_trees'] = dict(zip(numeric_features.columns, et.feature_importances_))
        
        # Logistic Regression coefficients (absolute value)
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_scaled, y)
        importance_scores['logistic_regression'] = dict(zip(numeric_features.columns, np.abs(lr.coef_[0])))
        
        self.feature_importance_scores['model_based'] = importance_scores
        return importance_scores
    
    def compute_shap_importance(self, df: pd.DataFrame, label_col: str = 'label', 
                               sample_size: int = 100) -> dict:
        """
        Compute SHAP-based feature importance
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            sample_size: Number of samples to use for SHAP analysis
            
        Returns:
            Dictionary containing SHAP importance scores
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available due to NumPy compatibility. Skipping SHAP analysis.")
            return {}
            
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")
            
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Select only numeric features and sample
        numeric_features = X.select_dtypes(include=[np.number])
        if len(numeric_features) > sample_size:
            sample_indices = np.random.choice(len(numeric_features), sample_size, replace=False)
            X_sample = numeric_features.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
        else:
            X_sample = numeric_features
            y_sample = y
            
        X_scaled = self.scaler.fit_transform(X_sample)
        
        # Train a model for SHAP analysis
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_scaled, y_sample)
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        # For binary classification, take the mean absolute SHAP values for class 1
        if isinstance(shap_values, list):
            shap_values = np.abs(shap_values[1])
        else:
            shap_values = np.abs(shap_values)
            
        # Mean SHAP value per feature
        mean_shap_values = np.mean(shap_values, axis=0)
        importance_scores = dict(zip(numeric_features.columns, mean_shap_values))
        
        self.feature_importance_scores['shap'] = importance_scores
        return importance_scores
    
    def compute_attack_type_specific_importance(self, df: pd.DataFrame, 
                                              label_col: str = 'label') -> dict:
        """
        Compute feature importance for specific attack types
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            Dictionary containing attack type-specific importance
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")
            
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Get unique attack types (excluding normal traffic)
        attack_types = sorted(y.unique())
        attack_type_importance = {}
        
        for attack_type in attack_types:
            if attack_type == 0:  # Skip normal traffic
                continue
                
            # Create binary classification for this attack type
            y_binary = (y == attack_type).astype(int)
            
            # Compute mutual information for this specific attack type
            numeric_features = X.select_dtypes(include=[np.number])
            mi_scores = mutual_info_classif(numeric_features, y_binary)
            
            attack_type_importance[f'attack_type_{attack_type}'] = dict(
                zip(numeric_features.columns, mi_scores)
            )
        
        self.feature_importance_scores['attack_type_specific'] = attack_type_importance
        return attack_type_importance
    
    def aggregate_importance_scores(self, method: str = 'mean') -> dict:
        """
        Aggregate importance scores from different methods
        
        Args:
            method: Aggregation method ('mean', 'median', 'max')
            
        Returns:
            Dictionary containing aggregated importance scores
        """
        if not self.feature_importance_scores:
            raise ValueError("No importance scores computed. Run importance computation methods first.")
            
        # Collect all importance scores
        all_scores = {}
        
        # Statistical scores
        if 'statistical' in self.feature_importance_scores:
            for score_type, scores in self.feature_importance_scores['statistical'].items():
                if score_type.endswith('_p'):  # Skip p-values
                    continue
                for feature, score in scores.items():
                    if feature not in all_scores:
                        all_scores[feature] = []
                    all_scores[feature].append(score)
        
        # Model-based scores
        if 'model_based' in self.feature_importance_scores:
            for model_type, scores in self.feature_importance_scores['model_based'].items():
                for feature, score in scores.items():
                    if feature not in all_scores:
                        all_scores[feature] = []
                    all_scores[feature].append(score)
        
        # SHAP scores
        if 'shap' in self.feature_importance_scores:
            for feature, score in self.feature_importance_scores['shap'].items():
                if feature not in all_scores:
                    all_scores[feature] = []
                all_scores[feature].append(score)
        
        # Aggregate scores
        aggregated_scores = {}
        for feature, scores in all_scores.items():
            if method == 'mean':
                aggregated_scores[feature] = np.mean(scores)
            elif method == 'median':
                aggregated_scores[feature] = np.median(scores)
            elif method == 'max':
                aggregated_scores[feature] = np.max(scores)
        
        # Sort by importance
        aggregated_scores = dict(sorted(aggregated_scores.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return aggregated_scores
    
    def select_top_features(self, aggregated_scores: dict, top_k: int = 20) -> list:
        """
        Select top k features based on aggregated importance scores
        
        Args:
            aggregated_scores: Aggregated importance scores
            top_k: Number of top features to select
            
        Returns:
            List of top feature names
        """
        return list(aggregated_scores.keys())[:top_k]
    
    def generate_feature_importance_report(self, df: pd.DataFrame, 
                                         label_col: str = 'label') -> str:
        """
        Generate comprehensive feature importance report
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            Formatted feature importance report string
        """
        # Compute all importance scores
        self.compute_statistical_importance(df, label_col)
        self.compute_model_based_importance(df, label_col)
        self.compute_shap_importance(df, label_col)
        self.compute_attack_type_specific_importance(df, label_col)
        
        # Aggregate scores
        aggregated_scores = self.aggregate_importance_scores('mean')
        top_features = self.select_top_features(aggregated_scores, 20)
        
        report = f"""
=== FEATURE IMPORTANCE ANALYSIS REPORT ===

Dataset Overview:
- Total Features Analyzed: {len(df.columns) - 1}
- Target Variable: {label_col}

Top 20 Most Important Features:
"""
        
        for i, feature in enumerate(top_features, 1):
            score = aggregated_scores[feature]
            report += f"{i:2d}. {feature:<30} {score:.4f}\n"
        
        report += f"""

Importance Method Summary:
- Statistical Tests (ANOVA F-test, Mutual Information): Computed
- Model-based (Random Forest, Extra Trees, Logistic Regression): Computed
- SHAP Values: Computed
- Attack Type-specific Analysis: Computed

Feature Selection Recommendations:
- Top 10 features for lightweight models: {', '.join(top_features[:10])}
- Top 20 features for balanced performance: {', '.join(top_features[:20])}
"""
        
        return report
