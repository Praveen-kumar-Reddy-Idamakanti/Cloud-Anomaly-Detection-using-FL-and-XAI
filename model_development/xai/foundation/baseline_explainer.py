"""
Baseline Explainer for XAI Foundation

Provides baseline pattern analysis including:
- Normal vs anomalous feature patterns
- Feature-wise baseline statistics
- Critical feature identification for attack types
- Pattern deviation analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class BaselineExplainer:
    """
    Baseline pattern explainer for normal vs anomalous traffic analysis
    """
    
    def __init__(self):
        self.baseline_stats = {}
        self.pattern_models = {}
        self.scaler = StandardScaler()
        
    def establish_baseline_patterns(self, df: pd.DataFrame, label_col: str = 'label') -> dict:
        """
        Establish baseline patterns for normal and anomalous traffic
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            Dictionary containing baseline patterns
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")
            
        # Separate normal and anomalous samples
        normal_samples = df[df[label_col] == 0]
        anomaly_samples = df[df[label_col] != 0]
        
        numeric_features = df.select_dtypes(include=[np.number]).columns.drop(label_col, errors='ignore')
        
        baseline_patterns = {
            'normal_patterns': {},
            'anomaly_patterns': {},
            'pattern_differences': {}
        }
        
        for feature in numeric_features:
            # Normal pattern statistics
            normal_data = normal_samples[feature].dropna()
            normal_stats = {
                'mean': normal_data.mean(),
                'std': normal_data.std(),
                'median': normal_data.median(),
                'q25': normal_data.quantile(0.25),
                'q75': normal_data.quantile(0.75),
                'min': normal_data.min(),
                'max': normal_data.max(),
                'skewness': stats.skew(normal_data),
                'kurtosis': stats.kurtosis(normal_data),
                'cv': normal_data.std() / normal_data.mean() if normal_data.mean() != 0 else np.inf
            }
            
            # Anomaly pattern statistics
            anomaly_data = anomaly_samples[feature].dropna()
            anomaly_stats = {
                'mean': anomaly_data.mean(),
                'std': anomaly_data.std(),
                'median': anomaly_data.median(),
                'q25': anomaly_data.quantile(0.25),
                'q75': anomaly_data.quantile(0.75),
                'min': anomaly_data.min(),
                'max': anomaly_data.max(),
                'skewness': stats.skew(anomaly_data),
                'kurtosis': stats.kurtosis(anomaly_data),
                'cv': anomaly_data.std() / anomaly_data.mean() if anomaly_data.mean() != 0 else np.inf
            }
            
            # Pattern differences
            mean_diff = abs(normal_stats['mean'] - anomaly_stats['mean'])
            std_diff = abs(normal_stats['std'] - anomaly_stats['std'])
            cv_ratio = anomaly_stats['cv'] / normal_stats['cv'] if normal_stats['cv'] != 0 else np.inf
            
            # Statistical significance test
            ks_stat, ks_p = stats.ks_2samp(normal_data, anomaly_data)
            
            pattern_differences = {
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'cv_ratio': cv_ratio,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p,
                'significant_difference': ks_p < 0.05,
                'effect_size': mean_diff / np.sqrt((normal_stats['std']**2 + anomaly_stats['std']**2) / 2)
            }
            
            baseline_patterns['normal_patterns'][feature] = normal_stats
            baseline_patterns['anomaly_patterns'][feature] = anomaly_stats
            baseline_patterns['pattern_differences'][feature] = pattern_differences
        
        self.baseline_stats = baseline_patterns
        return baseline_patterns
    
    def identify_critical_features(self, significance_threshold: float = 0.05, 
                                 effect_size_threshold: float = 0.5) -> dict:
        """
        Identify critical features that distinguish normal from anomalous traffic
        
        Args:
            significance_threshold: P-value threshold for statistical significance
            effect_size_threshold: Effect size threshold for practical significance
            
        Returns:
            Dictionary containing critical features analysis
        """
        if not self.baseline_stats:
            raise ValueError("Baseline patterns not established. Call establish_baseline_patterns first.")
            
        critical_features = {
            'statistically_significant': [],
            'practically_significant': [],
            'highly_discriminative': [],
            'feature_rankings': {}
        }
        
        feature_scores = {}
        
        for feature, differences in self.baseline_stats['pattern_differences'].items():
            # Check statistical significance
            if differences['significant_difference']:
                critical_features['statistically_significant'].append(feature)
            
            # Check practical significance (effect size)
            if differences['effect_size'] >= effect_size_threshold:
                critical_features['practically_significant'].append(feature)
            
            # Highly discriminative (both significant and large effect size)
            if (differences['significant_difference'] and 
                differences['effect_size'] >= effect_size_threshold):
                critical_features['highly_discriminative'].append(feature)
            
            # Composite score for ranking
            composite_score = (
                (1 - differences['ks_p_value']) * 0.4 +  # Statistical significance
                min(differences['effect_size'], 2) * 0.3 +  # Effect size (capped at 2)
                min(differences['cv_ratio'], 5) / 5 * 0.3  # Coefficient of variation ratio
            )
            feature_scores[feature] = composite_score
        
        # Sort features by composite score
        critical_features['feature_rankings'] = dict(
            sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return critical_features
    
    def analyze_attack_type_patterns(self, df: pd.DataFrame, label_col: str = 'label') -> dict:
        """
        Analyze patterns for specific attack types
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            Dictionary containing attack type-specific patterns
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")
            
        attack_types = sorted(df[label_col].unique())
        attack_patterns = {}
        
        numeric_features = df.select_dtypes(include=[np.number]).columns.drop(label_col, errors='ignore')
        
        for attack_type in attack_types:
            if attack_type == 0:  # Skip normal traffic
                continue
                
            attack_samples = df[df[label_col] == attack_type]
            normal_samples = df[df[label_col] == 0]
            
            attack_type_stats = {}
            
            for feature in numeric_features:
                attack_data = attack_samples[feature].dropna()
                normal_data = normal_samples[feature].dropna()
                
                if len(attack_data) == 0 or len(normal_data) == 0:
                    continue
                
                # Attack-specific statistics
                attack_stats = {
                    'mean': attack_data.mean(),
                    'std': attack_data.std(),
                    'median': attack_data.median(),
                    'min': attack_data.min(),
                    'max': attack_data.max()
                }
                
                # Comparison with normal
                normal_stats = {
                    'mean': normal_data.mean(),
                    'std': normal_data.std(),
                    'median': normal_data.median()
                }
                
                # Relative differences
                mean_ratio = attack_stats['mean'] / normal_stats['mean'] if normal_stats['mean'] != 0 else np.inf
                std_ratio = attack_stats['std'] / normal_stats['std'] if normal_stats['std'] != 0 else np.inf
                
                # Statistical test
                ks_stat, ks_p = stats.ks_2samp(attack_data, normal_data)
                
                attack_type_stats[feature] = {
                    'attack_stats': attack_stats,
                    'normal_stats': normal_stats,
                    'mean_ratio': mean_ratio,
                    'std_ratio': std_ratio,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'significant_difference': ks_p < 0.05
                }
            
            attack_patterns[f'attack_type_{attack_type}'] = attack_type_stats
        
        return attack_patterns
    
    def create_feature_profiles(self, df: pd.DataFrame, label_col: str = 'label') -> dict:
        """
        Create detailed feature profiles for normal and each attack type
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            Dictionary containing feature profiles
        """
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in DataFrame")
            
        feature_profiles = {}
        numeric_features = df.select_dtypes(include=[np.number]).columns.drop(label_col, errors='ignore')
        
        for feature in numeric_features:
            profile = {
                'overall_stats': {},
                'class_specific_stats': {},
                'discriminative_power': {}
            }
            
            # Overall statistics
            overall_data = df[feature].dropna()
            profile['overall_stats'] = {
                'mean': overall_data.mean(),
                'std': overall_data.std(),
                'median': overall_data.median(),
                'min': overall_data.min(),
                'max': overall_data.max(),
                'missing_percentage': (df[feature].isnull().sum() / len(df)) * 100
            }
            
            # Class-specific statistics
            for class_label in sorted(df[label_col].unique()):
                class_data = df[df[label_col] == class_label][feature].dropna()
                if len(class_data) > 0:
                    profile['class_specific_stats'][f'class_{class_label}'] = {
                        'mean': class_data.mean(),
                        'std': class_data.std(),
                        'median': class_data.median(),
                        'count': len(class_data)
                    }
            
            # Discriminative power (variance across classes)
            class_means = [stats['mean'] for stats in profile['class_specific_stats'].values()]
            if len(class_means) > 1:
                discriminative_score = np.std(class_means) / np.mean(class_means) if np.mean(class_means) != 0 else 0
                profile['discriminative_power'] = {
                    'score': discriminative_score,
                    'high_discriminative': discriminative_score > 0.5
                }
            
            feature_profiles[feature] = profile
        
        return feature_profiles
    
    def generate_baseline_report(self, df: pd.DataFrame, label_col: str = 'label') -> str:
        """
        Generate comprehensive baseline analysis report
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            
        Returns:
            Formatted baseline report string
        """
        # Establish baseline patterns
        baseline_patterns = self.establish_baseline_patterns(df, label_col)
        critical_features = self.identify_critical_features()
        attack_patterns = self.analyze_attack_type_patterns(df, label_col)
        feature_profiles = self.create_feature_profiles(df, label_col)
        
        report = f"""
=== BASELINE PATTERN ANALYSIS REPORT ===

Dataset Overview:
- Total Samples: {len(df):,}
- Features Analyzed: {len(df.columns) - 1}
- Normal Samples: {len(df[df[label_col] == 0]):,}
- Anomalous Samples: {len(df[df[label_col] != 0]):,}

Critical Features Analysis:
- Statistically Significant Features: {len(critical_features['statistically_significant'])}
- Practically Significant Features: {len(critical_features['practically_significant'])}
- Highly Discriminative Features: {len(critical_features['highly_discriminative'])}

Top 10 Most Discriminative Features:
"""
        
        top_features = list(critical_features['feature_rankings'].keys())[:10]
        for i, feature in enumerate(top_features, 1):
            score = critical_features['feature_rankings'][feature]
            report += f"{i:2d}. {feature:<30} {score:.4f}\n"
        
        report += f"""

Attack Type Analysis:
- Attack Types Identified: {len(attack_patterns)}
"""
        
        for attack_type, patterns in attack_patterns.items():
            significant_features = [feat for feat, stats in patterns.items() 
                                  if stats['significant_difference']]
            report += f"- {attack_type}: {len(significant_features)} significant features\n"
        
        report += """

Feature Profile Summary:
- High Discriminative Features: """ + str(len([f for f, p in feature_profiles.items() 
                                        if p.get('discriminative_power', {}).get('high_discriminative', False)]))
        
        report += """

Recommendations:
1. Focus on highly discriminative features for model development
2. Monitor critical features for real-time anomaly detection
3. Use attack type-specific patterns for targeted defense strategies
"""
        
        return report
