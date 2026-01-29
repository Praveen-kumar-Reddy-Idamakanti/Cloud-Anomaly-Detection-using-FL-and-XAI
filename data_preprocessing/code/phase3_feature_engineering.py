"""
Phase 3: Feature Engineering
Comprehensive feature engineering for CICIDS2017 dataset based on cleaned data
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class FeatureEngineeringPipeline:
    """Comprehensive feature engineering pipeline for network traffic data"""
    
    def __init__(self, cleaned_data_dir: str, output_dir: str):
        self.cleaned_data_dir = Path(cleaned_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature engineering statistics
        self.feature_stats = {}
        self.feature_results = {}
        
        # Define core network flow features to prioritize
        self.core_flow_features = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Flow Bytes/s', 'Flow Packets/s'
        ]
        
        # Define statistical features
        self.statistical_features = [
            'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance'
        ]
        
        # Define timing features
        self.timing_features = [
            'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        
        # Define flag features
        self.flag_features = [
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count'
        ]
        
        # Define ratio features
        self.ratio_features = [
            'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
        ]
        
    def run_complete_feature_engineering(self) -> Dict:
        """Run complete Phase 3 feature engineering pipeline"""
        logger.info("🚀 Starting Phase 3: Feature Engineering")
        
        # Step 3.1: Feature Selection
        logger.info("📊 Step 3.1: Feature Selection")
        self._perform_feature_selection()
        
        # Step 3.2: Feature Transformation
        logger.info("🔧 Step 3.2: Feature Transformation")
        self._perform_feature_transformation()
        
        # Step 3.3: Feature Creation
        logger.info("✨ Step 3.3: Feature Creation")
        self._create_new_features()
        
        # Step 3.4: Feature Scaling
        logger.info("⚖️ Step 3.4: Feature Scaling")
        self._perform_feature_scaling()
        
        # Generate feature engineering summary
        logger.info("📋 Generating Feature Engineering Summary")
        self._generate_feature_summary()
        
        # Save engineered features
        self._save_engineered_features()
        
        logger.info("✅ Phase 3 Complete: Feature Engineering")
        return self.feature_stats
    
    def _perform_feature_selection(self):
        """Perform feature selection to identify most relevant features"""
        cleaned_files = list(self.cleaned_data_dir.glob("cleaned_*.csv"))
        
        for file_path in cleaned_files:
            logger.info(f"Feature selection for: {file_path.name}")
            
            # Load cleaned data
            df = pd.read_csv(file_path)
            original_features = len(df.columns)
            
            # Remove Label column temporarily for feature selection
            labels = None
            if 'Label' in df.columns:
                labels = df['Label'].copy()
                df_features = df.drop('Label', axis=1)
            else:
                df_features = df.copy()
            
            # Step 3.1.1: Remove constant features
            constant_selector = VarianceThreshold(threshold=0.0)
            df_no_constant = df_features.copy()
            
            # Identify constant features
            constant_features = []
            for col in df_features.columns:
                if df_features[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                df_no_constant = df_features.drop(columns=constant_features)
                logger.info(f"Removed {len(constant_features)} constant features: {constant_features}")
            
            # Step 3.1.2: Remove highly correlated features
            correlation_matrix = df_no_constant.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with correlation > 0.95
            high_corr_features = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > 0.95)
            ]
            
            df_no_high_corr = df_no_constant.drop(columns=high_corr_features)
            if high_corr_features:
                logger.info(f"Removed {len(high_corr_features)} highly correlated features")
            
            # Step 3.1.3: Select top features based on importance
            selected_features = self._select_important_features(df_no_high_corr, labels)
            
            # Store results
            self.feature_results[file_path.name] = {
                'original_features': original_features,
                'constant_features_removed': len(constant_features),
                'high_corr_features_removed': len(high_corr_features),
                'final_selected_features': len(selected_features),
                'selected_feature_list': selected_features.tolist(),
                'feature_selection_actions': [
                    f"Removed {len(constant_features)} constant features",
                    f"Removed {len(high_corr_features)} highly correlated features",
                    f"Selected {len(selected_features)} most important features"
                ]
            }
    
    def _select_important_features(self, df: pd.DataFrame, labels: Optional[pd.Series] = None) -> np.ndarray:
        """Select most important features using statistical methods"""
        # If labels are available, use supervised feature selection
        if labels is not None:
            try:
                # Convert labels to binary (0 for BENIGN, 1 for others)
                binary_labels = labels.apply(lambda x: 0 if x == 'BENIGN' else 1)
                
                # Use SelectKBest with f_classif
                selector = SelectKBest(score_func=f_classif, k=min(30, len(df.columns)))
                selector.fit(df, binary_labels)
                selected_features = selector.get_support(indices=True)
                
                logger.info(f"Used supervised feature selection, selected {len(selected_features)} features")
                return selected_features
            except Exception as e:
                logger.warning(f"Supervised feature selection failed: {e}, using unsupervised")
        
        # Fallback to unsupervised selection based on feature variance
        feature_variances = df.var()
        top_features = feature_variances.nlargest(min(30, len(df.columns))).index
        selected_indices = [df.columns.get_loc(col) for col in top_features]
        
        logger.info(f"Used unsupervised feature selection, selected {len(selected_indices)} features")
        return np.array(selected_indices)
    
    def _perform_feature_transformation(self):
        """Perform feature transformations"""
        cleaned_files = list(self.cleaned_data_dir.glob("cleaned_*.csv"))
        
        for file_path in cleaned_files:
            logger.info(f"Feature transformation for: {file_path.name}")
            
            # Load cleaned data
            df = pd.read_csv(file_path)
            
            # Get selected features from previous step
            if file_path.name in self.feature_results:
                selected_feature_indices = self.feature_results[file_path.name]['selected_feature_list']
                selected_features = df.columns[selected_feature_indices].tolist()
                df_selected = df[selected_features].copy()
            else:
                df_selected = df.copy()
                selected_features = df.columns.tolist()
            
            # Apply transformations
            df_transformed = self._apply_transformations(df_selected)
            
            # Store transformation results
            if file_path.name in self.feature_results:
                self.feature_results[file_path.name]['feature_transformations'] = {
                    'original_selected_features': len(selected_features),
                    'transformed_features': len(df_transformed.columns),
                    'transformations_applied': self._get_transformation_summary(df_selected, df_transformed)
                }
            else:
                self.feature_results[file_path.name] = {
                    'feature_transformations': {
                        'original_selected_features': len(selected_features),
                        'transformed_features': len(df_transformed.columns),
                        'transformations_applied': self._get_transformation_summary(df_selected, df_transformed)
                    }
                }
    
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply various feature transformations"""
        df_transformed = df.copy()
        
        # Log transformation for heavily skewed features
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                skewness = df[col].skew()
                if abs(skewness) > 2.0 and df[col].min() > 0:
                    # Apply log transformation
                    df_transformed[f'{col}_log'] = np.log1p(df[col])
        
        # Square root transformation for moderately skewed features
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                skewness = df[col].skew()
                if 1.0 < abs(skewness) <= 2.0 and df[col].min() >= 0:
                    # Apply square root transformation
                    df_transformed[f'{col}_sqrt'] = np.sqrt(df[col] + 1)
        
        return df_transformed
    
    def _get_transformation_summary(self, original_df: pd.DataFrame, transformed_df: pd.DataFrame) -> List[str]:
        """Get summary of transformations applied"""
        transformations = []
        
        if len(transformed_df.columns) > len(original_df.columns):
            new_features = len(transformed_df.columns) - len(original_df.columns)
            transformations.append(f"Created {new_features} transformed features")
        
        # Check for log transformations
        log_features = [col for col in transformed_df.columns if col.endswith('_log')]
        if log_features:
            transformations.append(f"Applied log transformation to {len(log_features)} features")
        
        # Check for sqrt transformations
        sqrt_features = [col for col in transformed_df.columns if col.endswith('_sqrt')]
        if sqrt_features:
            transformations.append(f"Applied sqrt transformation to {len(sqrt_features)} features")
        
        return transformations
    
    def _create_new_features(self):
        """Create new engineered features"""
        cleaned_files = list(self.cleaned_data_dir.glob("cleaned_*.csv"))
        
        for file_path in cleaned_files:
            logger.info(f"Creating new features for: {file_path.name}")
            
            # Load cleaned data
            df = pd.read_csv(file_path)
            
            # Get transformed features from previous step
            if file_path.name in self.feature_results and 'feature_transformations' in self.feature_results[file_path.name]:
                # Use transformed data if available
                pass  # We'll work with original data for feature creation
            
            # Create new features
            df_engineered = self._create_engineered_features(df)
            
            # Store feature creation results
            if file_path.name in self.feature_results:
                self.feature_results[file_path.name]['feature_creation'] = {
                    'original_features': len(df.columns),
                    'engineered_features': len(df_engineered.columns),
                    'new_features_created': len(df_engineered.columns) - len(df.columns),
                    'engineered_feature_list': self._get_engineered_feature_list(df, df_engineered)
                }
            else:
                self.feature_results[file_path.name] = {
                    'feature_creation': {
                        'original_features': len(df.columns),
                        'engineered_features': len(df_engineered.columns),
                        'new_features_created': len(df_engineered.columns) - len(df.columns),
                        'engineered_feature_list': self._get_engineered_feature_list(df, df_engineered)
                    }
                }
    
    def _create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new engineered features"""
        df_engineered = df.copy()
        
        # 1. Packet Size Variance (squared version of std)
        if 'Packet Length Std' in df.columns:
            df_engineered['Packet_Size_Variance'] = df['Packet Length Std'] ** 2
        
        # 2. Flow Efficiency (bytes per second)
        if 'Total Length of Fwd Packets' in df.columns and 'Flow Duration' in df.columns:
            df_engineered['Flow_Efficiency'] = np.where(
                df['Flow Duration'] > 0,
                (df['Total Length of Fwd Packets'] + df.get('Total Length of Bwd Packets', 0)) / df['Flow Duration'],
                0
            )
        
        # 3. Burstiness Index
        if 'Flow Packets/s' in df.columns and 'Flow Bytes/s' in df.columns:
            df_engineered['Burstiness_Index'] = np.where(
                df['Flow Bytes/s'] > 0,
                df['Flow Packets/s'] / (df['Flow Bytes/s'] / df.get('Average Packet Size', 1)),
                0
            )
        
        # 4. Symmetry Ratio (forward/backward packet balance)
        if 'Total Fwd Packets' in df.columns and 'Total Backward Packets' in df.columns:
            total_packets = df['Total Fwd Packets'] + df['Total Backward Packets']
            df_engineered['Symmetry_Ratio'] = np.where(
                total_packets > 0,
                np.abs(df['Total Fwd Packets'] - df['Total Backward Packets']) / total_packets,
                0
            )
        
        # 5. Packet Size Ratio (max/min)
        if 'Max Packet Length' in df.columns and 'Min Packet Length' in df.columns:
            df_engineered['Packet_Size_Ratio'] = np.where(
                df['Min Packet Length'] > 0,
                df['Max Packet Length'] / df['Min Packet Length'],
                df['Max Packet Length']
            )
        
        # 6. Flow Intensity (packets per microsecond)
        if 'Flow Duration' in df.columns and 'Total Fwd Packets' in df.columns:
            df_engineered['Flow_Intensity'] = np.where(
                df['Flow Duration'] > 0,
                (df['Total Fwd Packets'] + df.get('Total Backward Packets', 0)) / df['Flow Duration'],
                0
            )
        
        # 7. Active Time Ratio
        if 'Active Mean' in df.columns and 'Idle Mean' in df.columns:
            total_time = df['Active Mean'] + df['Idle Mean']
            df_engineered['Active_Time_Ratio'] = np.where(
                total_time > 0,
                df['Active Mean'] / total_time,
                0
            )
        
        # 8. Flag Activity Score (sum of all TCP flags)
        flag_cols = ['FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 
                     'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count']
        available_flag_cols = [col for col in flag_cols if col in df.columns]
        if available_flag_cols:
            df_engineered['Flag_Activity_Score'] = df[available_flag_cols].sum(axis=1)
        
        # 9. IAT Coefficient of Variation
        if 'Flow IAT Mean' in df.columns and 'Flow IAT Std' in df.columns:
            df_engineered['IAT_CV'] = np.where(
                df['Flow IAT Mean'] > 0,
                df['Flow IAT Std'] / df['Flow IAT Mean'],
                0
            )
        
        # 10. Throughput Ratio (forward vs backward)
        if 'Total Length of Fwd Packets' in df.columns and 'Total Length of Bwd Packets' in df.columns:
            df_engineered['Throughput_Ratio'] = np.where(
                df['Total Length of Bwd Packets'] > 0,
                df['Total Length of Fwd Packets'] / df['Total Length of Bwd Packets'],
                df['Total Length of Fwd Packets']
            )
        
        return df_engineered
    
    def _get_engineered_feature_list(self, original_df: pd.DataFrame, engineered_df: pd.DataFrame) -> List[str]:
        """Get list of newly engineered features"""
        original_features = set(original_df.columns)
        engineered_features = set(engineered_df.columns)
        new_features = engineered_features - original_features
        return list(new_features)
    
    def _perform_feature_scaling(self):
        """Perform feature scaling for ML algorithms"""
        cleaned_files = list(self.cleaned_data_dir.glob("cleaned_*.csv"))
        
        for file_path in cleaned_files:
            logger.info(f"Feature scaling for: {file_path.name}")
            
            # Load cleaned data
            df = pd.read_csv(file_path)
            
            # Create engineered features
            df_engineered = self._create_engineered_features(df)
            
            # Separate numeric and categorical features
            numeric_features = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = df_engineered.select_dtypes(include=['object']).columns.tolist()
            
            # Remove Label from scaling if present
            if 'Label' in numeric_features:
                numeric_features.remove('Label')
            
            # Apply scaling
            df_scaled = df_engineered.copy()
            
            if len(numeric_features) > 0:
                # Use MinMaxScaler for autoencoder compatibility (scales to [0,1])
                scaler = MinMaxScaler()
                df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])
                
                # Store scaler for later use
                scaler_path = self.output_dir / f"scaler_{file_path.stem}.pkl"
                import pickle
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            
            # Store scaling results
            if file_path.name in self.feature_results:
                self.feature_results[file_path.name]['feature_scaling'] = {
                    'numeric_features_scaled': len(numeric_features),
                    'categorical_features_preserved': len(categorical_features),
                    'scaling_method': 'MinMaxScaler (0-1)',
                    'final_feature_count': len(df_scaled.columns)
                }
            else:
                self.feature_results[file_path.name] = {
                    'feature_scaling': {
                        'numeric_features_scaled': len(numeric_features),
                        'categorical_features_preserved': len(categorical_features),
                        'scaling_method': 'MinMaxScaler (0-1)',
                        'final_feature_count': len(df_scaled.columns)
                    }
                }
    
    def _generate_feature_summary(self):
        """Generate overall feature engineering summary"""
        total_files = len(self.feature_results)
        total_original_features = 0
        total_final_features = 0
        total_new_features_created = 0
        total_features_selected = 0
        
        for file_name, results in self.feature_results.items():
            # Count original features
            if 'original_features' in results:
                total_original_features += results['original_features']
            
            # Count final features
            if 'feature_scaling' in results:
                total_final_features += results['feature_scaling']['final_feature_count']
            
            # Count new features created
            if 'feature_creation' in results:
                total_new_features_created += results['feature_creation']['new_features_created']
            
            # Count selected features
            if 'final_selected_features' in results:
                total_features_selected += results['final_selected_features']
        
        # Initialize feature_stats first
        self.feature_stats = {
            'feature_engineering_timestamp': datetime.now().isoformat(),
            'dataset_summary': {
                'total_files_processed': total_files,
                'total_original_features': int(total_original_features),
                'total_final_features': int(total_final_features),
                'total_new_features_created': int(total_new_features_created),
                'total_features_selected': int(total_features_selected),
                'feature_reduction_ratio': round((1 - total_final_features / total_original_features) * 100, 2) if total_original_features > 0 else 0
            },
            'engineering_summary': {
                'average_features_per_file': round(total_final_features / total_files, 1) if total_files > 0 else 0,
                'feature_optimization_achieved': 'High' if total_final_features < total_original_features * 0.8 else 'Moderate',
                'scaling_applied': 'MinMaxScaler (0-1) for autoencoder compatibility',
                'feature_diversity': 'Mix of flow, statistical, timing, and engineered features'
            },
            'recommendations': []  # Will be filled by _generate_feature_recommendations
        }
        
        # Generate recommendations after feature_stats is initialized
        self.feature_stats['recommendations'] = self._generate_feature_recommendations()
    
    def _generate_feature_recommendations(self) -> List[str]:
        """Generate recommendations based on feature engineering results"""
        recommendations = []
        
        reduction_ratio = self.feature_stats['dataset_summary']['feature_reduction_ratio']
        
        if reduction_ratio > 30:
            recommendations.append("Excellent feature reduction achieved - optimal for ML training")
        elif reduction_ratio > 15:
            recommendations.append("Good feature reduction - suitable for efficient training")
        else:
            recommendations.append("Consider additional feature selection for better performance")
        
        avg_features = self.feature_stats['engineering_summary']['average_features_per_file']
        if avg_features < 50:
            recommendations.append("Optimal feature count for autoencoder training")
        elif avg_features < 100:
            recommendations.append("Reasonable feature count - should perform well")
        else:
            recommendations.append("High feature count - consider dimensionality reduction")
        
        recommendations.append("Features are scaled and ready for Phase 4: Label Processing")
        
        return recommendations
    
    def _save_engineered_features(self):
        """Save engineered features and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create engineered data directory
        engineered_dir = self.output_dir / "engineered_data"
        engineered_dir.mkdir(exist_ok=True)
        
        # Save each engineered dataset
        for file_name, results in self.feature_results.items():
            # Load cleaned data
            cleaned_path = self.cleaned_data_dir / file_name
            df = pd.read_csv(cleaned_path)
            
            # Apply all feature engineering steps
            df_engineered = self._create_engineered_features(df)
            
            # Apply scaling
            numeric_features = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
            if 'Label' in numeric_features:
                numeric_features.remove('Label')
            
            if len(numeric_features) > 0:
                scaler = MinMaxScaler()
                df_engineered[numeric_features] = scaler.fit_transform(df_engineered[numeric_features])
            
            # Save engineered data
            engineered_path = engineered_dir / f"engineered_{file_name.replace('cleaned_', '')}"
            df_engineered.to_csv(engineered_path, index=False)
            logger.info(f"Saved engineered data to: {engineered_path}")
        
        # Save feature engineering results
        results_file = self.output_dir / f"phase3_feature_engineering_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'feature_engineering_results': self.feature_results,
                'feature_engineering_statistics': self.feature_stats
            }, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / f"phase3_summary_report_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("PHASE 3: FEATURE ENGINEERING SUMMARY\n")
            f.write("=" * 45 + "\n\n")
            
            # Dataset summary
            ds_summary = self.feature_stats['dataset_summary']
            f.write(f"Dataset Summary:\n")
            f.write(f"- Files Processed: {ds_summary['total_files_processed']}\n")
            f.write(f"- Original Features: {ds_summary['total_original_features']}\n")
            f.write(f"- Final Features: {ds_summary['total_final_features']}\n")
            f.write(f"- New Features Created: {ds_summary['total_new_features_created']}\n")
            f.write(f"- Feature Reduction: {ds_summary['feature_reduction_ratio']}%\n\n")
            
            # Engineering summary
            e_summary = self.feature_stats['engineering_summary']
            f.write(f"Engineering Summary:\n")
            f.write(f"- Average Features per File: {e_summary['average_features_per_file']}\n")
            f.write(f"- Feature Optimization: {e_summary['feature_optimization_achieved']}\n")
            f.write(f"- Scaling Method: {e_summary['scaling_applied']}\n")
            f.write(f"- Feature Diversity: {e_summary['feature_diversity']}\n\n")
            
            # Recommendations
            f.write(f"Recommendations:\n")
            for i, rec in enumerate(self.feature_stats['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Summary report saved to: {summary_file}")


def main():
    """Main execution function"""
    # Define paths
    cleaned_data_dir = "data_preprocessing/cleaned_data"
    output_dir = "data_preprocessing"
    
    # Run feature engineering pipeline
    engineer = FeatureEngineeringPipeline(cleaned_data_dir, output_dir)
    results = engineer.run_complete_feature_engineering()
    
    # Print summary
    print("\n" + "="*55)
    print("PHASE 3: FEATURE ENGINEERING COMPLETE")
    print("="*55)
    print(f"Files Processed: {results['dataset_summary']['total_files_processed']}")
    print(f"Original Features: {results['dataset_summary']['total_original_features']}")
    print(f"Final Features: {results['dataset_summary']['total_final_features']}")
    print(f"New Features Created: {results['dataset_summary']['total_new_features_created']}")
    print(f"Feature Reduction: {results['dataset_summary']['feature_reduction_ratio']}%")
    print(f"Average Features per File: {results['engineering_summary']['average_features_per_file']}")
    print(f"Feature Optimization: {results['engineering_summary']['feature_optimization_achieved']}")
    print("="*55)


if __name__ == "__main__":
    main()
