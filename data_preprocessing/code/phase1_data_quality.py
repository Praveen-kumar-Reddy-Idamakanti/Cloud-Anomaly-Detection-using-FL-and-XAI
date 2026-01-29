"""
Phase 1: Data Quality Assessment
Complete data profiling and quality checks for CICIDS2017 dataset
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataQualityAssessment:
    """Comprehensive data quality assessment for network traffic data"""
    
    def __init__(self, raw_data_dir: str, output_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results storage
        self.profiling_results = {}
        self.quality_issues = {}
        self.summary_stats = {}
        
    def run_complete_assessment(self) -> Dict:
        """Run complete Phase 1 assessment"""
        logger.info("🚀 Starting Phase 1: Data Quality Assessment")
        
        # Step 1.1: Data Profiling
        logger.info("📊 Step 1.1: Data Profiling")
        self.profile_all_files()
        
        # Step 1.2: Quality Checks
        logger.info("🔍 Step 1.2: Quality Checks")
        self.perform_quality_checks()
        
        # Generate summary
        logger.info("📋 Generating Assessment Summary")
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        logger.info("✅ Phase 1 Complete: Data Quality Assessment")
        return self.summary_stats
    
    def profile_all_files(self):
        """Profile each CSV file in the dataset"""
        csv_files = list(self.raw_data_dir.glob("**/*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to analyze")
        
        for file_path in csv_files:
            logger.info(f"Profiling: {file_path.name}")
            file_profile = self._profile_single_file(file_path)
            self.profiling_results[file_path.name] = file_profile
    
    def _profile_single_file(self, file_path: Path) -> Dict:
        """Profile a single CSV file"""
        try:
            # Load sample first to check structure
            sample_df = pd.read_csv(file_path, nrows=1000)
            
            # Load full dataset
            df = pd.read_csv(file_path)
            
            profile = {
                'file_info': {
                    'file_name': file_path.name,
                    'file_size_mb': round(file_path.stat().st_size / (1024*1024), 2),
                    'total_records': len(df),
                    'total_features': len(df.columns)
                },
                'data_types': {},
                'missing_values': {},
                'feature_statistics': {},
                'label_distribution': {},
                'data_quality_score': 0
            }
            
            # Data types analysis
            profile['data_types'] = self._analyze_data_types(df)
            
            # Missing values analysis
            profile['missing_values'] = self._analyze_missing_values(df)
            
            # Feature statistics
            profile['feature_statistics'] = self._analyze_feature_statistics(df)
            
            # Label distribution (if Label column exists)
            if 'Label' in df.columns:
                profile['label_distribution'] = self._analyze_labels(df)
            
            # Calculate data quality score
            profile['data_quality_score'] = self._calculate_quality_score(profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error profiling {file_path}: {e}")
            return {'error': str(e)}
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict:
        """Analyze data types in the dataset"""
        type_counts = df.dtypes.value_counts().to_dict()
        
        # Convert numpy types to strings for JSON serialization
        type_analysis = {}
        for dtype, count in type_counts.items():
            dtype_str = str(dtype)
            type_analysis[dtype_str] = {
                'count': count,
                'percentage': round((count / len(df.columns)) * 100, 2)
            }
        
        return {
            'type_distribution': type_analysis,
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'mixed_type_features': self._find_mixed_type_features(df)
        }
    
    def _find_mixed_type_features(self, df: pd.DataFrame) -> List[str]:
        """Find features that might have mixed data types"""
        mixed_features = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric values are stored as strings
                try:
                    pd.to_numeric(df[col].dropna())
                    mixed_features.append(col)
                except:
                    pass
        return mixed_features
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values in the dataset"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_analysis = {
            'total_missing_values': int(missing_counts.sum()),
            'total_missing_percentage': round((missing_counts.sum() / (len(df) * len(df.columns))) * 100, 4),
            'features_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentage_by_feature': missing_percentages[missing_percentages > 0].round(2).to_dict(),
            'high_missing_features': missing_percentages[missing_percentages > 20].to_dict()
        }
        
        return missing_analysis
    
    def _analyze_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """Analyze statistical properties of features"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            return {'error': 'No numeric features found'}
        
        stats = {
            'numeric_features_summary': {},
            'potential_issues': {}
        }
        
        for col in numeric_df.columns:
            col_stats = {
                'mean': float(numeric_df[col].mean()),
                'std': float(numeric_df[col].std()),
                'min': float(numeric_df[col].min()),
                'max': float(numeric_df[col].max()),
                'q25': float(numeric_df[col].quantile(0.25)),
                'q50': float(numeric_df[col].quantile(0.50)),
                'q75': float(numeric_df[col].quantile(0.75)),
                'skewness': float(numeric_df[col].skew()),
                'kurtosis': float(numeric_df[col].kurtosis()),
                'zero_count': int((numeric_df[col] == 0).sum()),
                'negative_count': int((numeric_df[col] < 0).sum()),
                'unique_values': int(numeric_df[col].nunique())
            }
            stats['numeric_features_summary'][col] = col_stats
        
        # Identify potential issues
        stats['potential_issues'] = self._identify_feature_issues(numeric_df)
        
        return stats
    
    def _identify_feature_issues(self, df: pd.DataFrame) -> Dict:
        """Identify potential issues in features"""
        issues = {
            'constant_features': [],
            'highly_skewed_features': [],
            'features_with_outliers': [],
            'features_with_negative_values': []
        }
        
        for col in df.columns:
            # Constant features (zero variance)
            if df[col].std() == 0:
                issues['constant_features'].append(col)
            
            # Highly skewed features
            skewness = df[col].skew()
            if abs(skewness) > 2:
                issues['highly_skewed_features'].append({
                    'feature': col,
                    'skewness': float(skewness)
                })
            
            # Features with outliers (using IQR method)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                issues['features_with_outliers'].append({
                    'feature': col,
                    'outlier_count': len(outliers),
                    'outlier_percentage': round((len(outliers) / len(df)) * 100, 2)
                })
            
            # Features with negative values (that shouldn't be negative)
            if col in ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
                       'Total Length of Fwd Packets', 'Total Length of Bwd Packets']:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues['features_with_negative_values'].append({
                        'feature': col,
                        'negative_count': int(negative_count)
                    })
        
        return issues
    
    def _analyze_labels(self, df: pd.DataFrame) -> Dict:
        """Analyze label distribution"""
        label_counts = df['Label'].value_counts()
        label_percentages = (label_counts / len(df)) * 100
        
        # Separate benign and attack labels
        benign_count = label_counts.get('BENIGN', 0)
        attack_count = len(df) - benign_count
        
        # Identify unique attack types
        attack_types = label_counts[label_counts.index != 'BENIGN'].to_dict()
        
        return {
            'total_records': len(df),
            'benign_count': int(benign_count),
            'benign_percentage': round((benign_count / len(df)) * 100, 2),
            'attack_count': int(attack_count),
            'attack_percentage': round((attack_count / len(df)) * 100, 2),
            'unique_attack_types': len(attack_types),
            'attack_type_distribution': {k: int(v) for k, v in attack_types.items()},
            'class_balance_ratio': round(benign_count / max(attack_count, 1), 2)
        }
    
    def _calculate_quality_score(self, profile: Dict) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100
        
        # Deduct points for missing values
        missing_pct = profile['missing_values']['total_missing_percentage']
        score -= min(missing_pct * 2, 20)  # Max 20 points deduction
        
        # Deduct points for high missing features
        high_missing_count = len(profile['missing_values']['high_missing_features'])
        score -= min(high_missing_count * 5, 15)  # Max 15 points deduction
        
        # Deduct points for potential issues
        if 'feature_statistics' in profile and 'potential_issues' in profile['feature_statistics']:
            issues = profile['feature_statistics']['potential_issues']
            score -= min(len(issues['constant_features']) * 3, 10)
            score -= min(len(issues['features_with_negative_values']) * 2, 10)
        
        return max(0, round(score, 2))
    
    def perform_quality_checks(self):
        """Perform comprehensive quality checks"""
        for file_name, profile in self.profiling_results.items():
            if 'error' in profile:
                continue
                
            logger.info(f"Quality checking: {file_name}")
            file_issues = self._check_file_quality(file_name, profile)
            self.quality_issues[file_name] = file_issues
    
    def _check_file_quality(self, file_name: str, profile: Dict) -> Dict:
        """Check quality issues for a specific file"""
        issues = {
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check missing values
        missing_pct = profile['missing_values']['total_missing_percentage']
        if missing_pct > 10:
            issues['critical_issues'].append(f"High missing value rate: {missing_pct}%")
        elif missing_pct > 5:
            issues['warnings'].append(f"Moderate missing value rate: {missing_pct}%")
        
        # Check label imbalance
        if 'label_distribution' in profile:
            label_dist = profile['label_distribution']
            attack_pct = label_dist.get('attack_percentage', 0)
            if attack_pct < 1:
                issues['warnings'].append(f"Very low anomaly rate: {attack_pct}%")
            elif attack_pct > 50:
                issues['warnings'].append(f"High anomaly rate: {attack_pct}%")
        
        # Check feature issues
        if 'feature_statistics' in profile and 'potential_issues' in profile['feature_statistics']:
            potential_issues = profile['feature_statistics']['potential_issues']
            
            if len(potential_issues['constant_features']) > 0:
                issues['warnings'].append(f"Found {len(potential_issues['constant_features'])} constant features")
            
            if len(potential_issues['features_with_negative_values']) > 0:
                issues['critical_issues'].append(f"Found {len(potential_issues['features_with_negative_values'])} features with invalid negative values")
        
        # Generate recommendations
        if issues['critical_issues']:
            issues['recommendations'].append("Address critical issues before proceeding to preprocessing")
        if issues['warnings']:
            issues['recommendations'].append("Review warnings and consider mitigation strategies")
        
        return issues
    
    def generate_summary(self):
        """Generate overall assessment summary"""
        total_files = len(self.profiling_results)
        total_records = sum(p.get('file_info', {}).get('total_records', 0) for p in self.profiling_results.values())
        
        # Aggregate quality scores
        quality_scores = [p.get('data_quality_score', 0) for p in self.profiling_results.values() if 'data_quality_score' in p]
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0
        
        # Count issues
        total_critical = sum(len(q.get('critical_issues', [])) for q in self.quality_issues.values())
        total_warnings = sum(len(q.get('warnings', [])) for q in self.quality_issues.values())
        
        # Initialize summary_stats first
        self.summary_stats = {
            'assessment_timestamp': datetime.now().isoformat(),
            'dataset_summary': {
                'total_files_analyzed': total_files,
                'total_records': int(total_records),
                'total_features': max(p.get('file_info', {}).get('total_features', 0) for p in self.profiling_results.values()),
                'average_quality_score': round(avg_quality_score, 2)
            },
            'quality_summary': {
                'files_with_critical_issues': len([q for q in self.quality_issues.values() if q.get('critical_issues')]),
                'total_critical_issues': total_critical,
                'total_warnings': total_warnings,
                'overall_data_quality': 'Good' if avg_quality_score > 80 else 'Fair' if avg_quality_score > 60 else 'Poor'
            },
            'recommendations': []  # Will be filled by _generate_overall_recommendations
        }
        
        # Generate recommendations after summary_stats is initialized
        self.summary_stats['recommendations'] = self._generate_overall_recommendations()
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations based on assessment"""
        recommendations = []
        
        # Check overall quality
        avg_score = self.summary_stats['dataset_summary']['average_quality_score']
        
        if avg_score < 70:
            recommendations.append("Dataset has significant quality issues - consider thorough cleaning")
        elif avg_score < 85:
            recommendations.append("Dataset has moderate quality issues - address before preprocessing")
        
        # Check specific issues
        if self.summary_stats['quality_summary']['total_critical_issues'] > 0:
            recommendations.append("Address critical data quality issues immediately")
        
        # Check label distribution
        label_distributions = [p.get('label_distribution', {}) for p in self.profiling_results.values() if 'label_distribution' in p]
        if label_distributions:
            avg_anomaly_rate = np.mean([ld.get('attack_percentage', 0) for ld in label_distributions])
            if avg_anomaly_rate < 2:
                recommendations.append("Consider data augmentation for better anomaly representation")
            elif avg_anomaly_rate > 30:
                recommendations.append("Verify attack labels - unusually high anomaly rate")
        
        recommendations.append("Proceed to Phase 2: Data Cleaning after addressing critical issues")
        
        return recommendations
    
    def save_results(self):
        """Save assessment results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = self.output_dir / f"phase1_assessment_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'profiling_results': self.profiling_results,
                'quality_issues': self.quality_issues,
                'summary_statistics': self.summary_stats
            }, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / f"phase1_summary_report_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("PHASE 1: DATA QUALITY ASSESSMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Dataset summary
            ds_summary = self.summary_stats['dataset_summary']
            f.write(f"Dataset Summary:\n")
            f.write(f"- Files Analyzed: {ds_summary['total_files_analyzed']}\n")
            f.write(f"- Total Records: {ds_summary['total_records']:,}\n")
            f.write(f"- Features per File: {ds_summary['total_features']}\n")
            f.write(f"- Average Quality Score: {ds_summary['average_quality_score']}/100\n\n")
            
            # Quality summary
            q_summary = self.summary_stats['quality_summary']
            f.write(f"Quality Summary:\n")
            f.write(f"- Overall Quality: {q_summary['overall_data_quality']}\n")
            f.write(f"- Files with Critical Issues: {q_summary['files_with_critical_issues']}\n")
            f.write(f"- Total Critical Issues: {q_summary['total_critical_issues']}\n")
            f.write(f"- Total Warnings: {q_summary['total_warnings']}\n\n")
            
            # Recommendations
            f.write(f"Recommendations:\n")
            for i, rec in enumerate(self.summary_stats['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Summary report saved to: {summary_file}")


def main():
    """Main execution function"""
    # Define paths
    raw_data_dir = "data/raw/CICIDS2017/MachineLearningCVE"
    output_dir = "data_preprocessing/results"
    
    # Run assessment
    assessor = DataQualityAssessment(raw_data_dir, output_dir)
    results = assessor.run_complete_assessment()
    
    # Print summary
    print("\n" + "="*60)
    print("PHASE 1: DATA QUALITY ASSESSMENT COMPLETE")
    print("="*60)
    print(f"Files Analyzed: {results['dataset_summary']['total_files_analyzed']}")
    print(f"Total Records: {results['dataset_summary']['total_records']:,}")
    print(f"Average Quality Score: {results['dataset_summary']['average_quality_score']}/100")
    print(f"Overall Quality: {results['quality_summary']['overall_data_quality']}")
    print(f"Critical Issues: {results['quality_summary']['total_critical_issues']}")
    print(f"Warnings: {results['quality_summary']['total_warnings']}")
    print("="*60)


if __name__ == "__main__":
    main()
