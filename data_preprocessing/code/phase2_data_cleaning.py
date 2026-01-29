"""
Phase 2: Data Cleaning
Comprehensive data cleaning for CICIDS2017 dataset based on Phase 1 assessment
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from sklearn.impute import SimpleImputer
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DataCleaningPipeline:
    """Comprehensive data cleaning pipeline for network traffic data"""
    
    def __init__(self, raw_data_dir: str, output_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cleaning statistics
        self.cleaning_stats = {}
        self.cleaning_results = {}
        
        # Define network flow features that should not have negative values
        self.nonnegative_features = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Min',
            'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
            'min_seg_size_forward', 'Active Mean', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Max', 'Idle Min'
        ]
        
        # Define port ranges for validation
        self.valid_port_range = (0, 65535)
        
    def run_complete_cleaning(self) -> Dict:
        """Run complete Phase 2 cleaning pipeline"""
        logger.info("🚀 Starting Phase 2: Data Cleaning")
        
        # Load Phase 1 results to understand data characteristics
        phase1_results = self._load_phase1_results()
        
        # Step 2.1: Handle Missing Values
        logger.info("🧹 Step 2.1: Handle Missing Values")
        self._handle_missing_values()
        
        # Step 2.2: Remove Invalid Records
        logger.info("🔍 Step 2.2: Remove Invalid Records")
        self._remove_invalid_records()
        
        # Step 2.3: Remove Duplicates
        logger.info("🔄 Step 2.3: Remove Duplicates")
        self._remove_duplicates()
        
        # Step 2.4: Data Type Corrections
        logger.info("🔧 Step 2.4: Data Type Corrections")
        self._correct_data_types()
        
        # Generate cleaning summary
        logger.info("📋 Generating Cleaning Summary")
        self._generate_cleaning_summary()
        
        # Save results
        self._save_cleaned_data()
        
        logger.info("✅ Phase 2 Complete: Data Cleaning")
        return self.cleaning_stats
    
    def _load_phase1_results(self) -> Dict:
        """Load Phase 1 assessment results"""
        results_dir = self.output_dir.parent / "results"
        phase1_files = list(results_dir.glob("phase1_assessment_results_*.json"))
        
        if not phase1_files:
            logger.warning("No Phase 1 results found, proceeding with default cleaning")
            return {}
        
        latest_file = max(phase1_files, key=lambda x: x.name)
        with open(latest_file, 'r') as f:
            return json.load(f)
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        csv_files = list(self.raw_data_dir.glob("**/*.csv"))
        
        for file_path in csv_files:
            logger.info(f"Processing missing values for: {file_path.name}")
            
            # Load data
            df = pd.read_csv(file_path)
            original_shape = df.shape
            
            # Analyze missing values
            missing_analysis = self._analyze_missing_values(df)
            
            # Handle missing values based on analysis
            cleaned_df = self._clean_missing_values(df, missing_analysis)
            
            # Store results
            self.cleaning_results[file_path.name] = {
                'original_shape': original_shape,
                'cleaned_shape': cleaned_df.shape,
                'missing_analysis': missing_analysis,
                'cleaning_actions': self._get_missing_value_actions(missing_analysis)
            }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """Analyze missing values in detail"""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        # Categorize features by missing percentage
        high_missing = missing_percentages[missing_percentages > 20].index.tolist()
        moderate_missing = missing_percentages[(missing_percentages > 5) & (missing_percentages <= 20)].index.tolist()
        low_missing = missing_percentages[(missing_percentages > 0) & (missing_percentages <= 5)].index.tolist()
        
        return {
            'total_missing': missing_counts.sum(),
            'total_missing_percentage': (missing_counts.sum() / (len(df) * len(df.columns))) * 100,
            'high_missing_features': high_missing,
            'moderate_missing_features': moderate_missing,
            'low_missing_features': low_missing,
            'missing_by_feature': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentage_by_feature': missing_percentages[missing_percentages > 0].round(2).to_dict()
        }
    
    def _clean_missing_values(self, df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """Clean missing values based on analysis"""
        cleaned_df = df.copy()
        
        # Remove features with >50% missing values
        if analysis['high_missing_features']:
            logger.warning(f"Removing high missing features: {analysis['high_missing_features']}")
            cleaned_df = cleaned_df.drop(columns=analysis['high_missing_features'])
        
        # Remove records with >20% missing features
        missing_threshold = len(cleaned_df.columns) * 0.2
        cleaned_df = cleaned_df.dropna(thresh=len(cleaned_df.columns) - missing_threshold)
        
        # Impute remaining missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
        
        if len(numeric_columns) > 0:
            # Handle infinite values first
            cleaned_df[numeric_columns] = cleaned_df[numeric_columns].replace([np.inf, -np.inf], np.nan)
            
            # Use median imputation for numeric features
            imputer = SimpleImputer(strategy='median')
            cleaned_df[numeric_columns] = imputer.fit_transform(cleaned_df[numeric_columns])
        
        if len(categorical_columns) > 0:
            # Use mode imputation for categorical features
            for col in categorical_columns:
                if cleaned_df[col].isnull().sum() > 0:
                    mode_value = cleaned_df[col].mode()[0]
                    cleaned_df[col].fillna(mode_value, inplace=True)
        
        return cleaned_df
    
    def _get_missing_value_actions(self, analysis: Dict) -> List[str]:
        """Get summary of missing value cleaning actions"""
        actions = []
        
        if analysis['high_missing_features']:
            actions.append(f"Removed {len(analysis['high_missing_features'])} high-missing features")
        
        if analysis['total_missing_percentage'] > 0:
            actions.append(f"Imputed {analysis['total_missing']} missing values")
        
        if analysis['total_missing_percentage'] == 0:
            actions.append("No missing value treatment needed")
        
        return actions
    
    def _remove_invalid_records(self):
        """Remove records with invalid values"""
        csv_files = list(self.raw_data_dir.glob("**/*.csv"))
        
        for file_path in csv_files:
            logger.info(f"Removing invalid records from: {file_path.name}")
            
            # Get the cleaned data from previous step
            if file_path.name in self.cleaning_results:
                df = pd.read_csv(file_path)
                # Apply missing value cleaning first
                missing_analysis = self.cleaning_results[file_path.name]['missing_analysis']
                df = self._clean_missing_values(df, missing_analysis)
            else:
                df = pd.read_csv(file_path)
            
            original_shape = df.shape
            
            # Remove invalid records
            cleaned_df = self._clean_invalid_records(df)
            
            # Update results
            if file_path.name in self.cleaning_results:
                self.cleaning_results[file_path.name]['invalid_records_removed'] = {
                    'original_shape': original_shape,
                    'cleaned_shape': cleaned_df.shape,
                    'records_removed': original_shape[0] - cleaned_df.shape[0]
                }
            else:
                self.cleaning_results[file_path.name] = {
                    'invalid_records_removed': {
                        'original_shape': original_shape,
                        'cleaned_shape': cleaned_df.shape,
                        'records_removed': original_shape[0] - cleaned_df.shape[0]
                    }
                }
    
    def _clean_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean records with invalid values"""
        cleaned_df = df.copy()
        records_removed = 0
        
        # Remove records with negative values in non-negative features
        for feature in self.nonnegative_features:
            if feature in cleaned_df.columns:
                invalid_mask = cleaned_df[feature] < 0
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    logger.info(f"Removing {invalid_count} records with negative {feature}")
                    cleaned_df = cleaned_df[~invalid_mask]
                    records_removed += invalid_count
        
        # Validate port numbers
        port_features = ['Destination Port']
        for feature in port_features:
            if feature in cleaned_df.columns:
                invalid_mask = (cleaned_df[feature] < self.valid_port_range[0]) | \
                             (cleaned_df[feature] > self.valid_port_range[1])
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    logger.info(f"Removing {invalid_count} records with invalid {feature}")
                    cleaned_df = cleaned_df[~invalid_mask]
                    records_removed += invalid_count
        
        # Remove records with zero flow duration but non-zero packets (inconsistent)
        if 'Flow Duration' in cleaned_df.columns and 'Total Fwd Packets' in cleaned_df.columns:
            inconsistent_mask = (cleaned_df['Flow Duration'] == 0) & \
                              (cleaned_df['Total Fwd Packets'] + cleaned_df.get('Total Backward Packets', 0) > 0)
            inconsistent_count = inconsistent_mask.sum()
            if inconsistent_count > 0:
                logger.info(f"Removing {inconsistent_count} records with inconsistent flow data")
                cleaned_df = cleaned_df[~inconsistent_mask]
                records_removed += inconsistent_count
        
        return cleaned_df
    
    def _remove_duplicates(self):
        """Remove duplicate records"""
        csv_files = list(self.raw_data_dir.glob("**/*.csv"))
        
        for file_path in csv_files:
            logger.info(f"Removing duplicates from: {file_path.name}")
            
            # Load and clean data
            df = pd.read_csv(file_path)
            
            # Apply previous cleaning steps
            if file_path.name in self.cleaning_results:
                if 'missing_analysis' in self.cleaning_results[file_path.name]:
                    df = self._clean_missing_values(df, self.cleaning_results[file_path.name]['missing_analysis'])
                df = self._clean_invalid_records(df)
            
            original_shape = df.shape
            
            # Remove exact duplicates
            exact_duplicates = df.duplicated().sum()
            df_cleaned = df.drop_duplicates()
            
            # Remove near-duplicates (same key features)
            key_features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets']
            available_key_features = [f for f in key_features if f in df_cleaned.columns]
            
            if len(available_key_features) >= 3:
                near_duplicates = df_cleaned.duplicated(subset=available_key_features).sum()
                df_cleaned = df_cleaned.drop_duplicates(subset=available_key_features, keep='first')
            else:
                near_duplicates = 0
            
            # Update results
            if file_path.name in self.cleaning_results:
                self.cleaning_results[file_path.name]['duplicates_removed'] = {
                    'original_shape': original_shape,
                    'cleaned_shape': df_cleaned.shape,
                    'exact_duplicates_removed': int(exact_duplicates),
                    'near_duplicates_removed': int(near_duplicates),
                    'total_duplicates_removed': int(exact_duplicates + near_duplicates)
                }
            else:
                self.cleaning_results[file_path.name] = {
                    'duplicates_removed': {
                        'original_shape': original_shape,
                        'cleaned_shape': df_cleaned.shape,
                        'exact_duplicates_removed': int(exact_duplicates),
                        'near_duplicates_removed': int(near_duplicates),
                        'total_duplicates_removed': int(exact_duplicates + near_duplicates)
                    }
                }
    
    def _correct_data_types(self):
        """Correct data types in the dataset"""
        csv_files = list(self.raw_data_dir.glob("**/*.csv"))
        
        for file_path in csv_files:
            logger.info(f"Correcting data types for: {file_path.name}")
            
            # Load and clean data
            df = pd.read_csv(file_path)
            
            # Apply all previous cleaning steps
            if file_path.name in self.cleaning_results:
                if 'missing_analysis' in self.cleaning_results[file_path.name]:
                    df = self._clean_missing_values(df, self.cleaning_results[file_path.name]['missing_analysis'])
                df = self._clean_invalid_records(df)
                df = df.drop_duplicates()
            
            original_dtypes = df.dtypes.to_dict()
            
            # Correct data types
            df_cleaned = self._fix_data_types(df)
            
            # Update results
            if file_path.name in self.cleaning_results:
                self.cleaning_results[file_path.name]['data_type_corrections'] = {
                    'original_dtypes': {k: str(v) for k, v in original_dtypes.items()},
                    'corrected_dtypes': {k: str(v) for k, v in df_cleaned.dtypes.to_dict().items()},
                    'corrections_made': self._get_dtype_corrections(original_dtypes, df_cleaned.dtypes.to_dict())
                }
            else:
                self.cleaning_results[file_path.name] = {
                    'data_type_corrections': {
                        'original_dtypes': {k: str(v) for k, v in original_dtypes.items()},
                        'corrected_dtypes': {k: str(v) for k, v in df_cleaned.dtypes.to_dict().items()},
                        'corrections_made': self._get_dtype_corrections(original_dtypes, df_cleaned.dtypes.to_dict())
                    }
                }
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types in the dataframe"""
        df_cleaned = df.copy()
        
        # Convert numeric columns that might be stored as strings
        for col in df_cleaned.columns:
            if col == 'Label':  # Keep label as object
                continue
                
            # Try to convert to numeric if it's object type
            if df_cleaned[col].dtype == 'object':
                try:
                    # Remove common non-numeric characters
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col].astype(str).str.replace(',', '').str.replace(' ', ''), errors='coerce')
                except:
                    pass
            
            # Convert to appropriate numeric type
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                # Use float64 for precision, but could use int32 for integers
                if df_cleaned[col].dropna().apply(lambda x: x.is_integer()).all():
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], downcast='integer')
                else:
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], downcast='float')
        
        return df_cleaned
    
    def _get_dtype_corrections(self, original: Dict, corrected: Dict) -> List[str]:
        """Get summary of data type corrections made"""
        corrections = []
        
        for col in original:
            if col in corrected and original[col] != corrected[col]:
                corrections.append(f"{col}: {original[col]} → {corrected[col]}")
        
        return corrections
    
    def _generate_cleaning_summary(self):
        """Generate overall cleaning summary"""
        total_files = len(self.cleaning_results)
        total_original_records = 0
        total_cleaned_records = 0
        total_missing_values_handled = 0
        total_invalid_records_removed = 0
        total_duplicates_removed = 0
        total_dtype_corrections = 0
        
        for file_name, results in self.cleaning_results.items():
            # Get original and cleaned record counts
            if 'invalid_records_removed' in results:
                total_original_records += results['invalid_records_removed']['original_shape'][0]
                total_cleaned_records += results['invalid_records_removed']['cleaned_shape'][0]
            elif 'duplicates_removed' in results:
                total_original_records += results['duplicates_removed']['original_shape'][0]
                total_cleaned_records += results['duplicates_removed']['cleaned_shape'][0]
            
            # Count missing values handled
            if 'missing_analysis' in results:
                total_missing_values_handled += results['missing_analysis']['total_missing']
            
            # Count invalid records removed
            if 'invalid_records_removed' in results:
                total_invalid_records_removed += results['invalid_records_removed']['records_removed']
            
            # Count duplicates removed
            if 'duplicates_removed' in results:
                total_duplicates_removed += results['duplicates_removed']['total_duplicates_removed']
            
            # Count data type corrections
            if 'data_type_corrections' in results:
                total_dtype_corrections += len(results['data_type_corrections']['corrections_made'])
        
        # Initialize cleaning_stats first
        self.cleaning_stats = {
            'cleaning_timestamp': datetime.now().isoformat(),
            'dataset_summary': {
                'total_files_processed': total_files,
                'total_original_records': int(total_original_records),
                'total_cleaned_records': int(total_cleaned_records),
                'records_retained_percentage': round((total_cleaned_records / total_original_records) * 100, 2) if total_original_records > 0 else 0
            },
            'cleaning_summary': {
                'total_missing_values_handled': int(total_missing_values_handled),
                'total_invalid_records_removed': int(total_invalid_records_removed),
                'total_duplicates_removed': int(total_duplicates_removed),
                'total_data_type_corrections': int(total_dtype_corrections),
                'overall_data_quality_improvement': 'Significant' if total_invalid_records_removed + total_duplicates_removed > total_original_records * 0.05 else 'Moderate'
            },
            'recommendations': []  # Will be filled by _generate_cleaning_recommendations
        }
        
        # Generate recommendations after cleaning_stats is initialized
        self.cleaning_stats['recommendations'] = self._generate_cleaning_recommendations()
    
    def _generate_cleaning_recommendations(self) -> List[str]:
        """Generate recommendations based on cleaning results"""
        recommendations = []
        
        retention_rate = self.cleaning_stats['dataset_summary']['records_retained_percentage']
        
        if retention_rate < 80:
            recommendations.append("Low data retention rate - review cleaning criteria")
        elif retention_rate > 95:
            recommendations.append("High data retention - good data quality")
        
        if self.cleaning_stats['cleaning_summary']['total_invalid_records_removed'] > 0:
            recommendations.append("Invalid records successfully removed")
        
        if self.cleaning_stats['cleaning_summary']['total_duplicates_removed'] > 0:
            recommendations.append("Duplicate records successfully removed")
        
        recommendations.append("Data is now ready for Phase 3: Feature Engineering")
        
        return recommendations
    
    def _save_cleaned_data(self):
        """Save cleaned data and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create cleaned data directory
        cleaned_dir = self.output_dir / "cleaned_data"
        cleaned_dir.mkdir(exist_ok=True)
        
        # Save each cleaned dataset
        for file_name, results in self.cleaning_results.items():
            # Load original data
            original_path = self.raw_data_dir / file_name
            df = pd.read_csv(original_path)
            
            # Apply all cleaning steps
            if 'missing_analysis' in results:
                df = self._clean_missing_values(df, results['missing_analysis'])
            df = self._clean_invalid_records(df)
            df = df.drop_duplicates()
            df = self._fix_data_types(df)
            
            # Save cleaned data
            cleaned_path = cleaned_dir / f"cleaned_{file_name}"
            df.to_csv(cleaned_path, index=False)
            logger.info(f"Saved cleaned data to: {cleaned_path}")
        
        # Save cleaning results
        results_file = self.output_dir / f"phase2_cleaning_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'cleaning_results': self.cleaning_results,
                'cleaning_statistics': self.cleaning_stats
            }, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / f"phase2_summary_report_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("PHASE 2: DATA CLEANING SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            # Dataset summary
            ds_summary = self.cleaning_stats['dataset_summary']
            f.write(f"Dataset Summary:\n")
            f.write(f"- Files Processed: {ds_summary['total_files_processed']}\n")
            f.write(f"- Original Records: {ds_summary['total_original_records']:,}\n")
            f.write(f"- Cleaned Records: {ds_summary['total_cleaned_records']:,}\n")
            f.write(f"- Retention Rate: {ds_summary['records_retained_percentage']}%\n\n")
            
            # Cleaning summary
            c_summary = self.cleaning_stats['cleaning_summary']
            f.write(f"Cleaning Summary:\n")
            f.write(f"- Missing Values Handled: {c_summary['total_missing_values_handled']:,}\n")
            f.write(f"- Invalid Records Removed: {c_summary['total_invalid_records_removed']:,}\n")
            f.write(f"- Duplicates Removed: {c_summary['total_duplicates_removed']:,}\n")
            f.write(f"- Data Type Corrections: {c_summary['total_data_type_corrections']}\n")
            f.write(f"- Quality Improvement: {c_summary['overall_data_quality_improvement']}\n\n")
            
            # Recommendations
            f.write(f"Recommendations:\n")
            for i, rec in enumerate(self.cleaning_stats['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Summary report saved to: {summary_file}")


def main():
    """Main execution function"""
    # Define paths
    raw_data_dir = "data/raw/CICIDS2017/MachineLearningCVE"
    output_dir = "data_preprocessing"
    
    # Run cleaning pipeline
    cleaner = DataCleaningPipeline(raw_data_dir, output_dir)
    results = cleaner.run_complete_cleaning()
    
    # Print summary
    print("\n" + "="*50)
    print("PHASE 2: DATA CLEANING COMPLETE")
    print("="*50)
    print(f"Files Processed: {results['dataset_summary']['total_files_processed']}")
    print(f"Original Records: {results['dataset_summary']['total_original_records']:,}")
    print(f"Cleaned Records: {results['dataset_summary']['total_cleaned_records']:,}")
    print(f"Retention Rate: {results['dataset_summary']['records_retained_percentage']}%")
    print(f"Missing Values Handled: {results['cleaning_summary']['total_missing_values_handled']:,}")
    print(f"Invalid Records Removed: {results['cleaning_summary']['total_invalid_records_removed']:,}")
    print(f"Duplicates Removed: {results['cleaning_summary']['total_duplicates_removed']:,}")
    print(f"Quality Improvement: {results['cleaning_summary']['overall_data_quality_improvement']}")
    print("="*50)


if __name__ == "__main__":
    main()
