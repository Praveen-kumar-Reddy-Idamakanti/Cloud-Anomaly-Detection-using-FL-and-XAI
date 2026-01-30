"""
Phase 4: Label Processing
Comprehensive label processing for CICIDS2017 dataset - binary classification for anomaly detection
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class LabelProcessingPipeline:
    """Comprehensive label processing pipeline for network traffic data"""
    
    def __init__(self, raw_data_dir: str, cleaned_data_dir: str, engineered_data_dir: str, output_dir: str):
        self.raw_data_dir = Path(raw_data_dir)
        self.cleaned_data_dir = Path(cleaned_data_dir)
        self.engineered_data_dir = Path(engineered_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize label processing statistics
        self.label_stats = {}
        self.label_results = {}
        
        # Define attack categories for CICIDS2017
        self.attack_mapping = {
            # Normal traffic
            'BENIGN': 'Normal',
            
            # DoS/DDoS attacks
            'DDoS': 'DoS',
            'DoS': 'DoS',
            'DDoS attacks-LOIC-HTTP': 'DoS',
            'DDOS attack-LOIC-UDP': 'DoS',
            'DDoS attack-HOIC': 'DoS',
            
            # Port Scanning
            'PortScan': 'PortScan',
            'Port Scan': 'PortScan',
            
            # Web Attacks
            'Web Attack - Brute Force': 'WebAttack',
            'Web Attack - Sql Injection': 'WebAttack',
            'Web Attack - XSS': 'WebAttack',
            'Web Attack': 'WebAttack',
            
            # Infiltration
            'Infiltration': 'Infiltration',
            'Infiltrating': 'Infiltration',
            
            # Botnet
            'Bot': 'Botnet',
            'Botnet': 'Botnet',
            
            # Brute Force
            'FTP-BruteForce': 'BruteForce',
            'SSH-Bruteforce': 'BruteForce',
            'Brute Force': 'BruteForce',
            'Brute Force - Web': 'BruteForce',
            'Brute Force - XSS': 'BruteForce',
            'Brute Force - Sql Injection': 'BruteForce',
            
            # Heartbleed
            'Heartbleed': 'Heartbleed',
            
            # Other attacks
            'DoS attacks-GoldenEye': 'DoS',
            'DoS attacks-Slowloris': 'DoS',
            'DoS attacks-SlowHTTPTest': 'DoS',
            'DoS attacks-Hulk': 'DoS',
        }
        
        # Binary mapping for anomaly detection
        self.binary_mapping = {
            'Normal': 0,      # BENIGN traffic
            'Anomaly': 1      # All attack types
        }
        
    def run_complete_label_processing(self) -> Dict:
        """Run complete Phase 4 label processing pipeline"""
        logger.info("🚀 Starting Phase 4: Label Processing")
        
        # Step 4.1: Binary Classification
        logger.info("🏷️ Step 4.1: Binary Classification")
        self._perform_binary_classification()
        
        # Step 4.2: Label Validation
        logger.info("✅ Step 4.2: Label Validation")
        self._validate_labels()
        
        # Step 4.3: Class Balance Analysis
        logger.info("⚖️ Step 4.3: Class Balance Analysis")
        self._analyze_class_balance()
        
        # Step 4.4: Final Dataset Preparation
        logger.info("📦 Step 4.4: Final Dataset Preparation")
        self._prepare_final_datasets()
        
        # Step 4.5: Create Attack Type Lookup Functions
        logger.info("🔍 Step 4.5: Creating Attack Type Lookup Functions")
        lookup_results = self.create_attack_type_lookup_functions()
        
        # Generate label processing summary
        logger.info("📋 Generating Label Processing Summary")
        self._generate_label_summary()
        
        # Save processed datasets
        self._save_processed_datasets()
        
        logger.info("✅ Phase 4 Complete: Label Processing")
        return self.label_stats
    
    def _perform_binary_classification(self):
        """Convert multi-class labels to binary classification with attack categories"""
        cleaned_files = list(self.cleaned_data_dir.glob("cleaned_*.csv"))
        
        for cleaned_file in cleaned_files:
            logger.info(f"Processing labels for: {cleaned_file.name}")
            
            # Load cleaned data
            cleaned_df = pd.read_csv(cleaned_file)
            
            # Find corresponding raw file to get labels
            raw_filename = cleaned_file.name.replace("cleaned_", "")
            raw_file = self.raw_data_dir / raw_filename
            
            if not raw_file.exists():
                logger.warning(f"Raw file not found: {raw_filename}")
                continue
            
            # Load raw data to get labels
            raw_df = pd.read_csv(raw_file)
            
            # Debug: Print column names
            logger.info(f"Raw file columns: {list(raw_df.columns)}")
            
            # Check if Label column exists in raw data (case insensitive)
            label_column = None
            for col in raw_df.columns:
                if 'label' in col.lower():
                    label_column = col
                    break
            
            if label_column is None:
                logger.warning(f"No 'Label' column found in {raw_filename}")
                continue
            
            logger.info(f"Using label column: {label_column}")
            
            # Extract labels from raw data
            raw_labels = raw_df[label_column].copy()
            
            # Ensure the cleaned data has same number of rows as raw data
            if len(cleaned_df) != len(raw_labels):
                logger.warning(f"Row count mismatch: cleaned={len(cleaned_df)}, raw={len(raw_labels)}")
                # Truncate to minimum length
                min_length = min(len(cleaned_df), len(raw_labels))
                cleaned_df = cleaned_df.iloc[:min_length]
                raw_labels = raw_labels.iloc[:min_length]
            
            # Get original label distribution
            original_labels = raw_labels.value_counts().to_dict()
            
            # Map to attack categories
            attack_categories = raw_labels.map(self.attack_mapping)
            
            # Handle unmapped labels
            unmapped = attack_categories[attack_categories.isna()]
            if not unmapped.empty:
                logger.warning(f"Found {len(unmapped)} unmapped labels in {cleaned_file.name}")
                # Map unmapped to 'Other' category
                attack_categories = attack_categories.fillna('Other')
            
            # Create binary labels for anomaly detection
            binary_labels = attack_categories.apply(lambda x: 0 if x == 'Normal' else 1)
            
            # Create numeric attack category labels for multi-class classification
            category_encoder = LabelEncoder()
            category_numeric = category_encoder.fit_transform(attack_categories)
            
            # Create attack type labels (original labels) for detailed analysis
            attack_type_encoder = LabelEncoder()
            attack_type_numeric = attack_type_encoder.fit_transform(raw_labels)
            
            # Get processed label distribution
            binary_distribution = binary_labels.value_counts().to_dict()
            category_distribution = attack_categories.value_counts().to_dict()
            
            # Store results
            self.label_results[cleaned_file.name] = {
                'original_shape': cleaned_df.shape,
                'original_labels': original_labels,
                'attack_category_mapping': category_distribution,
                'binary_label_distribution': binary_distribution,
                'normal_samples': int(binary_distribution.get(0, 0)),
                'anomaly_samples': int(binary_distribution.get(1, 0)),
                'anomaly_percentage': round((binary_distribution.get(1, 0) / len(cleaned_df)) * 100, 2),
                'label_processing_actions': [
                    f"Mapped {len(original_labels)} original label types",
                    f"Created {len(category_distribution)} attack categories",
                    f"Converted to binary classification: Normal vs Anomaly",
                    f"Created multi-class labels for attack categories",
                    f"Anomaly rate: {round((binary_distribution.get(1, 0) / len(cleaned_df)) * 100, 2)}%"
                ],
                'raw_labels': raw_labels,
                'binary_labels': binary_labels,
                'attack_categories': attack_categories,
                'category_numeric': category_numeric,
                'attack_type_numeric': attack_type_numeric,
                'category_encoder': category_encoder,
                'attack_type_encoder': attack_type_encoder,
                'category_mapping': dict(zip(attack_categories.unique(), category_encoder.transform(attack_categories.unique()))),
                'attack_type_mapping': dict(zip(raw_labels.unique(), attack_type_encoder.transform(raw_labels.unique())))
            }
    
    def _validate_labels(self):
        """Validate label consistency and quality"""
        for file_name, results in self.label_results.items():
            logger.info(f"Validating labels for: {file_name}")
            
            # Use stored binary labels
            binary_labels = results['binary_labels']
            
            # Validation checks
            validation_results = {
                'has_null_labels': binary_labels.isnull().sum(),
                'unique_binary_labels': binary_labels.nunique(),
                'binary_label_range': [int(binary_labels.min()), int(binary_labels.max())],
                'label_consistency': binary_labels.isin([0, 1]).all(),
                'total_samples': len(binary_labels)
            }
            
            # Check for label encoding issues
            if validation_results['has_null_labels'] > 0:
                logger.warning(f"Found {validation_results['has_null_labels']} null labels in {file_name}")
            
            if validation_results['unique_binary_labels'] != 2:
                logger.warning(f"Expected 2 binary labels, found {validation_results['unique_binary_labels']} in {file_name}")
            
            if not validation_results['label_consistency']:
                logger.error(f"Label consistency check failed for {file_name}")
            
            # Store validation results
            self.label_results[file_name]['label_validation'] = validation_results
    
    def _analyze_class_balance(self):
        """Analyze class balance and provide recommendations"""
        for file_name, results in self.label_results.items():
            logger.info(f"Analyzing class balance for: {file_name}")
            
            normal_samples = results['normal_samples']
            anomaly_samples = results['anomaly_samples']
            total_samples = normal_samples + anomaly_samples
            
            # Calculate balance metrics
            normal_percentage = (normal_samples / total_samples) * 100
            anomaly_percentage = (anomaly_samples / total_samples) * 100
            
            # Handle division by zero
            if normal_samples == 0 or anomaly_samples == 0:
                imbalance_ratio = float('inf')
            else:
                imbalance_ratio = max(normal_samples, anomaly_samples) / min(normal_samples, anomaly_samples)
            
            # Class balance assessment
            if imbalance_ratio == float('inf'):
                balance_status = "Single Class Only"
            elif imbalance_ratio < 1.5:
                balance_status = "Well Balanced"
            elif imbalance_ratio < 3:
                balance_status = "Moderately Imbalanced"
            elif imbalance_ratio < 10:
                balance_status = "Highly Imbalanced"
            else:
                balance_status = "Severely Imbalanced"
            
            # Recommendations based on imbalance
            recommendations = []
            if imbalance_ratio > 2:
                recommendations.append("Consider class balancing techniques (SMOTE, undersampling)")
            if anomaly_percentage < 5:
                recommendations.append("Very low anomaly rate - consider anomaly detection specific techniques")
            if anomaly_percentage > 40:
                recommendations.append("High anomaly rate - verify data quality and labeling")
            
            # Store balance analysis
            self.label_results[file_name]['class_balance_analysis'] = {
                'normal_percentage': round(normal_percentage, 2),
                'anomaly_percentage': round(anomaly_percentage, 2),
                'imbalance_ratio': round(imbalance_ratio, 2),
                'balance_status': balance_status,
                'recommendations': recommendations
            }
    
    def _prepare_final_datasets(self):
        """Prepare final datasets with processed labels and attack categories"""
        processed_dir = self.output_dir / "processed_data"
        processed_dir.mkdir(exist_ok=True)
        
        for file_name in self.label_results.keys():
            logger.info(f"Preparing final dataset for: {file_name}")
            
            # Load cleaned data
            cleaned_file = self.cleaned_data_dir / file_name
            df = pd.read_csv(cleaned_file)
            
            # Get all label types from stored results
            binary_labels = self.label_results[file_name]['binary_labels']
            attack_categories = self.label_results[file_name]['attack_categories']
            category_numeric = self.label_results[file_name]['category_numeric']
            attack_type_numeric = self.label_results[file_name]['attack_type_numeric']
            
            # Ensure we have the right number of labels
            if len(df) != len(binary_labels):
                min_length = min(len(df), len(binary_labels))
                df = df.iloc[:min_length]
                binary_labels = binary_labels.iloc[:min_length]
                attack_categories = attack_categories.iloc[:min_length]
                category_numeric = category_numeric[:min_length]
                attack_type_numeric = attack_type_numeric[:min_length]
            
            # Add all label columns as last columns
            df_final = df.copy()
            df_final['Binary_Label'] = binary_labels.values
            df_final['Attack_Category'] = attack_categories.values
            df_final['Attack_Category_Numeric'] = category_numeric
            df_final['Attack_Type_Numeric'] = attack_type_numeric
            
            # Save processed dataset
            processed_filename = f"processed_{file_name.replace('cleaned_', '')}"
            processed_path = processed_dir / processed_filename
            df_final.to_csv(processed_path, index=False)
            
            logger.info(f"Saved processed dataset to: {processed_path}")
            
            # Store dataset info
            self.label_results[file_name]['final_dataset'] = {
                'filename': processed_filename,
                'shape': df_final.shape,
                'features': list(df_final.columns[:-4]),  # Exclude 4 label columns
                'target_columns': ['Binary_Label', 'Attack_Category', 'Attack_Category_Numeric', 'Attack_Type_Numeric'],
                'binary_target': 'Binary_Label',
                'category_target': 'Attack_Category_Numeric',
                'attack_type_target': 'Attack_Type_Numeric',
                'saved_path': str(processed_path)
            }
    
    def create_attack_type_lookup_functions(self):
        """Create lookup functions for post-processing attack types"""
        logger.info("Creating attack type lookup functions")
        
        # Collect all mappings from all files
        all_category_mappings = {}
        all_attack_type_mappings = {}
        
        for file_name, results in self.label_results.items():
            if 'category_mapping' in results:
                all_category_mappings.update(results['category_mapping'])
            if 'attack_type_mapping' in results:
                all_attack_type_mappings.update(results['attack_type_mapping'])
        
        # Create reverse mappings
        reverse_category_mapping = {v: k for k, v in all_category_mappings.items()}
        reverse_attack_type_mapping = {v: k for k, v in all_attack_type_mappings.items()}
        
        # Save lookup functions
        lookup_dir = self.output_dir / "lookup_functions"
        lookup_dir.mkdir(exist_ok=True)
        
        # Create category lookup function
        category_lookup_code = f'''
def get_attack_category(category_id):
    """Convert numeric category ID to attack category name"""
    category_mapping = {reverse_category_mapping}
    return category_mapping.get(category_id, "Unknown")

def get_attack_category_id(category_name):
    """Convert attack category name to numeric ID"""
    category_mapping = {all_category_mappings}
    return category_mapping.get(category_name, -1)
'''
        
        # Create attack type lookup function
        attack_type_lookup_code = f'''
def get_attack_type(attack_type_id):
    """Convert numeric attack type ID to attack type name"""
    attack_type_mapping = {reverse_attack_type_mapping}
    return attack_type_mapping.get(attack_type_id, "Unknown")

def get_attack_type_id(attack_type_name):
    """Convert attack type name to numeric ID"""
    attack_type_mapping = {all_attack_type_mappings}
    return attack_type_mapping.get(attack_type_name, -1)
'''
        
        # Create two-stage classification function
        two_stage_code = '''
def two_stage_classification(binary_prediction, category_prediction=None, attack_type_prediction=None):
    """
    Two-stage classification system for anomaly detection
    
    Args:
        binary_prediction: 0 (Normal) or 1 (Anomaly)
        category_prediction: Attack category ID (optional)
        attack_type_prediction: Specific attack type ID (optional)
    
    Returns:
        dict: Classification results at different levels
    """
    result = {
        'is_anomaly': bool(binary_prediction),
        'classification_level': 'Normal' if binary_prediction == 0 else 'Anomaly'
    }
    
    if binary_prediction == 1:  # If anomaly detected
        if category_prediction is not None:
            result['attack_category'] = get_attack_category(category_prediction)
            result['classification_level'] = result['attack_category']
            
            if attack_type_prediction is not None:
                result['attack_type'] = get_attack_type(attack_type_prediction)
                result['classification_level'] = result['attack_type']
    
    return result

def interpret_anomaly_result(binary_pred, category_pred=None, type_pred=None):
    """
    Interpret anomaly detection results with confidence levels
    """
    if binary_pred == 0:
        return {
            'status': 'Normal',
            'confidence': 'High',
            'details': 'No malicious activity detected'
        }
    
    result = {
        'status': 'Anomaly Detected',
        'confidence': 'High'
    }
    
    if category_pred is not None:
        category = get_attack_category(category_pred)
        result['attack_category'] = category
        result['details'] = f'Malicious activity detected: {category}'
        
        if type_pred is not None:
            attack_type = get_attack_type(type_pred)
            result['attack_type'] = attack_type
            result['details'] = f'Specific attack detected: {attack_type} ({category})'
    else:
        result['details'] = 'Malicious activity detected, but specific type unknown'
    
    return result
'''
        
        # Save lookup functions
        with open(lookup_dir / "attack_category_lookup.py", 'w', encoding='utf-8') as f:
            f.write(category_lookup_code)
        
        with open(lookup_dir / "attack_type_lookup.py", 'w', encoding='utf-8') as f:
            f.write(attack_type_lookup_code)
        
        with open(lookup_dir / "two_stage_classification.py", 'w', encoding='utf-8') as f:
            f.write(category_lookup_code + "\n" + attack_type_lookup_code + "\n" + two_stage_code)
        
        # Save mappings as JSON
        with open(lookup_dir / "mappings.json", 'w', encoding='utf-8') as f:
            # Convert numpy int64 to regular int for JSON serialization
            category_mapping_clean = {k: int(v) for k, v in all_category_mappings.items()}
            attack_type_mapping_clean = {k: int(v) for k, v in all_attack_type_mappings.items()}
            reverse_category_mapping_clean = {int(k): v for k, v in reverse_category_mapping.items()}
            reverse_attack_type_mapping_clean = {int(k): v for k, v in reverse_attack_type_mapping.items()}
            
            json.dump({
                'category_mapping': category_mapping_clean,
                'attack_type_mapping': attack_type_mapping_clean,
                'reverse_category_mapping': reverse_category_mapping_clean,
                'reverse_attack_type_mapping': reverse_attack_type_mapping_clean
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Lookup functions saved to: {lookup_dir}")
        
        return {
            'category_mapping': all_category_mappings,
            'attack_type_mapping': all_attack_type_mappings,
            'total_categories': len(all_category_mappings),
            'total_attack_types': len(all_attack_type_mappings)
        }
    
    def _generate_label_summary(self):
        """Generate overall label processing summary"""
        total_files = len(self.label_results)
        total_samples = 0
        total_normal = 0
        total_anomalies = 0
        overall_anomaly_rate = 0
        
        balance_statuses = []
        all_recommendations = []
        
        for file_name, results in self.label_results.items():
            # Count samples
            if 'final_dataset' in results:
                total_samples += results['final_dataset']['shape'][0]
            
            # Count normal and anomaly samples
            total_normal += results['normal_samples']
            total_anomalies += results['anomaly_samples']
            
            # Collect balance statuses
            if 'class_balance_analysis' in results:
                balance_statuses.append(results['class_balance_analysis']['balance_status'])
                all_recommendations.extend(results['class_balance_analysis']['recommendations'])
        
        # Calculate overall anomaly rate
        if total_samples > 0:
            overall_anomaly_rate = (total_anomalies / total_samples) * 100
        
        # Get unique recommendations
        unique_recommendations = list(set(all_recommendations))
        
        self.label_stats = {
            'label_processing_timestamp': datetime.now().isoformat(),
            'dataset_summary': {
                'total_files_processed': total_files,
                'total_samples': int(total_samples),
                'total_normal_samples': int(total_normal),
                'total_anomaly_samples': int(total_anomalies),
                'overall_anomaly_rate': round(overall_anomaly_rate, 2)
            },
            'processing_summary': {
                'label_conversion': 'Multi-class to Binary (Normal vs Anomaly)',
                'attack_categories': len(self.attack_mapping),
                'binary_mapping': self.binary_mapping,
                'most_common_balance_status': max(set(balance_statuses), key=balance_statuses.count) if balance_statuses else 'Unknown'
            },
            'quality_assessment': {
                'data_ready_for_ml': True,
                'binary_classification_ready': True,
                'anomaly_detection_ready': True,
                'class_balance_needs_attention': any('Imbalanced' in status for status in balance_statuses)
            },
            'recommendations': unique_recommendations if unique_recommendations else ["Dataset is well-balanced and ready for ML training"]
        }
    
    def _save_processed_datasets(self):
        """Save processed datasets and results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save label processing results
        results_file = self.output_dir / f"phase4_label_processing_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'label_processing_results': self.label_results,
                'label_processing_statistics': self.label_stats
            }, f, indent=2, default=str)
        
        # Save summary report
        summary_file = self.output_dir / f"phase4_summary_report_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("PHASE 4: LABEL PROCESSING SUMMARY\n")
            f.write("=" * 40 + "\n\n")
            
            # Dataset summary
            ds_summary = self.label_stats['dataset_summary']
            f.write(f"Dataset Summary:\n")
            f.write(f"- Files Processed: {ds_summary['total_files_processed']}\n")
            f.write(f"- Total Samples: {ds_summary['total_samples']:,}\n")
            f.write(f"- Normal Samples: {ds_summary['total_normal_samples']:,}\n")
            f.write(f"- Anomaly Samples: {ds_summary['total_anomaly_samples']:,}\n")
            f.write(f"- Overall Anomaly Rate: {ds_summary['overall_anomaly_rate']}%\n\n")
            
            # Processing summary
            p_summary = self.label_stats['processing_summary']
            f.write(f"Processing Summary:\n")
            f.write(f"- Label Conversion: {p_summary['label_conversion']}\n")
            f.write(f"- Attack Categories: {p_summary['attack_categories']}\n")
            f.write(f"- Binary Mapping: {p_summary['binary_mapping']}\n")
            f.write(f"- Common Balance Status: {p_summary['most_common_balance_status']}\n\n")
            
            # Quality assessment
            q_assessment = self.label_stats['quality_assessment']
            f.write(f"Quality Assessment:\n")
            f.write(f"- Ready for ML: {'Yes' if q_assessment['data_ready_for_ml'] else 'No'}\n")
            f.write(f"- Binary Classification Ready: {'Yes' if q_assessment['binary_classification_ready'] else 'No'}\n")
            f.write(f"- Anomaly Detection Ready: {'Yes' if q_assessment['anomaly_detection_ready'] else 'No'}\n")
            f.write(f"- Class Balance Attention Needed: {'Yes' if q_assessment['class_balance_needs_attention'] else 'No'}\n\n")
            
            # Recommendations
            f.write(f"Recommendations:\n")
            for i, rec in enumerate(self.label_stats['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Summary report saved to: {summary_file}")


def main():
    """Main execution function"""
    # Define paths
    raw_data_dir = "data/raw/CICIDS2017/MachineLearningCVE"
    cleaned_data_dir = "data_preprocessing/cleaned_data"
    engineered_data_dir = "data_preprocessing/engineered_data"
    output_dir = "data_preprocessing"
    
    # Run label processing pipeline
    processor = LabelProcessingPipeline(raw_data_dir, cleaned_data_dir, engineered_data_dir, output_dir)
    results = processor.run_complete_label_processing()
    
    # Print summary
    print("\n" + "="*50)
    print("PHASE 4: LABEL PROCESSING COMPLETE")
    print("="*50)
    print(f"Files Processed: {results['dataset_summary']['total_files_processed']}")
    print(f"Total Samples: {results['dataset_summary']['total_samples']:,}")
    print(f"Normal Samples: {results['dataset_summary']['total_normal_samples']:,}")
    print(f"Anomaly Samples: {results['dataset_summary']['total_anomaly_samples']:,}")
    print(f"Anomaly Rate: {results['dataset_summary']['overall_anomaly_rate']}%")
    print(f"Ready for ML: {'Yes' if results['quality_assessment']['data_ready_for_ml'] else 'No'}")
    print("="*50)


if __name__ == "__main__":
    main()
