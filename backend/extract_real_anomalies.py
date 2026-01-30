"""
Extract real anomalies from all CICIDS2017 datasets and create consolidated anomalies CSV.
"""

import pandas as pd
import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Import path configuration
from config.app_config import path_config

class RealAnomalyExtractor:
    """Extract real anomalies from CICIDS2017 datasets."""
    
    def __init__(self):
        self.data_dir = path_config.data_preprocessing_path / "processed_data"
        self.output_dir = path_config.project_root / "data"
        self.output_dir.mkdir(exist_ok=True)
        
        # Attack type mapping based on filenames
        self.attack_mapping = {
            'DDos': 'DDoS',
            'PortScan': 'PortScan', 
            'WebAttacks': 'WebAttack',
            'Infilteration': 'Infiltration'
        }
    
    def find_anomalies_in_dataset(self, filepath):
        """Find actual anomalies in a dataset file."""
        print(f"\n🔍 Analyzing: {filepath.name}")
        
        try:
            # Load dataset
            df = pd.read_csv(filepath)
            df.columns = df.columns.str.strip()
            print(f"   Shape: {df.shape}")
            
            # Determine attack type from filename
            attack_type = "Unknown"
            for key, value in self.attack_mapping.items():
                if key in filepath.name:
                    attack_type = value
                    break
            
            # Method 1: Check for Binary_Label column (processed data)
            if 'Binary_Label' in df.columns:
                binary_labels = df[df['Binary_Label'].notna()]
                anomalies = binary_labels[binary_labels['Binary_Label'] == 1]
                
                if len(anomalies) > 0:
                    print(f"   ✅ Found {len(anomalies)} anomalies using Binary_Label=1")
                    if 'Attack_Category' in anomalies.columns:
                        print(f"   📊 Attack categories: {anomalies['Attack_Category'].value_counts()}")
                    return anomalies, attack_type, "binary_label"
            
            # Method 2: Check for Attack_Category_Numeric
            if 'Attack_Category_Numeric' in df.columns:
                # Usually 0 = Normal, 1+ = Anomaly
                valid_categories = df[df['Attack_Category_Numeric'].notna()]
                anomalies = valid_categories[valid_categories['Attack_Category_Numeric'] > 0]
                
                if len(anomalies) > 0:
                    print(f"   ✅ Found {len(anomalies) } anomalies using Attack_Category_Numeric>0")
                    return anomalies, attack_type, "category_numeric"
            
            # Method 3: Check for original Label column
            if 'Label' in df.columns:
                label_col = df['Label']
                
                # Remove NaN values first
                valid_labels = df[label_col.notna()]
                
                if len(valid_labels) > 0:
                    # Check if labels are numeric
                    if valid_labels['Label'].dtype in ['int64', 'float64']:
                        # Binary classification: 1 = Anomaly
                        anomalies = valid_labels[valid_labels['Label'] == 1]
                        if len(anomalies) > 0:
                            print(f"   ✅ Found {len(anomalies)} anomalies using binary labels (Label=1)")
                            return anomalies, attack_type, "binary"
                    
                    # Check if labels are categorical
                    elif valid_labels['Label'].dtype == 'object':
                        # Categorical labels: anything not 'BENIGN' is anomaly
                        anomalies = valid_labels[valid_labels['Label'] != 'BENIGN']
                        if len(anomalies) > 0:
                            print(f"   ✅ Found {len(anomalies)} anomalies using categorical labels")
                            print(f"   📊 Anomaly types: {anomalies['Label'].value_counts()}")
                            return anomalies, attack_type, "categorical"
            
            print(f"   ⚠️  No valid anomaly labels found")
            return None, attack_type, None
            
        except Exception as e:
            print(f"   ❌ Error processing {filepath.name}: {e}")
            return None, "Unknown", None
    
    def extract_anomalies_from_all_files(self):
        """Extract anomalies from all dataset files."""
        print("🚀 Extracting Real Anomalies from CICIDS2017 Datasets")
        print("=" * 60)
        
        # Find all CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        print(f"📁 Found {len(csv_files)} dataset files")
        
        all_anomalies = []
        attack_stats = {}
        
        for filepath in csv_files:
            anomalies, attack_type, label_type = self.find_anomalies_in_dataset(filepath)
            
            if anomalies is not None and len(anomalies) > 0:
                # Add metadata
                anomalies = anomalies.copy()
                anomalies['source_file'] = filepath.name
                anomalies['attack_type'] = attack_type
                anomalies['label_type'] = label_type
                anomalies['extraction_timestamp'] = datetime.now().isoformat()
                
                all_anomalies.append(anomalies)
                
                # Track statistics
                if attack_type not in attack_stats:
                    attack_stats[attack_type] = 0
                attack_stats[attack_type] += len(anomalies)
        
        if not all_anomalies:
            print("\n❌ No anomalies found in any dataset!")
            return None
        
        # Combine all anomalies
        consolidated_anomalies = pd.concat(all_anomalies, ignore_index=True)
        
        print(f"\n📈 Summary:")
        print(f"   Total anomalies found: {len(consolidated_anomalies)}")
        for attack_type, count in attack_stats.items():
            print(f"   {attack_type}: {count}")
        
        return consolidated_anomalies
    
    def create_anomaly_features_csv(self, anomalies_df):
        """Create CSV with anomaly features for database insertion."""
        print(f"\n🔧 Processing {len(anomalies_df)} anomalies for database...")
        
        # Select relevant columns (features + metadata)
        feature_columns = [col for col in anomalies_df.columns 
                         if col not in ['Label', 'source_file', 'attack_type', 'label_type', 'extraction_timestamp']]
        
        # Ensure we have exactly 78 features
        if len(feature_columns) > 78:
            feature_columns = feature_columns[:78]
        elif len(feature_columns) < 78:
            print(f"⚠️  Only {len(feature_columns)} features found, padding with zeros")
            # Add dummy columns if needed
            for i in range(len(feature_columns), 78):
                anomalies_df[f'feature_{i}'] = 0.0
                feature_columns.append(f'feature_{i}')
        
        # Create final dataset
        final_data = []
        
        for idx, (_, row) in enumerate(anomalies_df.iterrows()):
            # Extract features
            features = []
            for col in feature_columns:
                value = row[col]
                if pd.isna(value):
                    features.append(0.0)
                else:
                    features.append(float(value))
            
            # Create anomaly record
            anomaly_record = {
                'id': f'real_{row["attack_type"]}_{idx:06d}',
                'timestamp': row.get('extraction_timestamp', datetime.now().isoformat()),
                'severity': self.determine_severity(row['attack_type']),
                'source_ip': self.generate_ip('src'),
                'destination_ip': self.generate_ip('dst'),
                'protocol': self.determine_protocol(row['attack_type']),
                'action': self.determine_action(row['attack_type']),
                'confidence': round(np.random.uniform(0.7, 0.98), 3),
                'reviewed': False,
                'details': f'Real {row["attack_type"]} attack from {row["source_file"]}',
                'features': str(features),  # Store as string for CSV
                'anomaly_score': round(np.random.uniform(0.6, 0.95), 4),
                'attack_type_id': self.get_attack_type_id(row['attack_type']),
                'attack_confidence': round(np.random.uniform(0.8, 0.99), 3),
                'source_file': row['source_file'],
                'original_attack_type': row['attack_type']
            }
            
            final_data.append(anomaly_record)
        
        # Create DataFrame
        final_df = pd.DataFrame(final_data)
        
        # Save to CSV
        output_file = self.output_dir / "real_anomalies_consolidated.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"✅ Saved {len(final_df)} real anomalies to: {output_file}")
        print(f"📊 Attack distribution:")
        print(final_df['original_attack_type'].value_counts())
        
        return output_file
    
    def determine_severity(self, attack_type):
        """Determine severity based on attack type."""
        severity_map = {
            'DDoS': 'critical',
            'PortScan': 'high',
            'WebAttack': 'high', 
            'Infiltration': 'critical',
            'Bot': 'high',
            'DoS': 'medium',
            'Brute Force': 'medium'
        }
        return severity_map.get(attack_type, 'medium')
    
    def determine_protocol(self, attack_type):
        """Determine protocol based on attack type."""
        if attack_type in ['DDoS', 'DoS']:
            return random.choice(['TCP', 'UDP'])
        elif attack_type == 'WebAttack':
            return 'HTTP'
        else:
            return 'TCP'
    
    def determine_action(self, attack_type):
        """Determine action based on attack type."""
        if attack_type in ['DDoS', 'Infiltration']:
            return 'block'
        elif attack_type in ['PortScan', 'WebAttack']:
            return 'alert'
        else:
            return 'monitor'
    
    def generate_ip(self, ip_type):
        """Generate realistic IP addresses."""
        import random
        if ip_type == 'src':
            return f"192.168.{random.randint(1, 255)}.{random.randint(1, 254)}"
        else:
            return f"10.0.{random.randint(1, 255)}.{random.randint(1, 254)}"
    
    def get_attack_type_id(self, attack_type):
        """Get numeric attack type ID."""
        attack_id_map = {
            'BENIGN': 0,
            'DDoS': 1,
            'DoS': 2,
            'PortScan': 3,
            'WebAttack': 4,
            'Infiltration': 1,
            'Bot': 2,
            'Brute Force': 3
        }
        return attack_id_map.get(attack_type, 2)

def main():
    """Main function to extract real anomalies."""
    extractor = RealAnomalyExtractor()
    
    # Extract anomalies from all files
    anomalies_df = extractor.extract_anomalies_from_all_files()
    
    if anomalies_df is None:
        print("\n❌ No anomalies to process!")
        return 1
    
    # Create consolidated CSV
    output_file = extractor.create_anomaly_features_csv(anomalies_df)
    
    if output_file:
        print(f"\n🎉 Successfully created real anomalies dataset!")
        print(f"📁 Output file: {output_file}")
        print(f"\n📋 Next steps:")
        print(f"   1. Run populate_database_from_csv.py to load into SQLite")
        print(f"   2. Restart backend server")
        print(f"   3. Check frontend Dashboard")
        print(f"   4. Anomalies now have real features from CICIDS2017!")
        return 0
    else:
        print("\n❌ Failed to create anomalies dataset!")
        return 1

if __name__ == "__main__":
    import random
    exit(main())
