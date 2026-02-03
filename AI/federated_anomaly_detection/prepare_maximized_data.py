#!/usr/bin/env python3
"""
Enhanced federated data preparation using ALL 8 processed data files
Distribute each file to separate clients for maximum data utilization
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_maximized_federated_data(
    processed_data_dir: str,
    output_dir: str,
    validation_split: float = 0.2,
    random_state: int = 42
) -> None:
    """
    Create federated learning data using ALL 8 processed files
    Each client gets one complete dataset file
    
    Args:
        processed_data_dir: Directory containing processed CSV files
        output_dir: Directory to save federated data splits
        validation_split: Fraction of data for validation
        random_state: Random seed for reproducibility
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get ALL processed CSV files
    processed_files = list(Path(processed_data_dir).glob("processed_*.csv"))
    processed_files.sort()  # Ensure consistent ordering
    
    logger.info(f"🎯 FOUND {len(processed_files)} PROCESSED FILES:")
    for i, file_path in enumerate(processed_files, 1):
        size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"   {i}. {file_path.name} ({size_mb:.1f} MB)")
    
    if len(processed_files) != 8:
        logger.warning(f"Expected 8 files, found {len(processed_files)}")
    
    all_client_data = []
    total_samples = 0
    total_anomalies = 0
    
    # Process each file as a separate client
    for client_id, file_path in enumerate(processed_files, 1):
        logger.info(f"\n🔄 Processing Client {client_id}: {file_path.name}")
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            logger.info(f"   📊 Loaded {len(df)} rows")
            
            # Remove non-numeric columns except label columns
            exclude_cols = ['Label', 'label', 'target', 'Binary_Label', 'Attack_Category', 'Attack_Category_Numeric', 'Attack_Type_Numeric']
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            
            # Find the best label column
            if 'Binary_Label' in df.columns:
                labels = df['Binary_Label'].values
                logger.info(f"   🏷️  Using Binary_Label")
            elif 'Label' in df.columns:
                # Convert string labels to binary
                label_map = {'Normal': 0, 'Benign': 0, 'normal': 0, 'benign': 0}
                labels = df['Label'].map(lambda x: label_map.get(str(x), 1)).fillna(1).astype(int).values
                logger.info(f"   🏷️  Using Label (converted to binary)")
            elif 'label' in df.columns:
                labels = df['label'].values
                logger.info(f"   🏷️  Using label")
            elif 'target' in df.columns:
                labels = df['target'].values
                logger.info(f"   🏷️  Using target")
            else:
                logger.warning(f"   ⚠️  No label column found, using dummy labels")
                labels = np.zeros(len(df), dtype=int)
            
            # Convert features to numeric
            features = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
            
            # Remove any rows with NaN or infinite values
            valid_mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
            features = features[valid_mask]
            labels = labels[valid_mask]
            
            logger.info(f"   ✅ Valid samples: {len(features)}")
            logger.info(f"   🚨 Anomalies: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
            
            # Normalize features to [0, 1] range
            X_min = features.min(axis=0, keepdims=True)
            X_max = features.max(axis=0, keepdims=True)
            features = (features - X_min) / (X_max - X_min + 1e-8)
            
            # Split into train/validation
            train_features, val_features, train_labels, val_labels = train_test_split(
                features, labels, test_size=validation_split, random_state=random_state, stratify=labels
            )
            
            # Store client data
            client_data = {
                'client_id': client_id,
                'source_file': file_path.name,
                'train_features': train_features,
                'train_labels': train_labels,
                'val_features': val_features,
                'val_labels': val_labels,
                'n_train': len(train_features),
                'n_val': len(val_features),
                'n_anomalies_train': np.sum(train_labels),
                'n_anomalies_val': np.sum(val_labels)
            }
            
            all_client_data.append(client_data)
            total_samples += len(features)
            total_anomalies += np.sum(labels)
            
            logger.info(f"   📊 Train: {len(train_features):,} samples")
            logger.info(f"   🔍 Val: {len(val_features):,} samples")
            logger.info(f"   🚨 Train anomalies: {np.sum(train_labels):,}")
            logger.info(f"   🚨 Val anomalies: {np.sum(val_labels):,}")
            
        except Exception as e:
            logger.error(f"   ❌ Error processing {file_path.name}: {e}")
            continue
    
    # Save all client data
    logger.info(f"\n💾 Saving federated data for {len(all_client_data)} clients...")
    
    metadata = {
        'total_clients': len(all_client_data),
        'total_samples': int(total_samples),
        'total_anomalies': int(total_anomalies),
        'overall_anomaly_rate': float((total_anomalies / total_samples) * 100),
        'validation_split': float(validation_split),
        'source_files': [client['source_file'] for client in all_client_data],
        'creation_date': pd.Timestamp.now().isoformat()
    }
    
    for client in all_client_data:
        client_file = output_dir / f"client_{client['client_id']}.npz"
        
        # Save client data
        np.savez_compressed(
            client_file,
            features=client['train_features'],
            labels=client['train_labels'],
            val_features=client['val_features'],
            val_labels=client['val_labels'],
            client_id=client['client_id'],
            source_file=client['source_file'],
            n_train=client['n_train'],
            n_val=client['n_val'],
            n_anomalies_train=client['n_anomalies_train'],
            n_anomalies_val=client['n_anomalies_val']
        )
        
        file_size_mb = client_file.stat().st_size / (1024 * 1024)
        logger.info(f"   💾 Client {client['client_id']}: {client_file.name} ({file_size_mb:.1f} MB)")
    
    # Save metadata
    metadata_file = output_dir / "metadata.json"
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    logger.info(f"\n🎉 MAXIMIZED FEDERATED DATA PREPARATION COMPLETED!")
    logger.info(f"=" * 60)
    logger.info(f"👥 Total clients: {metadata['total_clients']}")
    logger.info(f"📊 Total samples: {metadata['total_samples']:,}")
    logger.info(f"🚂 Total training samples: {sum(c['n_train'] for c in all_client_data):,}")
    logger.info(f"🔍 Total validation samples: {sum(c['n_val'] for c in all_client_data):,}")
    logger.info(f"🚨 Total anomalies: {metadata['total_anomalies']:,}")
    logger.info(f"📈 Overall anomaly rate: {metadata['overall_anomaly_rate']:.1f}%")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"💾 Metadata saved: {metadata_file}")

def main():
    parser = argparse.ArgumentParser(description="Create maximized federated learning data from all processed files")
    parser.add_argument("--processed-data-dir", type=str, 
                       default="../data_preprocessing/processed_data",
                       help="Directory containing processed CSV files")
    parser.add_argument("--output-dir", type=str, 
                       default="data/maximized",
                       help="Directory to save federated data splits")
    parser.add_argument("--validation-split", type=float, default=0.2,
                       help="Fraction of data for validation")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        create_maximized_federated_data(
            processed_data_dir=args.processed_data_dir,
            output_dir=args.output_dir,
            validation_split=args.validation_split,
            random_state=args.random_state
        )
        
        logger.info("✅ Maximization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during data preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
