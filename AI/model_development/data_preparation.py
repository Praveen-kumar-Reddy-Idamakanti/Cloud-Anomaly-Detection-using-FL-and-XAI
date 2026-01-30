"""
Phase 2: Data Preparation & Loading for Cloud Anomaly Detection
Load and prepare processed datasets for autoencoder training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class CloudAnomalyDataset(Dataset):
    """Custom Dataset for Cloud Anomaly Detection"""
    
    def __init__(self, features, labels=None):
        """
        Initialize dataset
        
        Args:
            features (np.array): Normalized features
            labels (np.array): Binary labels (optional)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


class DataPreparation:
    """Data preparation pipeline for autoencoder training"""
    
    def __init__(self, data_dir="data_preprocessing/processed_data", 
                 test_size=0.2, validation_split=0.2, random_state=42):
        """
        Initialize data preparation
        
        Args:
            data_dir (str): Directory containing processed datasets
            test_size (float): Proportion of data for testing
            validation_split (float): Proportion of training data for validation
            random_state (int): Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.test_size = test_size
        self.validation_split = validation_split
        self.random_state = random_state
        
        # Storage for data and metadata
        self.all_data = None
        self.normal_data = None
        self.anomaly_data = None
        self.scaler = None
        self.feature_names = None
        self.data_stats = {}
        
        logger.info(f"DataPreparation initialized:")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Test size: {self.test_size}")
        logger.info(f"  Validation split: {self.validation_split}")
    
    def load_processed_datasets(self):
        """Load all processed datasets from the preprocessing phase"""
        logger.info("Loading processed datasets...")
        
        # Find all processed CSV files
        processed_files = list(self.data_dir.glob("processed_*.csv"))
        
        if not processed_files:
            raise FileNotFoundError(f"No processed files found in {self.data_dir}")
        
        logger.info(f"Found {len(processed_files)} processed files:")
        for file in processed_files:
            logger.info(f"  - {file.name}")
        
        # Load and combine all datasets
        all_dfs = []
        total_samples = 0
        
        for file_path in processed_files:
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {file_path.name}: {df.shape}")
                all_dfs.append(df)
                total_samples += len(df)
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
                continue
        
        if not all_dfs:
            raise ValueError("No valid datasets loaded")
        
        # Combine all datasets
        self.all_data = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined dataset shape: {self.all_data.shape}")
        
        # Extract feature names (exclude label columns)
        label_columns = ['Binary_Label', 'Attack_Category', 'Attack_Category_Numeric', 'Attack_Type_Numeric']
        self.feature_names = [col for col in self.all_data.columns if col not in label_columns]
        
        logger.info(f"Total features: {len(self.feature_names)}")
        logger.info(f"Feature names: {self.feature_names[:10]}...")  # Show first 10
        
        return self.all_data
    
    def separate_normal_anomaly_data(self):
        """Separate normal and anomaly data"""
        logger.info("Separating normal and anomaly data...")
        
        if self.all_data is None:
            raise ValueError("Data not loaded. Call load_processed_datasets() first.")
        
        # Separate based on Binary_Label
        normal_mask = self.all_data['Binary_Label'] == 0
        anomaly_mask = self.all_data['Binary_Label'] == 1
        
        self.normal_data = self.all_data[normal_mask].copy()
        self.anomaly_data = self.all_data[anomaly_mask].copy()
        
        # Store statistics
        self.data_stats = {
            'total_samples': len(self.all_data),
            'normal_samples': len(self.normal_data),
            'anomaly_samples': len(self.anomaly_data),
            'normal_percentage': (len(self.normal_data) / len(self.all_data)) * 100,
            'anomaly_percentage': (len(self.anomaly_data) / len(self.all_data)) * 100
        }
        
        logger.info(f"Data separation complete:")
        logger.info(f"  Normal samples: {self.data_stats['normal_samples']:,} ({self.data_stats['normal_percentage']:.2f}%)")
        logger.info(f"  Anomaly samples: {self.data_stats['anomaly_samples']:,} ({self.data_stats['anomaly_percentage']:.2f}%)")
        
        return self.normal_data, self.anomaly_data
    
    def extract_features_and_normalize(self):
        """Extract features and normalize them"""
        logger.info("Extracting and normalizing features...")
        
        if self.normal_data is None:
            raise ValueError("Data not separated. Call separate_normal_anomaly_data() first.")
        
        # Extract features for normal data (for training)
        normal_features = self.normal_data[self.feature_names].values
        
        # Extract features for anomaly data (for testing)
        anomaly_features = self.anomaly_data[self.feature_names].values
        
        # Extract labels
        normal_labels = self.normal_data['Binary_Label'].values
        anomaly_labels = self.anomaly_data['Binary_Label'].values
        
        # Fit scaler on normal data only (to avoid data leakage)
        self.scaler = StandardScaler()
        normal_features_scaled = self.scaler.fit_transform(normal_features)
        anomaly_features_scaled = self.scaler.transform(anomaly_features)
        
        logger.info(f"Feature normalization complete:")
        logger.info(f"  Normal features shape: {normal_features_scaled.shape}")
        logger.info(f"  Anomaly features shape: {anomaly_features_scaled.shape}")
        logger.info(f"  Feature mean (normal): {np.mean(normal_features_scaled, axis=0)[:5]}")
        logger.info(f"  Feature std (normal): {np.std(normal_features_scaled, axis=0)[:5]}")
        
        return normal_features_scaled, normal_labels, anomaly_features_scaled, anomaly_labels
    
    def create_train_test_splits(self, normal_features, normal_labels, anomaly_features, anomaly_labels):
        """Create train, validation, and test splits"""
        logger.info("Creating train/validation/test splits...")
        
        # Split normal data (for training and validation)
        train_normal, temp_normal, train_normal_labels, temp_normal_labels = train_test_split(
            normal_features, normal_labels, 
            test_size=(self.test_size + self.validation_split), 
            random_state=self.random_state,
            stratify=normal_labels
        )
        
        # Split remaining normal data into validation and test
        val_size = self.validation_split / (self.test_size + self.validation_split)
        val_normal, test_normal, val_normal_labels, test_normal_labels = train_test_split(
            temp_normal, temp_normal_labels,
            test_size=(1 - val_size),
            random_state=self.random_state,
            stratify=temp_normal_labels
        )
        
        # Split anomaly data (for testing only)
        test_anomaly, test_anomaly_labels = anomaly_features, anomaly_labels
        
        # Combine test sets
        test_features = np.concatenate([test_normal, test_anomaly], axis=0)
        test_labels = np.concatenate([test_normal_labels, test_anomaly_labels], axis=0)
        
        logger.info(f"Data splits created:")
        logger.info(f"  Training (normal only): {train_normal.shape}")
        logger.info(f"  Validation (normal only): {val_normal.shape}")
        logger.info(f"  Test (mixed): {test_features.shape}")
        logger.info(f"  Test - Normal: {len(test_normal_labels)}, Anomaly: {len(test_anomaly_labels)}")
        
        return {
            'train_features': train_normal,
            'train_labels': train_normal_labels,
            'val_features': val_normal,
            'val_labels': val_normal_labels,
            'test_features': test_features,
            'test_labels': test_labels
        }
    
    def create_data_loaders(self, splits, batch_size=128):
        """Create PyTorch DataLoaders"""
        logger.info(f"Creating DataLoaders with batch size {batch_size}...")
        
        # Create datasets
        train_dataset = CloudAnomalyDataset(splits['train_features'], splits['train_labels'])
        val_dataset = CloudAnomalyDataset(splits['val_features'], splits['val_labels'])
        test_dataset = CloudAnomalyDataset(splits['test_features'], splits['test_labels'])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        logger.info(f"DataLoaders created:")
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Validation batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def prepare_data(self, batch_size=128):
        """Complete data preparation pipeline"""
        logger.info("Starting complete data preparation pipeline...")
        
        # Step 1: Load datasets
        self.load_processed_datasets()
        
        # Step 2: Separate normal and anomaly data
        self.separate_normal_anomaly_data()
        
        # Step 3: Extract and normalize features
        normal_features, normal_labels, anomaly_features, anomaly_labels = self.extract_features_and_normalize()
        
        # Step 4: Create train/test splits
        splits = self.create_train_test_splits(normal_features, normal_labels, anomaly_features, anomaly_labels)
        
        # Step 5: Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(splits, batch_size)
        
        # Prepare results dictionary
        results = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'splits': splits,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'data_stats': self.data_stats
        }
        
        logger.info("✅ Data preparation complete!")
        return results
    
    def save_preparation_info(self, filepath):
        """Save data preparation information"""
        info = {
            'data_stats': self.data_stats,
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'test_size': self.test_size,
            'validation_split': self.validation_split,
            'random_state': self.random_state,
            'preparation_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Data preparation info saved to: {filepath}")


def test_data_preparation():
    """Test the data preparation pipeline"""
    logger.info("Testing data preparation pipeline...")
    
    try:
        # Initialize data preparation
        data_prep = DataPreparation()
        
        # Prepare data
        results = data_prep.prepare_data(batch_size=128)
        
        # Test data loaders
        train_loader = results['train_loader']
        val_loader = results['val_loader']
        test_loader = results['test_loader']
        
        # Test batch iteration
        for batch_idx, (features, labels) in enumerate(train_loader):
            if batch_idx == 0:  # Just test first batch
                logger.info(f"✅ Train batch shape: {features.shape}, Labels: {labels.shape}")
                break
        
        for batch_idx, (features, labels) in enumerate(val_loader):
            if batch_idx == 0:  # Just test first batch
                logger.info(f"✅ Validation batch shape: {features.shape}, Labels: {labels.shape}")
                break
        
        for batch_idx, (features, labels) in enumerate(test_loader):
            if batch_idx == 0:  # Just test first batch
                logger.info(f"✅ Test batch shape: {features.shape}, Labels: {labels.shape}")
                logger.info(f"✅ Test batch unique labels: {torch.unique(labels).tolist()}")
                break
        
        # Save preparation info
        data_prep.save_preparation_info("model_artifacts/data_preparation_info.json")
        
        logger.info("🎉 Data preparation test successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data preparation test failed: {e}")
        return False


if __name__ == "__main__":
    # Test data preparation
    success = test_data_preparation()
    
    if success:
        logger.info("🎉 Phase 2: Data Preparation & Loading - COMPLETED")
        logger.info("✅ Data is ready for Phase 3: Autoencoder Implementation")
    else:
        logger.error("❌ Data preparation failed")
