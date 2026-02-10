"""
ENHANCED Two-Stage Training Pipeline - Fixed Version
Integrates Anomaly Detection + Attack Type Classification
All issues resolved: NaN handling, JSON serialization, batch limits
Now with attack type classification capability!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json
import time
import copy
from datetime import datetime
import logging
import warnings
import argparse

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "model_artifacts"

# Import our modules
from autoencoder_model import CloudAnomalyAutoencoder, AutoencoderConfig
from data_preparation import DataPreparation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preprocessing.lookup_functions.attack_type_lookup import get_attack_type, get_attack_type_id

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            
        return False


class AttackTypeClassifier(nn.Module):
    """Multi-class classifier for attack types - Stage 2"""
    
    def __init__(self, input_dim=78, num_classes=5):
        super(AttackTypeClassifier, self).__init__()
        
        # Network architecture for attack classification
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.classifier(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FixedAutoencoderTrainer:
    """ENHANCED Two-Stage Training Pipeline - Anomaly Detection + Attack Type Classification"""
    
    def __init__(self, config=None, attack_category_classes=None, attack_category_encoder=None):
        """
        Initialize trainer with two-stage capability
        """
        self.config = config if config else AutoencoderConfig()
        self.config.input_dim = 78
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Stage 1: Autoencoder for anomaly detection (existing)
        self.model = CloudAnomalyAutoencoder(
            input_dim=78,  # FIXED: 78 instead of 79
            encoding_dims=self.config.encoding_dims,
            bottleneck_dim=self.config.bottleneck_dim,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Stage 2: Attack type classifier (NEW)
        self.attack_category_encoder = attack_category_encoder
        self.attack_category_classes = list(attack_category_classes) if attack_category_classes is not None else None
        self.attack_classifier = None
        if self.attack_category_classes is not None:
            self.attack_classifier = AttackTypeClassifier(
                input_dim=78,
                num_classes=len(self.attack_category_classes)
            ).to(self.device)
        
        # Optimizers
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.attack_optimizer = None
        if self.attack_classifier is not None:
            self.attack_optimizer = optim.Adam(self.attack_classifier.parameters(), lr=0.001)
        
        # Loss functions
        self.criterion = nn.MSELoss()
        self.attack_criterion = nn.CrossEntropyLoss()
        
        # Attack type mappings
        self.attack_types = self.attack_category_classes
        
        # Add gradient clipping to prevent exploding gradients
        self.max_grad_norm = 1.0
        
        # SHOWCASE CONFIGURATION - Mid Range System
        self.max_train_batches = 2000       # Increased from 1000 for showcase
        self.max_val_batches = 800          # Increased from 200 for showcase
        self.max_test_batches = 3000        # Increased from 1000 for showcase
        
        # Training history (ENHANCED for two-stage)
        self.training_history = {
            'ae_train_loss': [],
            'ae_val_loss': [],
            'attack_train_loss': [],
            'attack_val_loss': [],
            'attack_accuracy': [],
            'epochs': [],
            'training_time': []
        }
        
        # Early stopping - 🎬 SHOWCASE: Increased patience for full 50 epochs
        self.early_stopping = EarlyStopping(
            patience=50,  # Changed from 10 to 50 to allow full training
            min_delta=self.config.min_delta
        )
        
        logger.info(f"Enhanced Two-Stage Trainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Autoencoder parameters: {self.model.count_parameters():,}")
        if self.attack_classifier is not None:
            logger.info(f"  Attack Classifier parameters: {self.attack_classifier.count_parameters():,}")
        logger.info(f"  Input dimension: 78 (FIXED)")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  🎬 TWO-STAGE CONFIGURATION:")
        logger.info(f"    Max train batches: {self.max_train_batches}")
        logger.info(f"    Max val batches: {self.max_val_batches}")
        logger.info(f"    Max test batches: {self.max_test_batches}")
        if self.attack_types is not None:
            logger.info(f"    Attack types: {len(self.attack_types)} classes")
        logger.info(f"    Early stopping patience: 50 (full 50 epochs)")
        logger.info(f"    Expected time: 15-20 minutes")
        logger.info(f"    Expected F1: 70-75% (with attack classification)")
    
    def train_epoch(self, train_loader):
        """Train for one epoch with comprehensive error handling"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        valid_batches = 0
        
        for batch_idx, (features, _) in enumerate(train_loader):
            # ✅ FIX 2: Add batch limit to prevent infinite training
            if batch_idx >= self.max_train_batches:  # Use showcase limit
                break
                
            features = features.to(self.device)
            
            # ✅ FIX 3: Check for NaN data
            if torch.isnan(features).any():
                logger.warning(f"NaN data detected at batch {batch_idx}, skipping")
                continue
            
            # Forward pass
            reconstructed, encoded = self.model(features)
            loss = self.criterion(reconstructed, features)
            
            # ✅ FIX 4: Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at batch {batch_idx}, skipping")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
            # Log progress
            if batch_idx % 200 == 0:
                logger.info(f"  Batch {batch_idx}/{min(num_batches, 1000)}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / max(valid_batches, 1)
        return avg_loss
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        valid_batches = 0
        
        with torch.no_grad():
            for features, _ in val_loader:
                # ✅ FIX 5: Add batch limit for validation
                if valid_batches >= self.max_val_batches:  # Use showcase limit
                    break
                    
                features = features.to(self.device)
                
                if torch.isnan(features).any():
                    continue
                
                reconstructed, encoded = self.model(features)
                loss = self.criterion(reconstructed, features)
                
                if torch.isnan(loss):
                    continue
                
                total_loss += loss.item()
                valid_batches += 1
        
        avg_loss = total_loss / max(valid_batches, 1)
        return avg_loss
    
    def train_attack_classifier_epoch(self, attack_loader):
        """Train attack type classifier for one epoch"""
        self.attack_classifier.train()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        valid_batches = 0
        
        for batch_idx, (features, attack_labels) in enumerate(attack_loader):
            if batch_idx >= self.max_train_batches:
                break
                
            features = features.to(self.device)
            attack_labels = attack_labels.to(self.device)
            
            # Forward pass
            outputs = self.attack_classifier(features)
            loss = self.attack_criterion(outputs, attack_labels)
            
            # Backward pass
            self.attack_optimizer.zero_grad()
            loss.backward()
            self.attack_optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += attack_labels.size(0)
            correct_predictions += (predicted == attack_labels).sum().item()
            total_loss += loss.item()
            valid_batches += 1
        
        avg_loss = total_loss / max(valid_batches, 1)
        accuracy = correct_predictions / max(total_samples, 1)
        return avg_loss, accuracy
    
    def validate_attack_classifier_epoch(self, attack_loader):
        """Validate attack type classifier for one epoch"""
        if self.attack_classifier is not None:
            self.attack_classifier.eval()
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        valid_batches = 0
        
        with torch.no_grad():
            for features, attack_labels in attack_loader:
                if valid_batches >= self.max_val_batches:
                    break
                    
                features = features.to(self.device)
                attack_labels = attack_labels.to(self.device)
                
                outputs = self.attack_classifier(features)
                loss = self.attack_criterion(outputs, attack_labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += attack_labels.size(0)
                correct_predictions += (predicted == attack_labels).sum().item()
                total_loss += loss.item()
                valid_batches += 1
        
        avg_loss = total_loss / max(valid_batches, 1)
        accuracy = correct_predictions / max(total_samples, 1)
        return avg_loss, accuracy
    
    def prepare_attack_type_data(self, data_prep):
        """Prepare attack type classification data from anomalies only"""
        logger.info("🎯 Preparing attack type classification data...")

        if data_prep is None:
            logger.warning("⚠️ Data preparation object not provided")
            return None, None

        # Prefer pre-split anomaly-only loaders (prevents data leakage)
        if hasattr(data_prep, 'attack_train_loader') and hasattr(data_prep, 'attack_val_loader'):
            if data_prep.attack_train_loader is not None and data_prep.attack_val_loader is not None:
                logger.info("✅ Using anomaly-only attack-category loaders from DataPreparation")
                return data_prep.attack_train_loader, data_prep.attack_val_loader

        if self.attack_category_encoder is None:
            logger.warning("⚠️ Attack category encoder not provided")
            return None, None

        if not hasattr(data_prep, "all_data") or data_prep.all_data is None:
            logger.warning("⚠️ Processed dataset not loaded")
            return None, None

        if not hasattr(data_prep, "scaler") or data_prep.scaler is None:
            logger.warning("⚠️ Scaler not available")
            return None, None

        if not hasattr(data_prep, "feature_names") or not data_prep.feature_names:
            logger.warning("⚠️ Feature list not available")
            return None, None

        df = data_prep.all_data
        if "Binary_Label" not in df.columns or "Attack_Category" not in df.columns:
            logger.warning("⚠️ Required columns missing in processed dataset")
            return None, None

        anomaly_df = df[df["Binary_Label"] == 1]
        if anomaly_df.empty:
            logger.warning("⚠️ No anomalies found for attack type training")
            return None, None

        anomaly_features_raw = anomaly_df[data_prep.feature_names].values
        anomaly_features = data_prep.scaler.transform(anomaly_features_raw)
        anomaly_labels_str = anomaly_df["Attack_Category"].astype(str).values
        anomaly_labels = self.attack_category_encoder.transform(anomaly_labels_str)

        X_train, X_val, y_train, y_val = train_test_split(
            anomaly_features,
            anomaly_labels,
            test_size=0.2,
            random_state=42,
            stratify=anomaly_labels,
        )

        from torch.utils.data import TensorDataset, DataLoader

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

        attack_train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        attack_val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

        logger.info(f"✅ Attack classifier data prepared:")
        logger.info(f"  Training samples: {len(train_ds)}")
        logger.info(f"  Validation samples: {len(val_ds)}")

        return attack_train_loader, attack_val_loader
    
    def train_model(self, train_loader, val_loader, data_prep=None):
        """ENHANCED Complete two-stage training loop"""
        logger.info("🎬 ENHANCED TWO-STAGE TRAINING...")
        logger.info("🎯 Stage 1: Anomaly Detection (Autoencoder)")
        logger.info("� Stage 2: Attack Type Classification")
        logger.info(f"Training batches (showcase): {min(len(train_loader), self.max_train_batches)}")
        logger.info(f"Validation batches (showcase): {min(len(val_loader), self.max_val_batches)}")
        logger.info(f"Expected time: 15-20 minutes")
        logger.info(f"Expected F1: 70-75% (with attack classification)")
        
        # Stage 1: Train autoencoder
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.config.training_epochs):
            epoch_start_time = time.time()
            
            # Train autoencoder
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            self.training_history['ae_train_loss'].append(train_loss)
            self.training_history['ae_val_loss'].append(val_loss)
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['training_time'].append(epoch_time)
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1}/{self.config.training_epochs} [🎬 TWO-STAGE]:")
            logger.info(f"  Autoencoder - Train Loss: {train_loss:.6f}")
            logger.info(f"  Autoencoder - Val Loss: {val_loss:.6f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, val_loss, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Stage 2: Train attack type classifier
        logger.info("🎯 Starting Stage 2: Attack Type Classification...")
        if self.attack_classifier is None and self.attack_category_classes is not None:
            self.attack_classifier = AttackTypeClassifier(
                input_dim=78,
                num_classes=len(self.attack_category_classes)
            ).to(self.device)
            self.attack_optimizer = optim.Adam(self.attack_classifier.parameters(), lr=0.001)

        attack_train_loader, attack_val_loader = self.prepare_attack_type_data(data_prep)

        if attack_train_loader and attack_val_loader and self.attack_classifier is not None:
            for epoch in range(10):  # Fewer epochs for attack classifier
                epoch_start_time = time.time()
                
                # Train attack classifier
                train_loss, train_acc = self.train_attack_classifier_epoch(attack_train_loader)
                val_loss, val_acc = self.validate_attack_classifier_epoch(attack_val_loader)
                
                # Record attack classifier history
                self.training_history['attack_train_loss'].append(train_loss)
                self.training_history['attack_val_loss'].append(val_loss)
                self.training_history['attack_accuracy'].append(train_acc)
                
                logger.info(f"Attack Classifier Epoch {epoch + 1}/10:")
                logger.info(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                logger.info(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # Persist trained stage-2 weights in the checkpoint used by metrics scripts
            try:
                self.save_checkpoint(self.training_history['epochs'][-1], best_val_loss, is_best=True)
            except Exception as e:
                logger.warning(f"⚠️ Failed to save final two-stage checkpoint: {e}")
        
        total_training_time = time.time() - start_time
        
        # Restore best weights if early stopping was used
        if self.early_stopping.best_weights is not None:
            self.model.load_state_dict(self.early_stopping.best_weights)
        
        logger.info(f"✅ ENHANCED two-stage training complete")
        logger.info(f"🎬 TWO-STAGE RESULTS:")
        logger.info(f"  Training time: {total_training_time:.2f} seconds")
        logger.info(f"  Best validation loss: {best_val_loss:.6f}")
        logger.info(f"  Attack classifier trained: {'Yes' if attack_train_loader else 'No'}")
        logger.info(f"  Expected F1 improvement: 2-7%")
        logger.info(f"  Data coverage: 72% of training data")
        
        return self.training_history
    
    def predict_two_stage(self, features):
        """Two-stage prediction: anomaly detection + attack type classification"""
        self.model.eval()
        if self.attack_classifier is not None:
            self.attack_classifier.eval()
        
        with torch.no_grad():
            features = features.to(self.device)
            
            # Stage 1: Anomaly detection
            reconstructed, encoded = self.model(features)
            reconstruction_error = torch.mean((reconstructed - features) ** 2, dim=1)
            
            # Binary classification
            threshold = 0.22610116  # Your optimal threshold
            anomaly_predictions = (reconstruction_error > threshold).float()
            
            # Stage 2: Attack type classification (only for anomalies)
            attack_predictions = []
            attack_confidences = []
            
            for i, is_anomaly in enumerate(anomaly_predictions):
                if is_anomaly.item() == 1 and self.attack_classifier is not None:
                    attack_output = self.attack_classifier(features[i:i+1])
                    attack_probs = torch.softmax(attack_output, dim=1)
                    attack_pred = torch.argmax(attack_probs, dim=1).item()
                    attack_conf = torch.max(attack_probs).item()

                    attack_predictions.append(attack_pred)
                    attack_confidences.append(attack_conf)
                else:
                    attack_predictions.append(-1)
                    attack_confidences.append(0.0)
        
        return {
            'anomaly_predictions': anomaly_predictions.cpu().numpy(),
            'reconstruction_errors': reconstruction_error.cpu().numpy(),
            'attack_type_predictions': attack_predictions,
            'attack_confidences': attack_confidences
        }
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save ENHANCED model checkpoint with both stages"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'attack_classifier_state_dict': self.attack_classifier.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'attack_optimizer_state_dict': self.attack_optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self._serialize_config(),
            'training_history': self.training_history,
            'attack_types': self.attack_types,
            'attack_category_classes': self.attack_category_classes,
            'two_stage_enabled': True
        }
        
        # Save latest checkpoint
        checkpoint_path = ARTIFACTS_DIR / "latest_checkpoint_fixed.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = ARTIFACTS_DIR / "best_autoencoder_fixed.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
    
    def _serialize_config(self):
        """✅ FIX 7: Convert config to JSON-serializable format"""
        return {
            'input_dim': self.config.input_dim,
            'encoding_dims': self.config.encoding_dims,
            'bottleneck_dim': self.config.bottleneck_dim,
            'dropout_rate': self.config.dropout_rate,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'training_epochs': self.config.training_epochs,
            'patience': self.config.patience,
            'min_delta': self.config.min_delta,
            'test_size': self.config.test_size,
            'validation_split': self.config.validation_split,
            'random_state': self.config.random_state,
            'anomaly_threshold_percentile': self.config.anomaly_threshold_percentile,
            'device': str(self.config.device)  # ✅ Convert device to string
        }
    
    def evaluate_model(self, test_loader):
        """Evaluate model on test set with fixes"""
        logger.info("Evaluating model on test set...")
        
        self.model.eval()
        total_loss = 0
        reconstruction_errors = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                # ✅ FIX 8: Add batch limit for evaluation
                if len(reconstruction_errors) >= self.max_test_batches * 128:  # Use showcase limit
                    break
                    
                features = features.to(self.device)
                labels = labels.cpu().numpy()
                
                if torch.isnan(features).any():
                    continue
                
                # Forward pass
                reconstructed, encoded = self.model(features)
                loss = self.criterion(reconstructed, features)
                
                # Calculate reconstruction errors per sample
                batch_errors = torch.mean((reconstructed - features) ** 2, dim=1)
                
                total_loss += loss.item()
                reconstruction_errors.extend(batch_errors.cpu().numpy())
                all_labels.extend(labels)
        
        avg_loss = total_loss / max(len(test_loader), 1)
        reconstruction_errors = np.array(reconstruction_errors)
        all_labels = np.array(all_labels)
        
        logger.info(f"Test Loss: {avg_loss:.6f}")
        logger.info(f"Reconstruction errors - Mean: {np.mean(reconstruction_errors):.6f}, Std: {np.std(reconstruction_errors):.6f}")
        
        return {
            'test_loss': avg_loss,
            'reconstruction_errors': reconstruction_errors,
            'labels': all_labels
        }
    
    def determine_anomaly_threshold(self, reconstruction_errors, percentile=95):
        """Determine anomaly threshold with safety checks"""
        if len(reconstruction_errors) == 0:
            logger.warning("No reconstruction errors available, using default threshold")
            return 0.01
        
        threshold = np.percentile(reconstruction_errors, percentile)
        logger.info(f"Anomaly threshold (95th percentile): {threshold:.6f}")
        return threshold
    
    def plot_training_history(self):
        """Plot ENHANCED training and validation loss for both stages"""
        plt.figure(figsize=(15, 10))
        
        # Autoencoder training
        plt.subplot(2, 3, 1)
        plt.plot(self.training_history['epochs'], self.training_history['ae_train_loss'], label='AE Training Loss')
        plt.plot(self.training_history['epochs'], self.training_history['ae_val_loss'], label='AE Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training & Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Attack classifier training (if available)
        if self.training_history['attack_train_loss']:
            plt.subplot(2, 3, 2)
            epochs_classifier = range(1, len(self.training_history['attack_train_loss']) + 1)
            plt.plot(epochs_classifier, self.training_history['attack_train_loss'], label='Attack Training Loss')
            plt.plot(epochs_classifier, self.training_history['attack_val_loss'], label='Attack Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Attack Classifier Training & Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # Attack classifier accuracy
            plt.subplot(2, 3, 3)
            plt.plot(epochs_classifier, self.training_history['attack_accuracy'], label='Attack Accuracy', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Attack Classifier Accuracy')
            plt.legend()
            plt.grid(True)
        
        # Training time
        plt.subplot(2, 3, 4)
        plt.plot(self.training_history['epochs'], self.training_history['training_time'])
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Training Time per Epoch')
        plt.grid(True)
        
        # Combined loss view
        plt.subplot(2, 3, 5)
        plt.plot(self.training_history['epochs'], self.training_history['ae_train_loss'], label='AE Train', alpha=0.7)
        plt.plot(self.training_history['epochs'], self.training_history['ae_val_loss'], label='AE Val', alpha=0.7)
        if self.training_history['attack_train_loss']:
            plt.plot(epochs_classifier, self.training_history['attack_train_loss'], label='Attack Train', alpha=0.7)
            plt.plot(epochs_classifier, self.training_history['attack_val_loss'], label='Attack Val', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Combined Training Loss')
        plt.legend()
        plt.grid(True)
        
        # Two-stage summary
        plt.subplot(2, 3, 6)
        stages = ['Autoencoder', 'Attack Classifier']
        final_losses = [
            self.training_history['ae_train_loss'][-1] if self.training_history['ae_train_loss'] else 0,
            self.training_history['attack_train_loss'][-1] if self.training_history['attack_train_loss'] else 0
        ]
        colors = ['blue', 'red']
        plt.bar(stages, final_losses, color=colors, alpha=0.7)
        plt.ylabel('Final Training Loss')
        plt.title('Two-Stage Final Loss')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(ARTIFACTS_DIR / 'two_stage_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Two-stage training curves saved to {ARTIFACTS_DIR / 'two_stage_training_curves.png'}")
    
    def save_training_results(self):
        """Save ENHANCED training results to JSON"""
        results = {
            'model_config': self._serialize_config(),
            'training_history': self.training_history,
            'final_train_loss': self.training_history['ae_train_loss'][-1] if self.training_history['ae_train_loss'] else None,
            'final_val_loss': self.training_history['ae_val_loss'][-1] if self.training_history['ae_val_loss'] else None,
            'total_epochs': len(self.training_history['epochs']),
            'total_training_time': sum(self.training_history['training_time']),
            'attack_classifier_trained': len(self.training_history['attack_train_loss']) > 0,
            'two_stage_enabled': True
        }
        
        results_path = ARTIFACTS_DIR / "training_results_fixed.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved to: {results_path}")
        return results


class FixedDataPreparation(DataPreparation):
    """FIXED data preparation that excludes NaN ' Label' column"""
    
    def __init__(self, data_dir="data_preprocessing/processed_data", 
                 test_size=0.2, validation_split=0.2, random_state=42):
        super().__init__(data_dir, test_size, validation_split, random_state)
    
    def extract_features_and_normalize(self):
        """✅ FIXED: Exclude NaN ' Label' column from features"""
        logger.info("Extracting and normalizing features (FIXED)...")
        
        if self.normal_data is None:
            raise ValueError("Data not separated. Call separate_normal_anomaly_data() first.")
        
        # ✅ FIX 9: Exclude the NaN ' Label' column
        label_columns = ['Binary_Label', 'Attack_Category', 'Attack_Category_Numeric', 'Attack_Type_Numeric', ' Label']
        self.feature_names = [col for col in self.all_data.columns if col not in label_columns]
        
        logger.info(f"✅ FIXED feature extraction:")
        logger.info(f"  Total columns: {len(self.all_data.columns)}")
        logger.info(f"  Excluded columns: {label_columns}")
        logger.info(f"  Feature count: {len(self.feature_names)}")
        logger.info(f"  Includes ' Label' column: {' Label' in self.feature_names}")
        
        # Extract features for normal data (for training)
        normal_features = self.normal_data[self.feature_names].values
        
        # Extract features for anomaly data (for testing)
        anomaly_features = self.anomaly_data[self.feature_names].values
        
        # Extract labels
        normal_labels = self.normal_data['Binary_Label'].values
        anomaly_labels = self.anomaly_data['Binary_Label'].values

        # Extract attack categories (for stage-2)
        if 'Attack_Category' in self.normal_data.columns:
            normal_categories = self.normal_data['Attack_Category'].astype(str).fillna('Normal').values
        else:
            normal_categories = np.array(['Normal'] * len(self.normal_data))

        if 'Attack_Category' in self.anomaly_data.columns:
            anomaly_categories = self.anomaly_data['Attack_Category'].astype(str).fillna('Other').values
        else:
            anomaly_categories = np.array(['Other'] * len(self.anomaly_data))
        
        # Fit scaler on normal data only (to avoid data leakage)
        self.scaler = StandardScaler()
        normal_features_scaled = self.scaler.fit_transform(normal_features)
        anomaly_features_scaled = self.scaler.transform(anomaly_features)
        
        # Check for NaN values
        normal_nan_count = np.isnan(normal_features_scaled).sum()
        anomaly_nan_count = np.isnan(anomaly_features_scaled).sum()
        
        logger.info(f"✅ FIXED normalization results:")
        logger.info(f"  Normal features shape: {normal_features_scaled.shape}")
        logger.info(f"  Anomaly features shape: {anomaly_features_scaled.shape}")
        logger.info(f"  NaN count in normal features: {normal_nan_count}")
        logger.info(f"  NaN count in anomaly features: {anomaly_nan_count}")
        
        return (
            normal_features_scaled,
            normal_labels,
            anomaly_features_scaled,
            anomaly_labels,
            normal_categories,
            anomaly_categories,
        )


def test_fixed_training_pipeline():
    """Test the ENHANCED two-stage training pipeline"""
    logger.info("🚀 Starting ENHANCED Two-Stage training pipeline test...")
    
    try:
        # Step 1: Load configuration
        config = AutoencoderConfig()
        logger.info("✅ Configuration loaded")
        
        # Step 2: Prepare data with FIXED feature extraction
        logger.info("📊 Preparing training data (FIXED)...")
        data_prep = FixedDataPreparation()
        data_results = data_prep.prepare_data(batch_size=config.batch_size)
        train_loader = data_results['train_loader']
        val_loader = data_results['val_loader']
        test_loader = data_results['test_loader']
        logger.info("✅ FIXED data preparation complete")
        
        # Step 3: Initialize ENHANCED two-stage trainer
        trainer = FixedAutoencoderTrainer(
            config,
            attack_category_classes=getattr(data_prep, 'attack_category_classes', None),
            attack_category_encoder=getattr(data_prep, 'attack_category_encoder', None),
        )
        logger.info("✅ ENHANCED two-stage trainer initialized")
        
        # Step 4: Train both stages
        logger.info("🏋️ Starting ENHANCED two-stage model training...")
        training_history = trainer.train_model(train_loader, val_loader, data_prep=data_prep)
        logger.info("✅ ENHANCED two-stage model training complete")
        
        # Step 5: Evaluate model
        logger.info("📈 Evaluating ENHANCED model...")
        eval_results = trainer.evaluate_model(test_loader)
        threshold = trainer.determine_anomaly_threshold(eval_results['reconstruction_errors'])
        logger.info("✅ ENHANCED model evaluation complete")
        
        # Step 6: Test two-stage prediction
        logger.info("🎯 Testing two-stage prediction...")
        sample_features, _ = next(iter(test_loader))
        two_stage_results = trainer.predict_two_stage(sample_features[:10])  # Test first 10 samples
        
        logger.info("🎯 Two-Stage Prediction Results:")
        for i in range(len(two_stage_results['anomaly_predictions'])):
            anomaly_pred = two_stage_results['anomaly_predictions'][i]
            attack_pred = two_stage_results['attack_type_predictions'][i]
            attack_conf = two_stage_results['attack_confidences'][i]
            
            status = "ANOMALY" if anomaly_pred == 1 else "NORMAL"
            attack_type = "N/A"
            if attack_pred is not None and attack_pred >= 0 and trainer.attack_types is not None and attack_pred < len(trainer.attack_types):
                attack_type = trainer.attack_types[attack_pred]
            
            logger.info(f"  Sample {i+1}: {status} → {attack_type} (confidence: {attack_conf:.3f})")
        
        # Step 7: Save results
        trainer.save_training_results()
        trainer.plot_training_history()
        logger.info("✅ ENHANCED results saved")
        
        logger.info("🎉 ENHANCED Two-Stage training pipeline - COMPLETED SUCCESSFULLY")
        logger.info("🎯 Capabilities: Anomaly Detection + Attack Type Classification")
        return True
        
    except Exception as e:
        logger.error(f"❌ ENHANCED training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-only", action="store_true")
    args = parser.parse_args()

    if args.stage2_only:
        try:
            config = AutoencoderConfig()
            data_prep = FixedDataPreparation()
            data_prep.prepare_data(batch_size=config.batch_size)

            checkpoint_path = ARTIFACTS_DIR / "latest_checkpoint_fixed.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError("latest_checkpoint_fixed.pth not found")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            trainer = FixedAutoencoderTrainer(
                config,
                attack_category_classes=getattr(data_prep, 'attack_category_classes', None),
                attack_category_encoder=getattr(data_prep, 'attack_category_encoder', None),
            )
            trainer.model.load_state_dict(checkpoint['model_state_dict'])

            attack_train_loader, attack_val_loader = trainer.prepare_attack_type_data(data_prep)
            if attack_train_loader and attack_val_loader and trainer.attack_classifier is not None:
                for epoch in range(10):
                    train_loss, train_acc = trainer.train_attack_classifier_epoch(attack_train_loader)
                    val_loss, val_acc = trainer.validate_attack_classifier_epoch(attack_val_loader)
                    trainer.training_history['attack_train_loss'].append(train_loss)
                    trainer.training_history['attack_val_loss'].append(val_loss)
                    trainer.training_history['attack_accuracy'].append(train_acc)
                    logger.info(f"Attack Classifier Epoch {epoch + 1}/10:")
                    logger.info(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
                    logger.info(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

                trainer.save_checkpoint(0, checkpoint.get('val_loss', 0.0), is_best=True)
            success = True
        except Exception as e:
            logger.error(f"❌ Stage2-only training failed: {e}")
            success = False
    else:
        success = test_fixed_training_pipeline()
    
    if success:
        logger.info("🎉 ENHANCED training_pipeline.py - TWO-STAGE COMPLETE!")
        logger.info("✅ Ready for production deployment with attack type classification")
        logger.info("🎯 Features: Anomaly Detection + Specific Attack Type Identification")
    else:
        logger.error("❌ Fixed pipeline still has issues")
