"""
Phase 3: Autoencoder Implementation & Training Pipeline
Complete training system for Cloud Anomaly Detection using FL and XAI
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
import copy
from datetime import datetime
import logging
import warnings

# Import our modules
from autoencoder_model import CloudAnomalyAutoencoder, AutoencoderConfig
from data_preparation import DataPreparation

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
            
        return self.counter >= self.patience


class AutoencoderTrainer:
    """Complete training pipeline for autoencoder"""
    
    def __init__(self, config=None):
        """
        Initialize trainer
        
        Args:
            config (AutoencoderConfig): Training configuration
        """
        self.config = config if config else AutoencoderConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = CloudAnomalyAutoencoder(
            input_dim=self.config.input_dim,
            encoding_dims=self.config.encoding_dims,
            bottleneck_dim=self.config.bottleneck_dim,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Add gradient clipping to prevent exploding gradients
        self.max_grad_norm = 1.0
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'training_time': []
        }
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta
        )
        
        logger.info(f"AutoencoderTrainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model parameters: {self.model.count_parameters():,}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Epochs: {self.config.epochs}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, (features, _) in enumerate(train_loader):
            features = features.to(self.device)
            
            # Check for NaN data
            if torch.isnan(features).any():
                logger.warning(f"NaN data detected at batch {batch_idx}, skipping")
                continue
            
            # Forward pass
            reconstructed, encoded = self.model(features)
            loss = self.criterion(reconstructed, features)
            
            # Check for NaN loss
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
            
            # Log progress
            if batch_idx % 500 == 0:
                logger.info(f"  Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for features, _ in val_loader:
                features = features.to(self.device)
                
                # Forward pass
                reconstructed, encoded = self.model(features)
                loss = self.criterion(reconstructed, features)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_model(self, train_loader, val_loader):
        """Complete training loop"""
        logger.info("Starting model training...")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate_epoch(val_loader)
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['training_time'].append(epoch_time)
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_loss:.6f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch + 1, val_loss, is_best=True)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_training_time = time.time() - start_time
        
        # Restore best weights if early stopping was used
        if self.early_stopping.best_weights is not None:
            self.model.load_state_dict(self.early_stopping.best_weights)
            logger.info("Restored best model weights")
        
        logger.info(f"Training completed in {total_training_time:.2f} seconds")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.training_history
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        checkpoint_path = Path("model_artifacts/latest_checkpoint.pth")
        checkpoint_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path("model_artifacts/best_autoencoder.pth")
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with validation loss: {val_loss:.6f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        logger.info(f"Validation loss: {checkpoint['val_loss']:.6f}")
        
        return checkpoint
    
    def evaluate_model(self, test_loader):
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set...")
        
        self.model.eval()
        total_loss = 0
        reconstruction_errors = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.cpu().numpy()
                
                # Forward pass
                reconstructed, encoded = self.model(features)
                loss = self.criterion(reconstructed, features)
                
                # Calculate reconstruction errors per sample
                batch_errors = torch.mean((reconstructed - features) ** 2, dim=1)
                
                total_loss += loss.item()
                reconstruction_errors.extend(batch_errors.cpu().numpy())
                all_labels.extend(labels)
        
        avg_loss = total_loss / len(test_loader)
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
        """Determine anomaly threshold based on reconstruction errors"""
        threshold = np.percentile(reconstruction_errors, percentile)
        logger.info(f"Anomaly threshold (95th percentile): {threshold:.6f}")
        return threshold
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['epochs'], self.training_history['train_loss'], label='Training Loss')
        plt.plot(self.training_history['epochs'], self.training_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['epochs'], self.training_history['training_time'])
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Training Time per Epoch')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('model_artifacts/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Training curves saved to model_artifacts/training_curves.png")
    
    def save_training_results(self):
        """Save training results and configuration"""
        results = {
            'training_history': self.training_history,
            'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
            'total_epochs': len(self.training_history['epochs']),
            'total_training_time': sum(self.training_history['training_time']),
            'model_parameters': self.model.count_parameters(),
            'config': self.config.__dict__,
            'completion_date': datetime.now().isoformat()
        }
        
        results_path = Path("model_artifacts/training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved to: {results_path}")
        return results


def complete_phase3_implementation():
    """Complete Phase 3: Autoencoder Implementation"""
    logger.info("🚀 Starting Phase 3: Autoencoder Implementation")
    
    try:
        # Step 1: Load configuration
        config = AutoencoderConfig()
        logger.info("✅ Configuration loaded")
        
        # Step 2: Prepare data
        logger.info("📊 Preparing training data...")
        data_prep = DataPreparation()
        data_results = data_prep.prepare_data(batch_size=config.batch_size)
        train_loader = data_results['train_loader']
        val_loader = data_results['val_loader']
        test_loader = data_results['test_loader']
        logger.info("✅ Data preparation complete")
        
        # Step 3: Initialize trainer
        trainer = AutoencoderTrainer(config)
        logger.info("✅ Trainer initialized")
        
        # Step 4: Train model
        logger.info("🏋️ Starting model training...")
        training_history = trainer.train_model(train_loader, val_loader)
        logger.info("✅ Model training complete")
        
        # Step 5: Evaluate model
        logger.info("📈 Evaluating model...")
        eval_results = trainer.evaluate_model(test_loader)
        threshold = trainer.determine_anomaly_threshold(eval_results['reconstruction_errors'])
        logger.info("✅ Model evaluation complete")
        
        # Step 6: Save results
        trainer.save_training_results()
        trainer.plot_training_history()
        logger.info("✅ Results saved")
        
        # Step 7: Save final model
        final_model_path = Path("model_artifacts/final_autoencoder.pth")
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'config': config.__dict__,
            'threshold': threshold,
            'training_history': training_history
        }, final_model_path)
        logger.info(f"✅ Final model saved to: {final_model_path}")
        
        logger.info("🎉 Phase 3: Autoencoder Implementation - COMPLETED SUCCESSFULLY")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 3 failed: {e}")
        return False


if __name__ == "__main__":
    success = complete_phase3_implementation()
    
    if success:
        logger.info("🎉 Phase 3: Autoencoder Implementation - COMPLETED")
        logger.info("✅ Model is trained and ready for Phase 4: Model Training & Validation")
        logger.info("✅ Ready for Phase 5: XAI Integration")
    else:
        logger.error("❌ Phase 3 implementation failed")
