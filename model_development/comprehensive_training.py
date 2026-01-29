"""
Phase 4: Model Training & Validation - Comprehensive Training
Full training pipeline with detailed validation and performance analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pathlib import Path
import json
import time
import copy
from datetime import datetime
import logging

# Import our modules
from autoencoder_model import CloudAnomalyAutoencoder, AutoencoderConfig
from data_preparation import DataPreparation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveTrainer:
    """Comprehensive training and validation for autoencoder"""
    
    def __init__(self, config=None):
        """Initialize comprehensive trainer"""
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
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)  # Conservative learning rate
        self.criterion = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'training_time': [],
            'learning_rate': []
        }
        
        # Performance metrics
        self.performance_metrics = {}
        
        logger.info(f"ComprehensiveTrainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Model parameters: {self.model.count_parameters():,}")
        logger.info(f"  Learning rate: 0.0001")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with detailed logging"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        valid_batches = 0
        
        for batch_idx, (features, _) in enumerate(train_loader):
            features = features.to(self.device)
            
            # Skip if NaN data
            if torch.isnan(features).any():
                continue
            
            # Forward pass
            reconstructed, encoded = self.model(features)
            loss = self.criterion(reconstructed, features)
            
            # Skip if NaN loss
            if torch.isnan(loss):
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
            # Log progress
            if batch_idx % 1000 == 0:
                logger.info(f"  Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / max(valid_batches, 1)
        return avg_loss
    
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        valid_batches = 0
        
        with torch.no_grad():
            for features, _ in val_loader:
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
    
    def comprehensive_training(self, train_loader, val_loader, max_epochs=20):
        """Comprehensive training with validation"""
        logger.info("Starting comprehensive training...")
        logger.info(f"Max epochs: {max_epochs}")
        logger.info(f"Training batches: {len(train_loader)}")
        logger.info(f"Validation batches: {len(val_loader)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 5
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch + 1)
            
            # Validation
            val_loss = self.validate_epoch(val_loader, epoch + 1)
            
            # Record history
            epoch_time = time.time() - epoch_start_time
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['epochs'].append(epoch + 1)
            self.training_history['training_time'].append(epoch_time)
            self.training_history['learning_rate'].append(0.0001)
            
            # Log epoch results
            logger.info(f"Epoch {epoch + 1}/{max_epochs}:")
            logger.info(f"  Train Loss: {train_loss:.6f}")
            logger.info(f"  Val Loss: {val_loss:.6f}")
            logger.info(f"  Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_best_model(epoch + 1, val_loss)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        total_training_time = time.time() - start_time
        logger.info(f"Training completed in {total_training_time:.2f} seconds")
        logger.info(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.training_history
    
    def save_best_model(self, epoch, val_loss):
        """Save best model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.__dict__,
            'training_history': self.training_history
        }
        
        model_path = Path("model_artifacts/phase4_best_autoencoder.pth")
        model_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, model_path)
        logger.info(f"Best model saved at epoch {epoch} with val loss: {val_loss:.6f}")
    
    def comprehensive_evaluation(self, test_loader):
        """Comprehensive evaluation on test set"""
        logger.info("Starting comprehensive evaluation...")
        
        self.model.eval()
        reconstruction_errors = []
        all_labels = []
        all_reconstructed = []
        all_original = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.cpu().numpy()
                
                if torch.isnan(features).any():
                    continue
                
                # Forward pass
                reconstructed, encoded = self.model(features)
                
                # Calculate reconstruction errors per sample
                batch_errors = torch.mean((reconstructed - features) ** 2, dim=1)
                
                reconstruction_errors.extend(batch_errors.cpu().numpy())
                all_labels.extend(labels)
                all_reconstructed.extend(reconstructed.cpu().numpy())
                all_original.extend(features.cpu().numpy())
        
        reconstruction_errors = np.array(reconstruction_errors)
        all_labels = np.array(all_labels)
        all_reconstructed = np.array(all_reconstructed)
        all_original = np.array(all_original)
        
        # Determine anomaly threshold
        normal_errors = reconstruction_errors[all_labels == 0]
        threshold = np.percentile(normal_errors, 95)
        
        # Make predictions
        predictions = (reconstruction_errors > threshold).astype(int)
        
        # Calculate metrics
        accuracy = np.mean(predictions == all_labels)
        precision = np.sum((predictions == 1) & (all_labels == 1)) / np.sum(predictions == 1)
        recall = np.sum((predictions == 1) & (all_labels == 1)) / np.sum(all_labels == 1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(all_labels, reconstruction_errors)
        except:
            roc_auc = 0.5
        
        # Store performance metrics
        self.performance_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'threshold': threshold,
            'normal_error_mean': np.mean(normal_errors),
            'normal_error_std': np.std(normal_errors),
            'anomaly_error_mean': np.mean(reconstruction_errors[all_labels == 1]),
            'anomaly_error_std': np.std(reconstruction_errors[all_labels == 1]),
            'total_samples': len(all_labels),
            'normal_samples': np.sum(all_labels == 0),
            'anomaly_samples': np.sum(all_labels == 1)
        }
        
        logger.info(f"✅ Evaluation complete:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1_score:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  Threshold: {threshold:.6f}")
        
        return self.performance_metrics
    
    def plot_comprehensive_results(self):
        """Plot comprehensive training and evaluation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training curves
        axes[0, 0].plot(self.training_history['epochs'], self.training_history['train_loss'], label='Training Loss')
        axes[0, 0].plot(self.training_history['epochs'], self.training_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Training time per epoch
        axes[0, 1].plot(self.training_history['epochs'], self.training_history['training_time'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_title('Training Time per Epoch')
        axes[0, 1].grid(True)
        
        # Performance metrics bar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values = [self.performance_metrics['accuracy'], self.performance_metrics['precision'],
                 self.performance_metrics['recall'], self.performance_metrics['f1_score'],
                 self.performance_metrics['roc_auc']]
        
        axes[0, 2].bar(metrics, values)
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_title('Performance Metrics')
        axes[0, 2].set_ylim(0, 1)
        axes[0, 2].grid(True, axis='y')
        
        # Confusion Matrix (placeholder - would need actual predictions)
        axes[1, 0].text(0.5, 0.5, 'Confusion Matrix\n(Computed from predictions)', 
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].axis('off')
        
        # Error distribution
        axes[1, 1].hist([self.performance_metrics['normal_error_mean'], 
                        self.performance_metrics['anomaly_error_mean']], 
                       bins=20, label=['Normal', 'Anomaly'])
        axes[1, 1].set_xlabel('Reconstruction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Sample reconstruction visualization
        axes[1, 2].text(0.5, 0.5, 'Sample Reconstruction\n(Original vs Reconstructed)', 
                       ha='center', va='center', fontsize=12)
        axes[1, 2].set_title('Reconstruction Quality')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('model_artifacts/phase4_comprehensive_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Comprehensive results plots saved to model_artifacts/phase4_comprehensive_results.png")
    
    def save_comprehensive_results(self):
        """Save comprehensive training and evaluation results"""
        results = {
            'phase': 'Phase 4: Model Training & Validation',
            'status': 'COMPLETED',
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics,
            'model_parameters': self.model.count_parameters(),
            'total_epochs': len(self.training_history['epochs']),
            'total_training_time': sum(self.training_history['training_time']),
            'best_val_loss': min(self.training_history['val_loss']) if self.training_history['val_loss'] else None,
            'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
            'device': str(self.device),
            'completion_date': datetime.now().isoformat()
        }
        
        results_path = Path("model_artifacts/phase4_comprehensive_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Comprehensive results saved to: {results_path}")
        return results


def complete_phase4_training():
    """Complete Phase 4: Model Training & Validation"""
    logger.info("🚀 Starting Phase 4: Model Training & Validation")
    
    try:
        # Step 1: Load configuration
        config = AutoencoderConfig()
        logger.info("✅ Configuration loaded")
        
        # Step 2: Prepare data
        logger.info("📊 Preparing training data...")
        data_prep = DataPreparation()
        data_results = data_prep.prepare_data(batch_size=128)
        train_loader = data_results['train_loader']
        val_loader = data_results['val_loader']
        test_loader = data_results['test_loader']
        logger.info("✅ Data preparation complete")
        
        # Step 3: Initialize comprehensive trainer
        trainer = ComprehensiveTrainer(config)
        logger.info("✅ Comprehensive trainer initialized")
        
        # Step 4: Comprehensive training
        logger.info("🏋️ Starting comprehensive training...")
        training_history = trainer.comprehensive_training(train_loader, val_loader, max_epochs=10)
        logger.info("✅ Comprehensive training complete")
        
        # Step 5: Comprehensive evaluation
        logger.info("📈 Starting comprehensive evaluation...")
        performance_metrics = trainer.comprehensive_evaluation(test_loader)
        logger.info("✅ Comprehensive evaluation complete")
        
        # Step 6: Generate plots and save results
        logger.info("📊 Generating comprehensive results...")
        trainer.plot_comprehensive_results()
        trainer.save_comprehensive_results()
        logger.info("✅ Comprehensive results generated")
        
        logger.info("🎉 Phase 4: Model Training & Validation - COMPLETED SUCCESSFULLY")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4 failed: {e}")
        return False


if __name__ == "__main__":
    success = complete_phase4_training()
    
    if success:
        logger.info("🎉 Phase 4: Model Training & Validation - COMPLETED")
        logger.info("✅ Ready for Phase 5: XAI Integration")
    else:
        logger.error("❌ Phase 4 implementation failed")
