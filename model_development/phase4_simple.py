"""
Phase 4: Model Training & Validation - Simple Implementation
Complete Phase 4 with working training and basic validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
import logging

# Import our modules
from autoencoder_model import CloudAnomalyAutoencoder, AutoencoderConfig
from data_preparation import DataPreparation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def complete_phase4_simple():
    """Complete Phase 4 with simple working implementation"""
    logger.info("🚀 Starting Phase 4: Model Training & Validation (Simple)")
    
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
        
        # Step 3: Initialize model
        logger.info("🏗️ Initializing model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CloudAnomalyAutoencoder(
            input_dim=config.input_dim,
            encoding_dims=config.encoding_dims,
            bottleneck_dim=config.bottleneck_dim,
            dropout_rate=config.dropout_rate
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        
        logger.info(f"✅ Model initialized on {device}")
        logger.info(f"  Parameters: {model.count_parameters():,}")
        
        # Step 4: Training with validation
        logger.info("🏋️ Starting training with validation...")
        training_history = {'train_loss': [], 'val_loss': [], 'epochs': []}
        
        best_val_loss = float('inf')
        
        for epoch in range(8):  # 8 epochs for Phase 4
            # Training
            model.train()
            train_loss = 0
            train_batches = 0
            
            for batch_idx, (features, _) in enumerate(train_loader):
                if batch_idx >= 500:  # Limit to 500 batches for efficiency
                    break
                    
                features = features.to(device)
                
                if torch.isnan(features).any():
                    continue
                
                optimizer.zero_grad()
                reconstructed, encoded = model(features)
                loss = criterion(reconstructed, features)
                
                if torch.isnan(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / max(train_batches, 1)
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch_idx, (features, _) in enumerate(val_loader):
                    if batch_idx >= 200:  # Limit validation batches
                        break
                        
                    features = features.to(device)
                    
                    if torch.isnan(features).any():
                        continue
                    
                    reconstructed, encoded = model(features)
                    loss = criterion(reconstructed, features)
                    
                    if torch.isnan(loss):
                        continue
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / max(val_batches, 1)
            
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['epochs'].append(epoch + 1)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'epoch': epoch + 1,
                    'config': config.__dict__
                }, Path("model_artifacts/phase4_best_model.pth"))
            
            logger.info(f"Epoch {epoch+1}/8: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        logger.info("✅ Training complete")
        
        # Step 5: Basic evaluation
        logger.info("📈 Basic evaluation...")
        model.eval()
        test_loss = 0
        test_batches = 0
        
        with torch.no_grad():
            for batch_idx, (features, _) in enumerate(test_loader):
                if batch_idx >= 200:  # Limit test batches
                    break
                    
                features = features.to(device)
                
                if torch.isnan(features).any():
                    continue
                
                reconstructed, encoded = model(features)
                loss = criterion(reconstructed, features)
                
                if torch.isnan(loss):
                    continue
                
                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / max(test_batches, 1)
        logger.info(f"✅ Test Loss: {avg_test_loss:.6f}")
        
        # Step 6: Save results
        logger.info("💾 Saving Phase 4 results...")
        
        results = {
            'phase': 'Phase 4: Model Training & Validation',
            'status': 'COMPLETED',
            'model_parameters': model.count_parameters(),
            'training_epochs': len(training_history['epochs']),
            'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else None,
            'final_val_loss': training_history['val_loss'][-1] if training_history['val_loss'] else None,
            'best_val_loss': best_val_loss,
            'test_loss': avg_test_loss,
            'training_history': training_history,
            'device': str(device),
            'completion_date': datetime.now().isoformat()
        }
        
        results_path = Path("model_artifacts/phase4_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Results saved to: {results_path}")
        logger.info("🎉 Phase 4: Model Training & Validation - COMPLETED SUCCESSFULLY")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4 failed: {e}")
        return False


if __name__ == "__main__":
    success = complete_phase4_simple()
    
    if success:
        logger.info("🎉 Phase 4: Model Training & Validation - COMPLETED")
        logger.info("✅ Ready for Phase 5: XAI Integration")
    else:
        logger.error("❌ Phase 4 implementation failed")
