"""
Debug the comprehensive training evaluation error
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Import modules
import sys
sys.path.append('model_development')
from autoencoder_model import AutoencoderConfig
from data_preparation import DataPreparation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def debug_comprehensive_evaluation():
    """Debug the comprehensive evaluation error"""
    logger.info("🔍 Debugging Comprehensive Training Evaluation Error...")
    
    try:
        # Step 1: Load the trained model
        logger.info("📥 Loading trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint_path = Path("model_artifacts/phase4_best_autoencoder.pth")
        if not checkpoint_path.exists():
            logger.error("❌ Model not found!")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model
        from autoencoder_model import CloudAnomalyAutoencoder
        model = CloudAnomalyAutoencoder(
            input_dim=79,  # This is the issue - should be 78!
            encoding_dims=[64, 32, 16, 8],
            bottleneck_dim=4,
            dropout_rate=0.1
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"✅ Model loaded: 79 → 4 → 79")
        
        # Step 2: Load test data
        logger.info("📊 Loading test data...")
        data_prep = DataPreparation()
        data_results = data_prep.prepare_data(batch_size=128)
        test_loader = data_results['test_loader']
        
        logger.info(f"✅ Test data loaded: {len(test_loader)} batches")
        
        # Step 3: Debug the evaluation step by step
        logger.info("🔍 Debugging evaluation step by step...")
        
        reconstruction_errors = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(test_loader):
                if batch_idx >= 5:  # Just test first 5 batches
                    break
                    
                logger.info(f"  Processing batch {batch_idx}...")
                logger.info(f"    Features shape: {features.shape}")
                logger.info(f"    Labels shape: {labels.shape}")
                logger.info(f"    Unique labels: {torch.unique(labels)}")
                
                features = features.to(device)
                labels = labels.cpu().numpy()
                
                # Check for NaN
                if torch.isnan(features).any():
                    logger.warning(f"    NaN data found in batch {batch_idx}, skipping")
                    continue
                
                # Forward pass
                try:
                    reconstructed, encoded = model(features)
                    logger.info(f"    Forward pass successful: {reconstructed.shape}")
                except Exception as e:
                    logger.error(f"    ❌ Forward pass failed: {e}")
                    continue
                
                # Calculate reconstruction errors
                try:
                    batch_errors = torch.mean((reconstructed - features) ** 2, dim=1)
                    logger.info(f"    Batch errors shape: {batch_errors.shape}")
                    logger.info(f"    Batch errors range: {batch_errors.min():.6f} - {batch_errors.max():.6f}")
                except Exception as e:
                    logger.error(f"    ❌ Error calculation failed: {e}")
                    continue
                
                reconstruction_errors.extend(batch_errors.cpu().numpy())
                all_labels.extend(labels)
                
                logger.info(f"    Accumulated errors: {len(reconstruction_errors)}")
                logger.info(f"    Accumulated labels: {len(all_labels)}")
        
        # Step 4: Analyze the accumulated data
        logger.info("📊 Analyzing accumulated data...")
        reconstruction_errors = np.array(reconstruction_errors)
        all_labels = np.array(all_labels)
        
        logger.info(f"  Total reconstruction errors: {len(reconstruction_errors)}")
        logger.info(f"  Total labels: {len(all_labels)}")
        logger.info(f"  Unique labels in accumulated data: {np.unique(all_labels)}")
        logger.info(f"  Label distribution: {np.bincount(all_labels.astype(int))}")
        
        # Step 5: Debug the specific error location
        logger.info("🔍 Debugging the specific error location...")
        
        # This is where the error occurs
        normal_errors = reconstruction_errors[all_labels == 0]
        logger.info(f"  Normal errors shape: {normal_errors.shape}")
        logger.info(f"  Normal errors count: {len(normal_errors)}")
        
        if len(normal_errors) == 0:
            logger.error("❌ ERROR: No normal samples found in accumulated data!")
            logger.error("   This causes: np.percentile(normal_errors, 95) -> IndexError")
            logger.error("   Root cause: All processed batches might contain only anomalies")
            return False
        
        # Test the percentile calculation
        try:
            threshold = np.percentile(normal_errors, 95)
            logger.info(f"  ✅ Threshold calculation successful: {threshold}")
        except Exception as e:
            logger.error(f"  ❌ Threshold calculation failed: {e}")
            return False
        
        logger.info("🎉 Debugging completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Debugging failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = debug_comprehensive_evaluation()
    
    if success:
        logger.info("✅ Comprehensive training issue identified and can be fixed")
    else:
        logger.error("❌ Need further investigation")
