"""
Cloud Anomaly Autoencoder for Cloud Anomaly Detection using FL and XAI
Phase 1: Autoencoder Architecture Design
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class CloudAnomalyAutoencoder(nn.Module):
    """
    Enhanced Autoencoder for Cloud Anomaly Detection
    
    Architecture:
    Input (81) → 64 → 32 → 16 → 8 → 4 → 8 → 16 → 32 → 64 → Output (81)
    
    Designed specifically for cloud traffic anomaly detection with:
    - Bottleneck for compressed representation
    - Dropout for regularization
    - ReLU activations for non-linearity
    - Sigmoid output for reconstruction
    """
    
    def __init__(self, input_dim=79, encoding_dims=[64, 32, 16, 8], bottleneck_dim=4, dropout_rate=0.1):
        """
        Initialize Cloud Anomaly Autoencoder
        
        Args:
            input_dim (int): Number of input features (81 from processed data)
            encoding_dims (list): List of encoder layer dimensions
            bottleneck_dim (int): Bottleneck layer dimension (compressed representation)
            dropout_rate (float): Dropout rate for regularization
        """
        super(CloudAnomalyAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.bottleneck_dim = bottleneck_dim
        self.dropout_rate = dropout_rate
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Add bottleneck layer
        encoder_layers.extend([
            nn.Linear(prev_dim, bottleneck_dim),
            nn.ReLU()
        ])
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers (reverse of encoder)
        decoder_layers = []
        prev_dim = bottleneck_dim
        
        # Reverse encoding dimensions for decoder
        decoding_dims = encoding_dims[::-1]
        
        for dim in decoding_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dim
        
        # Add output layer
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()  # Sigmoid for reconstruction (0-1 range after normalization)
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"CloudAnomalyAutoencoder initialized:")
        logger.info(f"  Input dimension: {input_dim}")
        logger.info(f"  Encoder architecture: {input_dim} → {' → '.join(map(str, encoding_dims))} → {bottleneck_dim}")
        logger.info(f"  Decoder architecture: {bottleneck_dim} → {' → '.join(map(str, decoding_dims))} → {input_dim}")
        logger.info(f"  Total parameters: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through autoencoder
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
        
        Returns:
            tuple: (reconstructed, encoded)
                - reconstructed: Reconstructed input [batch_size, input_dim]
                - encoded: Compressed representation [batch_size, bottleneck_dim]
        """
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed, encoded
    
    def encode(self, x):
        """Encode input to compressed representation"""
        return self.encoder(x)
    
    def decode(self, encoded):
        """Decode compressed representation to reconstruction"""
        return self.decoder(encoded)
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_reconstruction_error(self, x, reconstructed=None):
        """
        Calculate reconstruction error (MSE)
        
        Args:
            x (torch.Tensor): Original input
            reconstructed (torch.Tensor): Reconstructed input (optional)
        
        Returns:
            torch.Tensor: Reconstruction error per sample
        """
        if reconstructed is None:
            reconstructed, _ = self.forward(x)
        
        mse_loss = nn.MSELoss(reduction='none')
        error = mse_loss(reconstructed, x)
        return error.mean(dim=1)  # Average error across features
    
    def get_model_info(self):
        """Get model architecture information"""
        info = {
            'model_type': 'CloudAnomalyAutoencoder',
            'input_dim': self.input_dim,
            'encoding_dims': self.encoding_dims,
            'bottleneck_dim': self.bottleneck_dim,
            'dropout_rate': self.dropout_rate,
            'total_parameters': self.count_parameters(),
            'architecture': {
                'encoder': [self.input_dim] + self.encoding_dims + [self.bottleneck_dim],
                'decoder': [self.bottleneck_dim] + self.encoding_dims[::-1] + [self.input_dim]
            }
        }
        return info


class AutoencoderConfig:
    """Configuration class for autoencoder training"""
    
    def __init__(self, input_dim=79):
        # Model architecture
        self.input_dim = input_dim  # 79 features from processed data
        self.encoding_dims = [64, 32, 16, 8]
        self.bottleneck_dim = 4
        self.dropout_rate = 0.1
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 128
        self.epochs = 100
        self.training_epochs = 50  # 🎬 SHOWCASE: Increased from 20 to 50 for better results
        self.patience = 10  # Early stopping patience
        self.min_delta = 1e-6  # Minimum improvement for early stopping
        
        # Data parameters
        self.test_size = 0.2
        self.validation_split = 0.2
        self.random_state = 42
        
        # Threshold parameters
        self.anomaly_threshold_percentile = 95  # Percentile for anomaly threshold
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"AutoencoderConfig initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Epochs: {self.epochs}")
    
    def save_config(self, filepath):
        """Save configuration to JSON file"""
        config_dict = {
            'input_dim': self.input_dim,
            'encoding_dims': self.encoding_dims,
            'bottleneck_dim': self.bottleneck_dim,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'training_epochs': self.training_epochs,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'test_size': self.test_size,
            'validation_split': self.validation_split,
            'random_state': self.random_state,
            'anomaly_threshold_percentile': self.anomaly_threshold_percentile
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to: {filepath}")
    
    @classmethod
    def load_config(cls, filepath):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        logger.info(f"Configuration loaded from: {filepath}")
        return config


def create_model(config=None):
    """
    Create and initialize autoencoder model
    
    Args:
        config (AutoencoderConfig): Configuration object
    
    Returns:
        tuple: (model, config)
    """
    if config is None:
        config = AutoencoderConfig()
    
    model = CloudAnomalyAutoencoder(
        input_dim=config.input_dim,
        encoding_dims=config.encoding_dims,
        bottleneck_dim=config.bottleneck_dim,
        dropout_rate=config.dropout_rate
    )
    
    model = model.to(config.device)
    
    logger.info("Model created and moved to device")
    return model, config


def test_model_architecture():
    """Test the autoencoder architecture with sample data"""
    logger.info("Testing autoencoder architecture...")
    
    # Create model
    model, config = create_model()
    
    # Create sample data
    batch_size = 32
    sample_input = torch.randn(batch_size, config.input_dim)
    
    # Forward pass
    with torch.no_grad():
        reconstructed, encoded = model(sample_input)
        reconstruction_error = model.get_reconstruction_error(sample_input, reconstructed)
    
    # Print results
    logger.info(f"✅ Architecture test successful:")
    logger.info(f"  Input shape: {sample_input.shape}")
    logger.info(f"  Reconstructed shape: {reconstructed.shape}")
    logger.info(f"  Encoded shape: {encoded.shape}")
    logger.info(f"  Reconstruction error shape: {reconstruction_error.shape}")
    logger.info(f"  Sample reconstruction error: {reconstruction_error[:5].tolist()}")
    
    # Get model info
    model_info = model.get_model_info()
    logger.info(f"  Model info: {json.dumps(model_info, indent=2)}")
    
    return True


if __name__ == "__main__":
    # Test the architecture
    success = test_model_architecture()
    
    if success:
        logger.info("🎉 Phase 1: Autoencoder Architecture Design - COMPLETED")
        logger.info("✅ Model architecture is ready for Phase 2: Data Preparation")
    else:
        logger.error("❌ Architecture test failed")
    
    # Save configuration
    config = AutoencoderConfig()
    config_path = Path("model_artifacts/architecture_config.json")
    config_path.parent.mkdir(exist_ok=True)
    config.save_config(config_path)
