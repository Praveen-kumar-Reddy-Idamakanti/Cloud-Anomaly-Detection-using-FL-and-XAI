"""
Integration adapter for using CloudAnomalyAutoencoder in federated learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import sys
import os

# Add the model_development path to import CloudAnomalyAutoencoder
model_dev_path = os.path.join(os.path.dirname(__file__), '..', '..', 'model_development')
if model_dev_path not in sys.path:
    sys.path.append(model_dev_path)

try:
    from auto_encoder_model import CloudAnomalyAutoencoder, AutoencoderConfig
except ImportError as e:
    print(f"Warning: Could not import CloudAnomalyAutoencoder: {e}")
    CloudAnomalyAutoencoder = None
    AutoencoderConfig = None


if CloudAnomalyAutoencoder is None:
    # Fallback class when CloudAnomalyAutoencoder is not available
    class CloudAnomalyAutoencoder(nn.Module):
        def __init__(self, input_dim=79, encoding_dims=[64, 32, 16, 8], bottleneck_dim=4, dropout_rate=0.1):
            super(CloudAnomalyAutoencoder, self).__init__()
            # Simple fallback architecture
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, bottleneck_dim),
                nn.ReLU()
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, input_dim),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            reconstructed = self.decoder(encoded)
            return reconstructed, encoded


class FederatedCloudAnomalyAutoencoder(CloudAnomalyAutoencoder):
    """
    Adapter class to make CloudAnomalyAutoencoder compatible with federated learning
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, **kwargs):
        """
        Initialize with federated learning compatible parameters
        
        Args:
            input_dim: Number of input features
            encoding_dim: Encoding dimension (maps to bottleneck_dim)
            **kwargs: Additional parameters for CloudAnomalyAutoencoder
        """
        # Map FL parameters to CloudAnomalyAutoencoder parameters
        super().__init__(
            input_dim=input_dim,
            encoding_dims=[64, 32, 16, encoding_dim],  # Dynamic encoding dims
            bottleneck_dim=encoding_dim,
            dropout_rate=kwargs.get('dropout_rate', 0.3)  # Match FL dropout
        )
        
        self.encoding_dim = encoding_dim
        
    def forward(self, x):
        """
        Forward pass compatible with federated learning expectations
        Returns only reconstructed output (FL compatibility)
        """
        reconstructed, encoded = super().forward(x)
        return reconstructed
    
    def forward_with_encoding(self, x):
        """
        Forward pass that returns both reconstructed and encoded
        For use when encoded representation is needed
        """
        return super().forward(x)
    
    def train_epoch(self, dataloader, optimizer, device, scheduler=None, val_dataloader=None, 
                    class_weights=None):
        """
        Training method compatible with federated learning client
        """
        self.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, targets = batch[0], batch[1]
            else:
                data = batch
                targets = None
            
            data = data.to(device)
            
            # Forward pass
            reconstructed = self(data)
            
            # Calculate element-wise reconstruction error
            reconstruction_errors = F.mse_loss(reconstructed, data, reduction='none').mean(dim=1)
            
            # Apply class weights if provided and targets are available
            if class_weights is not None and targets is not None:
                targets = targets.to(device)
                weights = torch.ones_like(reconstruction_errors)
                
                for class_idx, weight in enumerate(class_weights):
                    class_mask = (targets == class_idx)
                    if class_mask.any():
                        weights[class_mask] = weight
                
                weighted_errors = reconstruction_errors * weights
                loss = weighted_errors.mean()
            else:
                loss = reconstruction_errors.mean()
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
        
        # Calculate average loss for the epoch
        avg_train_loss = total_loss / len(dataloader.dataset)
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_dataloader is not None:
                    val_loss = self.evaluate(val_dataloader, device)
                    scheduler.step(val_loss)
                else:
                    print("Warning: ReduceLROnPlateau scheduler requires a validation dataloader")
            else:
                scheduler.step()
        
        return avg_train_loss
    
    def evaluate(self, dataloader, device):
        """Evaluate the model on the given dataset"""
        self.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                    
                batch = batch.to(device)
                reconstructed = self(batch)
                loss = F.mse_loss(reconstructed, batch, reduction='sum')
                total_loss += loss.item()
                
        return total_loss / len(dataloader.dataset)
    
    def detect_anomaly(self, dataloader, device, threshold: float = None):
        """Detect anomalies based on reconstruction error"""
        self.eval()
        all_errors = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                    
                batch = batch.to(device)
                reconstructed = self(batch)
                mse = F.mse_loss(reconstructed, batch, reduction='none').mean(dim=1)
                all_errors.append(mse.cpu().numpy())
        
        errors = np.concatenate(all_errors)
        
        if threshold is None:
            # Use 95th percentile as threshold if not provided
            threshold = np.percentile(errors, 95)
            
        anomalies = errors > threshold
        return anomalies, errors, threshold


def create_federated_model(input_dim: int, learning_rate: float = 1e-3, device: str = "cpu",
                         class_weights: torch.Tensor = None, use_cloud_anomaly: bool = True) -> Tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, torch.Tensor]:
    """
    Create and return the autoencoder model for federated learning
    
    Args:
        input_dim: Number of input features
        learning_rate: Initial learning rate for the optimizer
        device: Device to run the model on ('cpu' or 'cuda')
        class_weights: Optional tensor of weights for each class [normal_weight, anomaly_weight]
        use_cloud_anomaly: Whether to use CloudAnomalyAutoencoder (True) or original AnomalyDetector (False)
        
    Returns:
        Tuple of (model, optimizer, scheduler, class_weights)
    """
    if use_cloud_anomaly:
        # Use the enhanced CloudAnomalyAutoencoder
        model = FederatedCloudAnomalyAutoencoder(input_dim=input_dim, encoding_dim=32)
    else:
        # Fallback to original model
        from federated_anomaly_detection.models.autoencoder import AnomalyDetector
        model = AnomalyDetector(input_dim=input_dim, encoding_dim=32)
    
    model = model.to(device)
    
    # Calculate class weights if not provided
    if class_weights is None:
        class_weights = torch.ones(2, device=device)
    elif not isinstance(class_weights, torch.Tensor):
        class_weights = torch.tensor(class_weights, device=device)
    
    # Use AdamW with weight decay for better regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
        amsgrad=True
    )
    
    # Add learning rate scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
    
    return model, optimizer, scheduler, class_weights


if __name__ == "__main__":
    # Test the integration
    print("Testing FederatedCloudAnomalyAutoencoder...")
    
    # Create model
    model, optimizer, scheduler, class_weights = create_federated_model(
        input_dim=79, 
        use_cloud_anomaly=True
    )
    
    print(f"✅ Model created successfully")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   Device: {next(model.parameters()).device}")
    
    # Test forward pass
    batch_size = 32
    sample_input = torch.randn(batch_size, 79)
    
    with torch.no_grad():
        output = model(sample_input)
        print(f"✅ Forward pass successful")
        print(f"   Input shape: {sample_input.shape}")
        print(f"   Output shape: {output.shape}")
    
    print("🎉 Integration test completed successfully!")
