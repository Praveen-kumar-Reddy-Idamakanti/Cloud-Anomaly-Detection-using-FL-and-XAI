import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super(AnomalyDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, encoding_dim),
            nn.LeakyReLU(0.1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def train_epoch(self, dataloader, optimizer, device, scheduler=None, val_dataloader=None, 
                    class_weights=None):
        """
        Train the model for one epoch with optional class weights
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer for training
            device: Device to run training on
            scheduler: Learning rate scheduler (optional)
            val_dataloader: Validation data loader (required if using ReduceLROnPlateau scheduler)
            class_weights: Optional tensor of weights for each class [normal_weight, anomaly_weight]
            
        Returns:
            Average training loss for the epoch
        """
        self.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                data, targets = batch[0], batch[1]  # Unpack if batch is (data, targets)
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
                # Move targets to device if they're not already there
                targets = targets.to(device)
                
                # Initialize weights tensor
                weights = torch.ones_like(reconstruction_errors)
                
                # Apply class weights
                for class_idx, weight in enumerate(class_weights):
                    class_mask = (targets == class_idx)
                    if class_mask.any():
                        weights[class_mask] = weight
                
                # Apply weights to reconstruction errors
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
            
            total_loss += loss.item() * data.size(0)  # Multiply by batch size for correct averaging
        
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
                    batch = batch[0]  # Handle case where batch is (data, target)
                    
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
                    batch = batch[0]  # Handle case where batch is (data, target)
                    
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

def create_model(input_dim: int, learning_rate: float = 1e-3, device: str = "cpu",
                class_weights: torch.Tensor = None) -> Tuple[AnomalyDetector, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, torch.Tensor]:
    """
    Create and return the autoencoder model, optimizer, and learning rate scheduler
    
    Args:
        input_dim: Number of input features
        learning_rate: Initial learning rate for the optimizer
        device: Device to run the model on ('cpu' or 'cuda')
        class_weights: Optional tensor of weights for each class [normal_weight, anomaly_weight]
        
    Returns:
        Tuple of (model, optimizer, scheduler, class_weights)
    """
    model = AnomalyDetector(input_dim).to(device)
    
    # Calculate class weights if not provided
    if class_weights is None:
        # Default weights (will be 1.0 for both classes if not specified)
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
