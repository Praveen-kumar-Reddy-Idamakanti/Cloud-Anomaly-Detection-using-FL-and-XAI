#!/usr/bin/env python3
"""
Shared model architecture to ensure consistency between server and clients
"""
import torch
import torch.nn as nn

class SharedFederatedAutoencoder(nn.Module):
    """Consistent autoencoder architecture for both server and clients"""
    def __init__(self, input_dim=79, encoding_dim=64):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder - EXACT reverse of encoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_shared_model(input_dim=79):
    """Create model with consistent architecture"""
    return SharedFederatedAutoencoder(input_dim=input_dim, encoding_dim=64)

def get_model_parameter_count(model):
    """Get total parameter count"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Test model creation
    model = create_shared_model(79)
    print(f"✅ Shared model created successfully")
    print(f"📊 Total parameters: {get_model_parameter_count(model):,}")
    
    # Test forward pass
    test_input = torch.randn(10, 79)
    output = model(test_input)
    print(f"✅ Forward pass successful: {test_input.shape} → {output.shape}")
    
    # Print model architecture
    print("\n📋 Model Architecture:")
    print(model)
