import torch
from pathlib import Path

# Check what's in the checkpoint
checkpoint_path = Path("model_artifacts/best_autoencoder_fixed.pth")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("🔍 Checkpoint Contents:")
print(f"Config input_dim: {checkpoint['config']['input_dim']}")
print(f"Config keys: {list(checkpoint['config'].keys())}")
print(f"Model state dict keys: {list(checkpoint['model_state_dict'].keys())[:5]}...")

# Check encoder first layer shape
encoder_first_weight = checkpoint['model_state_dict']['encoder.0.weight']
print(f"Encoder first layer shape: {encoder_first_weight.shape}")

# Check decoder last layer shape  
decoder_last_weight = checkpoint['model_state_dict']['decoder.12.weight']
print(f"Decoder last layer shape: {decoder_last_weight.shape}")
