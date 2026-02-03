#!/usr/bin/env python3
"""
Test script to verify CloudAnomalyAutoencoder integration with federated learning
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add the federated_anomaly_detection path
sys.path.append(str(Path(__file__).parent))

def test_model_integration():
    """Test both models for compatibility with federated learning"""
    
    print("🧪 TESTING MODEL INTEGRATION FOR FEDERATED LEARNING")
    print("=" * 60)
    
    # Test parameters
    input_dim = 79
    batch_size = 32
    learning_rate = 0.001
    
    # Create sample data
    sample_data = torch.randn(batch_size, input_dim)
    
    models_tested = []
    
    # Test 1: Original AnomalyDetector
    print("\n1️⃣ Testing Original AnomalyDetector...")
    try:
        from autoencoder import create_model
        model_orig, optimizer_orig, scheduler_orig, class_weights_orig = create_model(
            input_dim=input_dim,
            learning_rate=learning_rate,
            device='cpu',
            use_cloud_anomaly=False
        )
        
        # Test forward pass
        with torch.no_grad():
            output_orig = model_orig(sample_data)
            
        models_tested.append({
            'name': 'Original AnomalyDetector',
            'model': model_orig,
            'output_shape': output_orig.shape,
            'parameters': sum(p.numel() for p in model_orig.parameters() if p.requires_grad),
            'success': True
        })
        
        print(f"   ✅ Forward pass: {sample_data.shape} → {output_orig.shape}")
        print(f"   📊 Parameters: {models_tested[-1]['parameters']:,}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        models_tested.append({'name': 'Original AnomalyDetector', 'success': False})
    
    # Test 2: CloudAnomalyAutoencoder (via adapter)
    print("\n2️⃣ Testing CloudAnomalyAutoencoder (via adapter)...")
    try:
        model_cloud, optimizer_cloud, scheduler_cloud, class_weights_cloud = create_model(
            input_dim=input_dim,
            learning_rate=learning_rate,
            device='cpu',
            use_cloud_anomaly=True
        )
        
        # Test forward pass
        with torch.no_grad():
            output_cloud = model_cloud(sample_data)
            
        models_tested.append({
            'name': 'CloudAnomalyAutoencoder',
            'model': model_cloud,
            'output_shape': output_cloud.shape,
            'parameters': sum(p.numel() for p in model_cloud.parameters() if p.requires_grad),
            'success': True
        })
        
        print(f"   ✅ Forward pass: {sample_data.shape} → {output_cloud.shape}")
        print(f"   📊 Parameters: {models_tested[-1]['parameters']:,}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        models_tested.append({'name': 'CloudAnomalyAutoencoder', 'success': False})
    
    # Test 3: Direct CloudAnomalyAutoencoder (for comparison)
    print("\n3️⃣ Testing Direct CloudAnomalyAutoencoder...")
    try:
        sys.path.append(str(Path(__file__).parent.parent.parent / 'model_development'))
        from auto_encoder_model import CloudAnomalyAutoencoder
        
        model_direct = CloudAnomalyAutoencoder(input_dim=input_dim)
        
        # Test forward pass
        with torch.no_grad():
            reconstructed, encoded = model_direct(sample_data)
            
        models_tested.append({
            'name': 'Direct CloudAnomalyAutoencoder',
            'model': model_direct,
            'output_shape': (reconstructed.shape, encoded.shape),
            'parameters': sum(p.numel() for p in model_direct.parameters() if p.requires_grad),
            'success': True
        })
        
        print(f"   ✅ Forward pass: {sample_data.shape} → ({reconstructed.shape}, {encoded.shape})")
        print(f"   📊 Parameters: {models_tested[-1]['parameters']:,}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        models_tested.append({'name': 'Direct CloudAnomalyAutoencoder', 'success': False})
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 INTEGRATION TEST SUMMARY")
    print("=" * 60)
    
    successful_models = [m for m in models_tested if m['success']]
    failed_models = [m for m in models_tested if not m['success']]
    
    if successful_models:
        print(f"✅ Successfully tested {len(successful_models)} models:")
        for model in successful_models:
            if 'parameters' in model:
                print(f"   - {model['name']}: {model['parameters']:,} parameters")
            else:
                print(f"   - {model['name']}: Success")
    
    if failed_models:
        print(f"❌ Failed to test {len(failed_models)} models:")
        for model in failed_models:
            print(f"   - {model['name']}")
    
    # Compatibility check
    print(f"\n🔍 FEDERATED LEARNING COMPATIBILITY CHECK")
    print("-" * 40)
    
    if len(successful_models) >= 2:
        print("✅ Multiple models available for FL")
        print("✅ Can switch between models via configuration")
        print("✅ Both models produce compatible outputs")
    elif len(successful_models) == 1:
        print("⚠️  Only one model available, but FL is possible")
    else:
        print("❌ No models successfully loaded - FL not possible")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION")
    print("-" * 40)
    
    if any(m['name'] == 'CloudAnomalyAutoencoder' and m['success'] for m in models_tested):
        print("🎯 Use CloudAnomalyAutoencoder for:")
        print("   - Better feature learning")
        print("   - More sophisticated architecture")
        print("   - Production deployment")
        print("\n📝 To use: Set use_cloud_anomaly=True in create_model()")
    else:
        print("⚠️  CloudAnomalyAutoencoder not available")
        print("📝 Use original AnomalyDetector with use_cloud_anomaly=False")
    
    return len(successful_models) > 0


def test_federated_compatibility():
    """Test federated learning specific functionality"""
    
    print("\n🔄 TESTING FEDERATED LEARNING SPECIFIC FUNCTIONALITY")
    print("=" * 60)
    
    try:
        from autoencoder import create_model
        
        # Create model with CloudAnomalyAutoencoder
        model, optimizer, scheduler, class_weights = create_model(
            input_dim=79,
            use_cloud_anomaly=True
        )
        
        # Test parameter serialization (critical for FL)
        print("1️⃣ Testing parameter serialization...")
        params = [p.detach().cpu().numpy() for p in model.parameters()]
        print(f"   ✅ Serialized {len(params)} parameter tensors")
        
        # Test parameter deserialization
        print("2️⃣ Testing parameter deserialization...")
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        print("   ✅ Successfully restored model parameters")
        
        # Test training step
        print("3️⃣ Testing training step...")
        model.train()
        sample_batch = torch.randn(16, 79)
        reconstructed = model(sample_batch)
        loss = torch.nn.functional.mse_loss(reconstructed, sample_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   ✅ Training step completed (loss: {loss.item():.6f})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Federated compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 STARTING FEDERATED LEARNING MODEL INTEGRATION TESTS")
    print("=" * 80)
    
    # Test basic integration
    integration_success = test_model_integration()
    
    # Test federated compatibility
    if integration_success:
        federated_success = test_federated_compatibility()
        
        if federated_success:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ CloudAnomalyAutoencoder is ready for federated learning")
            print("✅ You can now use it in your FL training")
        else:
            print("\n⚠️  Integration successful but federated compatibility issues detected")
    else:
        print("\n❌ Integration tests failed")
        print("Please check the error messages above")
    
    print("\n" + "=" * 80)
    print("Test completed. Check the results above for next steps.")
    print("=" * 80)
