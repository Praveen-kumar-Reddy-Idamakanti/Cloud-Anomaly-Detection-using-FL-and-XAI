"""
Configuration for using different autoencoder models in federated learning
"""

class FederatedModelConfig:
    """Configuration for federated learning model selection"""
    
    # Model selection
    USE_CLOUD_ANOMALY_AUTOENCODER = True  # Set to True to use CloudAnomalyAutoencoder, False for original
    
    # Model parameters
    DEFAULT_INPUT_DIM = 79  # Default for cloud anomaly data
    DEFAULT_ENCODING_DIM = 32
    
    # Training parameters
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_EPOCHS = 5
    
    # Feature comparison
    MODEL_FEATURES = {
        'original_anomaly_detector': {
            'architecture': 'input → 128 → 64 → 32 → 64 → 128 → input',
            'activation': 'LeakyReLU',
            'normalization': 'BatchNorm1d',
            'dropout': 0.3,
            'weight_init': 'Kaiming Normal',
            'parameters_est': '~15,000-20,000',
            'strengths': ['Proven in FL', 'Lightweight', 'Fast training'],
            'use_cases': ['Quick prototyping', 'Resource-constrained environments']
        },
        'cloud_anomaly_autoencoder': {
            'architecture': 'input → 64 → 32 → 16 → 8 → 4 → 8 → 16 → 32 → 64 → input',
            'activation': 'ReLU',
            'normalization': 'None',
            'dropout': 0.1 (configurable),
            'weight_init': 'Xavier Uniform',
            'parameters_est': '~12,000-15,000',
            'strengths': ['Deeper architecture', 'Better compression', 'More sophisticated'],
            'use_cases': ['Production deployment', 'Better feature learning', 'Complex patterns']
        }
    }
    
    @classmethod
    def get_model_info(cls):
        """Get information about the currently selected model"""
        model_name = 'cloud_anomaly_autoencoder' if cls.USE_CLOUD_ANOMALY_AUTOENCODER else 'original_anomaly_detector'
        return {
            'selected_model': model_name,
            'features': cls.MODEL_FEATURES[model_name]
        }
    
    @classmethod
    def print_model_comparison(cls):
        """Print comparison between models"""
        print("=" * 80)
        print("FEDERATED LEARNING MODEL COMPARISON")
        print("=" * 80)
        
        for model_name, features in cls.MODEL_FEATURES.items():
            status = "✅ SELECTED" if (
                (model_name == 'cloud_anomaly_autoencoder' and cls.USE_CLOUD_ANOMALY_AUTOENCODER) or
                (model_name == 'original_anomaly_detector' and not cls.USE_CLOUD_ANOMALY_AUTOENCODER)
            ) else "❌"
            
            print(f"\n{status} {model_name.upper().replace('_', ' ')}")
            print("-" * 60)
            for key, value in features.items():
                if isinstance(value, list):
                    print(f"  {key.replace('_', ' ').title()}: {', '.join(value)}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\n" + "=" * 80)
        print(f"CURRENT SELECTION: {cls.get_model_info()['selected_model'].upper().replace('_', ' ')}")
        print("To switch models, modify USE_CLOUD_ANOMALY_AUTOENCODER in this file")
        print("=" * 80)


# Usage instructions
"""
HOW TO SWITCH BETWEEN MODELS:

1. USING CloudAnomalyAutoencoder (from model_development):
   - Set USE_CLOUD_ANOMALY_AUTOENCODER = True
   - Benefits: Deeper architecture, better compression, more sophisticated
   - Best for: Production deployment, complex patterns

2. USING Original AnomalyDetector:
   - Set USE_CLOUD_ANOMALY_AUTOENCODER = False  
   - Benefits: Proven in FL, lightweight, fast training
   - Best for: Quick prototyping, resource-constrained environments

3. IN CLIENT CODE:
   The model will be automatically selected based on the configuration.
   No changes needed in client/server code.

4. COMMAND LINE USAGE:
   You can also override via command line when starting clients:
   python client.py --node_id 1 --use-cloud-anomaly  # Use CloudAnomalyAutoencoder
   python client.py --node_id 1 --use-original       # Use original model
"""
