"""
Startup script to initialize the model service
"""

import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from services.model_service import model_service

def initialize_model():
    """Initialize the model on startup"""
    logger = logging.getLogger(__name__)
    
    print("🚀 Initializing model service...")
    
    try:
        result = model_service.load_latest_model()
        if result:
            print(f"✅ Model loaded successfully!")
            print(f"   Model path: {result.get('model_path')}")
            print(f"   Input dimensions: {result.get('input_dim')}")
            print(f"   Two-stage enabled: {result.get('two_stage_enabled')}")
            print(f"   Attack types: {result.get('attack_types', [])}")
            return True
        else:
            print("❌ Failed to load model")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

if __name__ == "__main__":
    success = initialize_model()
    if success:
        print("🎉 Model service ready!")
    else:
        print("⚠️  Model service initialization failed")
