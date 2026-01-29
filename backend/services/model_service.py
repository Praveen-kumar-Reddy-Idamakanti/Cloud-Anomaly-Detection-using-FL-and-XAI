"""
Model service for handling model loading, inference, and management.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch

# Try to import the actual model, fall back to mock if not available
try:
    from federated_anomaly_detection.models.autoencoder import create_model, AnomalyDetector
    from use_model import load_model, detect_anomalies
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    logging.warning("Federated learning modules not available, using mock implementations")

logger = logging.getLogger(__name__)

# Global model variables
model: Optional[Any] = None
model_info: Dict[str, Any] = {}
device: Optional[torch.device] = None


class ModelService:
    """Service class for model operations."""
    
    def __init__(self):
        self.model = model
        self.model_info = model_info
        self.device = device
    
    def load_latest_model(self) -> Optional[Dict[str, Any]]:
        """
        Load the latest trained model from the logs directory.
        
        Returns:
            Dictionary with model information or None if loading fails
        """
        global model, model_info, device
        
        if not MODEL_AVAILABLE:
            logger.warning("Model loading not available - using mock")
            self.model = "mock_model"
            self.model_info = {"input_dim": 9, "status": "mock"}
            self.device = torch.device("cpu")
            return {
                "model_path": "mock_path",
                "input_dim": 9,
                "last_trained": "mock_date",
                "accuracy": None,
                "status": "mock"
            }
        
        try:
            # Look for the latest model in logs/server directory
            logs_dir = Path("logs/server")
            if not logs_dir.exists():
                raise FileNotFoundError("No logs directory found")
            
            # Find the most recent model
            model_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
            if not model_dirs:
                raise FileNotFoundError("No model directories found")
            
            latest_dir = max(model_dirs, key=lambda x: x.name)
            
            # Look for best_model.pth first, then fall back to the latest round model
            model_path = latest_dir / "best_model.pth"
            
            if not model_path.exists():
                # Look for model files with pattern model_round_X.pth
                model_files = list(latest_dir.glob("model_round_*.pth"))
                if model_files:
                    # Use the highest round number
                    model_path = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
                    logger.info(f"Using model from round: {model_path.name}")
                else:
                    raise FileNotFoundError(f"No model files found in {latest_dir}")
            
            logger.info(f"Loading model from {model_path}")
            model, model_info, device = load_model(str(model_path))
            
            # Update service instance variables
            self.model = model
            self.model_info = model_info
            self.device = device
            
            return {
                "model_path": str(model_path),
                "input_dim": model_info.get("input_dim", 9),
                "last_trained": latest_dir.name,
                "accuracy": model_info.get("accuracy"),
                "status": "loaded"
            }
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        return {
            "model_path": self.model_info.get("model_path", "unknown"),
            "input_dim": self.model_info.get("input_dim", 9),
            "last_trained": self.model_info.get("last_trained", "unknown"),
            "accuracy": self.model_info.get("accuracy"),
            "status": "loaded"
        }
    
    def detect_anomalies(self, features: np.ndarray, threshold: float = 0.4) -> Dict[str, np.ndarray]:
        """
        Detect anomalies in the given features.
        
        Args:
            features: Input features as numpy array
            threshold: Anomaly detection threshold
            
        Returns:
            Dictionary with detection results
        """
        if not MODEL_AVAILABLE:
            # Mock implementation
            return {
                'is_anomaly': np.zeros(len(features), dtype=int),
                'anomaly_scores': np.random.random(len(features)) * 0.3
            }
        
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Ensure correct input dimension
        expected_dim = self.model_info.get("input_dim", 9)
        if features.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, got {features.shape[1]}")
        
        # Detect anomalies
        results = detect_anomalies(self.model, features, threshold, self.device)
        return results
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None
    
    def get_input_dimension(self) -> int:
        """Get the input dimension of the loaded model."""
        return self.model_info.get("input_dim", 9)
    
    def get_device(self) -> torch.device:
        """Get the device the model is running on."""
        return self.device or torch.device("cpu")


# Global model service instance
model_service = ModelService()
