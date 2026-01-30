"""
Model service for handling model loading, inference, and management.
Enhanced to support two-stage prediction with attack type classification.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch

# Import path configuration
from config.app_config import path_config

# Try to import the actual model, fall back to mock if not available
try:
    from federated_anomaly_detection.models.autoencoder import create_model, AnomalyDetector
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    logging.warning("Federated learning modules not available, using mock implementations")

# Try to import the enhanced two-stage model
try:
    sys.path.append(str(path_config.model_development_path))
    from train import FixedAutoencoderTrainer, AttackTypeClassifier
    TWO_STAGE_AVAILABLE = True
except ImportError:
    TWO_STAGE_AVAILABLE = False
    logging.warning("Two-stage model not available, using standard model")

logger = logging.getLogger(__name__)

# Global model variables
model: Optional[Any] = None
attack_classifier: Optional[Any] = None
model_info: Dict[str, Any] = {}
device: Optional[torch.device] = None
attack_types = path_config.get_attack_types()


class ModelService:
    """Service class for model operations."""
    
    def __init__(self):
        self.model = model
        self.attack_classifier = attack_classifier
        self.model_info = model_info
        self.device = device
        self.two_stage_enabled = TWO_STAGE_AVAILABLE and self.attack_classifier is not None
    
    def load_latest_model(self) -> Optional[Dict[str, Any]]:
        """
        Load the latest trained model from the logs directory.
        Enhanced to support two-stage model loading.
        
        Returns:
            Dictionary with model information or None if loading fails
        """
        global model, model_info, device, attack_classifier
        
        if not MODEL_AVAILABLE:
            logger.warning("Model loading not available - using mock")
            self.model = "mock_model"
            self.model_info = {"input_dim": 78, "status": "mock"}
            self.device = torch.device("cpu")
            return {
                "model_path": "mock_path",
                "input_dim": 78,
                "last_trained": "mock_date",
                "accuracy": None,
                "status": "mock",
                "two_stage_enabled": False,
                "attack_types": []
            }
        
        try:
            # Try to load two-stage model first
            if TWO_STAGE_AVAILABLE:
                return self._load_two_stage_model()
            else:
                return self._load_standard_model()
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
    
    def _load_two_stage_model(self) -> Optional[Dict[str, Any]]:
        """Load the enhanced two-stage model."""
        global model, model_info, device, attack_classifier
        
        # Look for the latest model in model_artifacts directory
        model_dir = path_config.model_artifacts_path
        if not model_dir.exists():
            logger.warning("No model_artifacts directory found, falling back to logs")
            return self._load_standard_model()
        
        # Look for best_autoencoder_fixed.pth
        model_path = model_dir / "best_autoencoder_fixed.pth"
        
        if not model_path.exists():
            logger.warning("No enhanced model found, falling back to standard model")
            return self._load_standard_model()
        
        try:
            logger.info(f"Loading two-stage model from {model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Initialize models
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load autoencoder
            from model_development.autoencoder_model import CloudAnomalyAutoencoder, AutoencoderConfig
            config = AutoencoderConfig()
            model = CloudAnomalyAutoencoder(
                input_dim=path_config.get_model_input_dim(),
                encoding_dims=config.encoding_dims,
                bottleneck_dim=config.bottleneck_dim,
                dropout_rate=config.dropout_rate
            ).to(device)
            
            # Load attack classifier
            attack_classifier = AttackTypeClassifier(input_dim=path_config.get_model_input_dim(), num_classes=5).to(device)
            
            # Load state dicts
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'attack_classifier_state_dict' in checkpoint:
                attack_classifier.load_state_dict(checkpoint['attack_classifier_state_dict'])
                self.two_stage_enabled = True
            
            # Update service variables
            self.model = model
            self.attack_classifier = attack_classifier
            self.device = device
            # Use actual model input dimension from config
            self.model_info = {
                "input_dim": path_config.get_model_input_dim(),
                "model_path": str(model_path),
                "status": "loaded"
            }
            
            return {
                "model_path": str(model_path),
                "input_dim": path_config.get_model_input_dim(),
                "last_trained": checkpoint.get('epoch', 'unknown'),
                "accuracy": checkpoint.get('accuracy'),
                "status": "loaded",
                "two_stage_enabled": self.two_stage_enabled,
                "attack_types": attack_types
            }
            
        except Exception as e:
            logger.error(f"Failed to load two-stage model: {e}")
            return self._load_standard_model()
    
    def _load_standard_model(self) -> Optional[Dict[str, Any]]:
        """Load the standard autoencoder model."""
        global model, model_info, device
        
        # Look for the latest model in logs/server directory
        logs_dir = path_config.logs_path / "server"
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
        
        logger.info(f"Loading standard model from {model_path}")
        
        # Use standard loading logic
        try:
            from use_model import load_model, detect_anomalies
            model, model_info, device = load_model(str(model_path))
            
            # Update service instance variables
            self.model = model
            self.model_info = model_info
            self.device = device
            
            return {
                "model_path": str(model_path),
                "input_dim": path_config.get_model_input_dim(),
                "last_trained": latest_dir.name,
                "accuracy": model_info.get("accuracy"),
                "status": "loaded",
                "two_stage_enabled": False,
                "attack_types": []
            }
        except ImportError:
            logger.warning("use_model not available, creating mock model")
            self.model = "mock_model"
            self.model_info = {"input_dim": 78, "status": "mock"}
            self.device = torch.device("cpu")
            return {
                "model_path": "mock_path",
                "input_dim": 78,
                "last_trained": "mock_date",
                "accuracy": None,
                "status": "mock",
                "two_stage_enabled": False,
                "attack_types": []
            }
    
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
            "input_dim": path_config.get_model_input_dim(),
            "last_trained": self.model_info.get("last_trained", "unknown"),
            "accuracy": self.model_info.get("accuracy"),
            "status": "loaded",
            "two_stage_enabled": self.two_stage_enabled,
            "attack_types": attack_types if self.two_stage_enabled else []
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
        expected_dim = path_config.get_model_input_dim()
        if features.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, got {features.shape[1]}")
        
        # Detect anomalies
        try:
            from use_model import detect_anomalies
            results = detect_anomalies(self.model, features, threshold, self.device)
            return results
        except ImportError:
            # Fallback mock implementation
            return {
                'is_anomaly': np.random.choice([0, 1], len(features), p=[0.9, 0.1]),
                'anomaly_scores': np.random.random(len(features)) * 0.5
            }
    
    def detect_anomalies_two_stage(self, features: np.ndarray, threshold: float = None) -> Dict[str, Any]:
        """
        Detect anomalies using two-stage prediction (autoencoder + attack classifier).
        
        Args:
            features: Input features as numpy array (78 features)
            threshold: Anomaly detection threshold (from config if not provided)
            
        Returns:
            Dictionary with two-stage detection results
        """
        if threshold is None:
            threshold = path_config.get_anomaly_threshold()
        
        if not self.two_stage_enabled:
            # Fallback to standard detection
            logger.warning("Two-stage model not available, using standard detection")
            standard_results = self.detect_anomalies(features, threshold)
            return {
                'anomaly_predictions': standard_results['is_anomaly'].tolist(),
                'reconstruction_errors': standard_results['anomaly_scores'].tolist(),
                'attack_type_predictions': [0] * len(features),  # All BENIGN
                'attack_confidences': [1.0] * len(features),
                'threshold': threshold,
                'attack_types': attack_types
            }
        
        if self.model is None or self.attack_classifier is None:
            raise ValueError("Two-stage model not properly loaded")
        
        try:
            # Ensure correct input dimension
            expected_dim = path_config.get_model_input_dim()
            if features.shape[1] != expected_dim:
                raise ValueError(f"Expected {expected_dim} features, got {features.shape[1]}")
            
            # Convert to tensor
            features_tensor = torch.from_numpy(features).float().to(self.device)
            
            # Stage 1: Anomaly detection with autoencoder
            self.model.eval()
            with torch.no_grad():
                reconstructed, encoded = self.model(features_tensor)
                reconstruction_errors = torch.mean((reconstructed - features_tensor) ** 2, dim=1)
                
                # Binary classification
                anomaly_predictions = (reconstruction_errors > threshold).float()
            
            # Stage 2: Attack type classification for anomalies
            attack_predictions = []
            attack_confidences = []
            
            self.attack_classifier.eval()
            with torch.no_grad():
                for i, is_anomaly in enumerate(anomaly_predictions):
                    if is_anomaly.item() == 1:  # If anomaly detected
                        attack_output = self.attack_classifier(features_tensor[i:i+1])
                        attack_probs = torch.softmax(attack_output, dim=1)
                        attack_pred = torch.argmax(attack_probs, dim=1).item()
                        attack_conf = torch.max(attack_probs).item()
                        
                        attack_predictions.append(attack_pred)
                        attack_confidences.append(attack_conf)
                    else:
                        attack_predictions.append(0)  # BENIGN
                        attack_confidences.append(1.0)  # High confidence for normal
            
            return {
                'anomaly_predictions': anomaly_predictions.cpu().numpy().astype(int).tolist(),
                'reconstruction_errors': reconstruction_errors.cpu().numpy().tolist(),
                'attack_type_predictions': attack_predictions,
                'attack_confidences': attack_confidences,
                'threshold': threshold,
                'attack_types': attack_types
            }
            
        except Exception as e:
            logger.error(f"Two-stage prediction failed: {e}")
            # Fallback to standard detection
            standard_results = self.detect_anomalies(features, threshold)
            return {
                'anomaly_predictions': standard_results['is_anomaly'].tolist(),
                'reconstruction_errors': standard_results['anomaly_scores'].tolist(),
                'attack_type_predictions': [0] * len(features),  # All BENIGN
                'attack_confidences': [1.0] * len(features),
                'threshold': threshold,
                'attack_types': attack_types
            }
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None
    
    def get_input_dimension(self) -> int:
        """Get the input dimension of the loaded model."""
        return path_config.get_model_input_dim()
    
    def get_device(self) -> torch.device:
        """Get the device the model is running on."""
        return self.device or torch.device("cpu")


# Global model service instance
model_service = ModelService()
