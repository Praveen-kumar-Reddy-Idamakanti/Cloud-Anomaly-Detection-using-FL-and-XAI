"""
XAI (Explainable AI) service for generating model explanations.
"""

import logging
import traceback
from typing import List, Dict, Any, Optional
import numpy as np
import torch

# Try to import SHAP, fall back to mock if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError as e:
    SHAP_AVAILABLE = False
    logging.warning(f"SHAP not available: {e}. Using mock implementations")

from services.model_service import model_service

logger = logging.getLogger(__name__)


class XAIService:
    """Service class for XAI operations."""
    
    def __init__(self):
        self.model_service = model_service
    
    def shap_prediction_function(self, data: np.ndarray) -> np.ndarray:
        """
        Prediction function for SHAP: takes a numpy array and returns anomaly scores.
        
        Args:
            data: Input data as numpy array
            
        Returns:
            Anomaly scores as numpy array
        """
        model = self.model_service.model
        device = self.model_service.device
        
        if model is None or device is None:
            raise RuntimeError("Model not loaded for SHAP explanation.")
        
        # Ensure data is float32
        data = data.astype(np.float32)
        
        # Convert numpy array to torch tensor
        input_tensor = torch.from_numpy(data).to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        with torch.no_grad():
            reconstructed = model(input_tensor)
            # Anomaly score is typically the reconstruction error (e.g., MSE)
            anomaly_scores = torch.mean(torch.pow(input_tensor - reconstructed, 2), dim=1)
            
        return anomaly_scores.cpu().numpy()
    
    def generate_shap_explanation(self, features: List[float]) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a given anomalous data point.
        
        Args:
            features: List of features to explain
            
        Returns:
            Dictionary containing SHAP explanation
        """
        if not SHAP_AVAILABLE:
            # Mock implementation
            return {
                "model_type": "Autoencoder",
                "explanation_type": "SHAP (Mock)",
                "feature_importances": [
                    {"feature": f"feature_{i}", "importance": 0.1} for i in range(len(features))
                ],
                "note": "SHAP not available - showing mock explanation"
            }
        
        try:
            # Get model information
            input_dim = self.model_service.get_input_dimension()
            
            if not self.model_service.is_model_loaded():
                raise ValueError("Model not loaded. Cannot generate explanation.")
            
            # Create a background dataset for SHAP
            # For simplicity, using a small random background, but ideally,
            # this should be a representative sample of non-anomalous data.
            background_data = np.random.rand(100, input_dim).astype(np.float32)
            
            # Create a KernelExplainer
            explainer = shap.KernelExplainer(self.shap_prediction_function, background_data)
            
            # Convert the single instance to explain to a numpy array
            instance_to_explain = np.array([features], dtype=np.float32)
            
            # Generate SHAP values
            shap_values = explainer.shap_values(instance_to_explain)
            
            # Map SHAP values to features
            feature_importances = []
            for i, value in enumerate(shap_values[0]):  # Assuming shap_values[0] for single output
                feature_importances.append({"feature": f"feature_{i}", "importance": float(value)})
            
            # Sort by absolute importance for better visualization
            feature_importances.sort(key=lambda x: abs(x["importance"]), reverse=True)
            
            return {
                "model_type": "Autoencoder",
                "explanation_type": "SHAP",
                "feature_importances": feature_importances,
                "note": "SHAP values indicate the contribution of each feature to the anomaly score (reconstruction error)."
            }
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}")
            traceback.print_exc()
            raise ValueError(f"Failed to generate SHAP explanation: {e}")
    
    def get_mock_explanation(self, anomaly_id: str) -> Dict[str, Any]:
        """
        Generate a mock explanation for testing purposes.
        
        Args:
            anomaly_id: ID of the anomaly to explain
            
        Returns:
            Mock explanation data
        """
        return {
            "id": f"explanation-{anomaly_id}",
            "anomaly_id": anomaly_id,
            "model_type": "Autoencoder",
            "data_type": "mock_data",
            "status": "work_in_progress",
            "note": "This is mock data used for testing purposes. Real federated learning explanations will be available once the system is fully integrated.",
            "shap": [
                {"feature": "packet_size", "importance": 0.45},
                {"feature": "protocol", "importance": 0.32},
                {"feature": "time_of_day", "importance": 0.28},
                {"feature": "source_ip_reputation", "importance": 0.15},
                {"feature": "connection_frequency", "importance": 0.12},
            ],
            "lime": [
                {"feature": "packet_size", "importance": 0.38},
                {"feature": "protocol", "importance": 0.35},
                {"feature": "time_of_day", "importance": 0.25},
                {"feature": "source_ip_reputation", "importance": 0.18},
                {"feature": "connection_frequency", "importance": 0.14},
            ],
            "contributing_factors": [
                "Unusual time of access",
                "Connection from untrusted IP range",
                "Abnormal data transfer volume",
                "Suspicious protocol usage"
            ],
            "recommendations": [
                "Monitor source IP for additional suspicious activity",
                "Verify legitimacy of data transfers",
                "Apply additional authentication for this source"
            ]
        }


# Global XAI service instance
xai_service = XAIService()
