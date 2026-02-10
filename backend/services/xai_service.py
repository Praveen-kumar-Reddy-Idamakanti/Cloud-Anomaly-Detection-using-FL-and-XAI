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

# Network Traffic Feature Names Mapping
NETWORK_FEATURE_NAMES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
    "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Fwd Segment Size Avg",
    "Bwd Segment Size Avg", "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init Fwd Win Bytes",
    "Init Bwd Win Bytes", "Fwd Act Data Packets", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean",
    "Idle Std", "Idle Max", "Idle Min"
]


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
            # Mock implementation with meaningful feature names
            feature_importances = []
            for i in range(len(features)):
                feature_name = NETWORK_FEATURE_NAMES[i] if i < len(NETWORK_FEATURE_NAMES) else f"feature_{i}"
                feature_importances.append({
                    "feature": feature_name,
                    "importance": 0.1,
                    "feature_index": i
                })
            
            return {
                "model_type": "Autoencoder",
                "explanation_type": "SHAP (Mock)",
                "feature_importances": feature_importances,
                "note": "SHAP not available - showing mock explanation with real feature names"
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
            
            # Map SHAP values to features with meaningful names
            feature_importances = []
            for i, value in enumerate(shap_values[0]):  # Assuming shap_values[0] for single output
                # Use actual feature name if available, otherwise fallback to feature index
                feature_name = NETWORK_FEATURE_NAMES[i] if i < len(NETWORK_FEATURE_NAMES) else f"feature_{i}"
                feature_importances.append({
                    "feature": feature_name, 
                    "importance": float(value),
                    "feature_index": i  # Include index for reference
                })
            
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
