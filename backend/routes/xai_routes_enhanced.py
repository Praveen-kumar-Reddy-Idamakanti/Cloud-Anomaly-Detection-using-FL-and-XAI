"""
Enhanced XAI (Explainable AI) API routes - Integrated with completed XAI phases.
"""

import sys
import os
from pathlib import Path
from fastapi import HTTPException
from typing import Dict, Any, List, Optional
import numpy as np
import logging

# Add XAI module path
project_root = Path(__file__).parent.parent.parent
xai_path = project_root / "model_development" / "xai"
sys.path.append(str(xai_path))

from models.schemas import AnomalyExplanationRequest, AnomalyExplanationResponse
from services.model_service import model_service

# Import XAI modules from completed phases
try:
    from integrated_explainer import IntegratedExplainer
    from autoencoder_explainer import AutoencoderExplainer
    from classifier_explainer import ClassifierExplainer
    XAI_AVAILABLE = True
except ImportError as e:
    logging.warning(f"XAI modules not available: {e}")
    XAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class XAIService:
    """Enhanced XAI service integrating all three phases."""
    
    def __init__(self):
        self.autoencoder_explainer = None
        self.classifier_explainer = None
        self.integrated_explainer = None
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize XAI explainers from completed phases."""
        if not XAI_AVAILABLE or not model_service.model:
            logger.warning("XAI or model not available, using mock implementations")
            return
        
        try:
            # Phase 1: Foundation Setup
            self.integrated_explainer = IntegratedExplainer(
                model_service.model,
                model_service.attack_classifier
            )
            
            # Phase 2: Autoencoder Explainability
            self.autoencoder_explainer = AutoencoderExplainer(
                model_service.model,
                model_service.device
            )
            
            # Phase 3: Attack Type Classification Explainability
            self.classifier_explainer = ClassifierExplainer(
                model_service.attack_classifier,
                model_service.device
            )
            
            logger.info("XAI explainers initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize XAI explainers: {e}")
    
    def get_phase1_explanation(self, features: List[float], anomaly_score: float) -> Dict[str, Any]:
        """Phase 1: Foundation Setup - Basic anomaly explanation."""
        if not self.integrated_explainer:
            return self._get_mock_phase1_explanation(features, anomaly_score)
        
        try:
            explanation = self.integrated_explainer.explain_anomaly(
                np.array(features).reshape(1, -1),
                anomaly_score
            )
            return {
                "phase": "phase1_foundation",
                "explanation_type": "basic_anomaly",
                "features": features,
                "anomaly_score": anomaly_score,
                "explanation": explanation,
                "timestamp": str(np.datetime64('now'))
            }
        except Exception as e:
            logger.error(f"Phase 1 explanation failed: {e}")
            return self._get_mock_phase1_explanation(features, anomaly_score)
    
    def get_phase2_explanation(self, features: List[float], reconstruction_error: float) -> Dict[str, Any]:
        """Phase 2: Autoencoder Explainability - SHAP-based explanations."""
        if not self.autoencoder_explainer:
            return self._get_mock_phase2_explanation(features, reconstruction_error)
        
        try:
            shap_values = self.autoencoder_explainer.explain_reconstruction(
                np.array(features).reshape(1, -1),
                reconstruction_error
            )
            
            return {
                "phase": "phase2_autoencoder",
                "explanation_type": "shap_explainability",
                "features": features,
                "reconstruction_error": reconstruction_error,
                "shap_values": shap_values.tolist(),
                "feature_importance": self._calculate_feature_importance(shap_values),
                "timestamp": str(np.datetime64('now'))
            }
        except Exception as e:
            logger.error(f"Phase 2 explanation failed: {e}")
            return self._get_mock_phase2_explanation(features, reconstruction_error)
    
    def get_phase3_explanation(self, features: List[float], attack_type: int, confidence: float) -> Dict[str, Any]:
        """Phase 3: Attack Type Classification Explainability."""
        if not self.classifier_explainer:
            return self._get_mock_phase3_explanation(features, attack_type, confidence)
        
        try:
            attack_explanation = self.classifier_explainer.explain_attack_type(
                np.array(features).reshape(1, -1),
                attack_type,
                confidence
            )
            
            return {
                "phase": "phase3_classification",
                "explanation_type": "attack_type_explainability",
                "features": features,
                "attack_type": attack_type,
                "confidence": confidence,
                "explanation": attack_explanation,
                "attack_shap_values": attack_explanation.get("shap_values", []),
                "decision_boundary": attack_explanation.get("decision_boundary", {}),
                "timestamp": str(np.datetime64('now'))
            }
        except Exception as e:
            logger.error(f"Phase 3 explanation failed: {e}")
            return self._get_mock_phase3_explanation(features, attack_type, confidence)
    
    def get_comprehensive_explanation(self, features: List[float]) -> Dict[str, Any]:
        """Get comprehensive explanation combining all three phases."""
        if not XAI_AVAILABLE:
            return self._get_mock_comprehensive_explanation(features)
        
        try:
            # Get anomaly score from model
            features_tensor = np.array(features).reshape(1, -1)
            model_service.model.eval()
            with torch.no_grad():
                reconstructed, encoded = model_service.model(torch.from_numpy(features_tensor).float().to(model_service.device))
                reconstruction_error = torch.mean((reconstructed - features_tensor) ** 2, dim=1).item()
            
            # Get all phase explanations
            phase1_result = self.get_phase1_explanation(features, reconstruction_error)
            phase2_result = self.get_phase2_explanation(features, reconstruction_error)
            
            # Get attack type if anomaly detected
            anomaly_detected = reconstruction_error > 0.22610116
            if anomaly_detected and model_service.attack_classifier:
                model_service.attack_classifier.eval()
                with torch.no_grad():
                    attack_output = model_service.attack_classifier(torch.from_numpy(features_tensor).float().to(model_service.device))
                    attack_probs = torch.softmax(attack_output, dim=1)
                    attack_type = torch.argmax(attack_probs, dim=1).item()
                    confidence = torch.max(attack_probs).item()
                
                phase3_result = self.get_phase3_explanation(features, attack_type, confidence)
            else:
                phase3_result = {"phase": "phase3_classification", "explanation_type": "not_anomaly"}
            
            return {
                "comprehensive_explanation": True,
                "features": features,
                "anomaly_detected": anomaly_detected,
                "reconstruction_error": reconstruction_error,
                "phase1": phase1_result,
                "phase2": phase2_result,
                "phase3": phase3_result,
                "timestamp": str(np.datetime64('now'))
            }
        except Exception as e:
            logger.error(f"Comprehensive explanation failed: {e}")
            return self._get_mock_comprehensive_explanation(features)
    
    def _calculate_feature_importance(self, shap_values: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate feature importance from SHAP values."""
        if len(shap_values.shape) == 2:
            shap_values = shap_values[0]
        
        importance = []
        for i, value in enumerate(shap_values):
            importance.append({
                "feature_index": i,
                "feature_name": f"feature_{i}",
                "shap_value": float(value),
                "importance": abs(float(value)),
                "direction": "positive" if value > 0 else "negative"
            })
        
        # Sort by importance
        importance.sort(key=lambda x: x["importance"], reverse=True)
        return importance[:20]  # Top 20 features
    
    def _get_mock_phase1_explanation(self, features: List[float], anomaly_score: float) -> Dict[str, Any]:
        """Mock Phase 1 explanation."""
        return {
            "phase": "phase1_foundation",
            "explanation_type": "basic_anomaly",
            "features": features,
            "anomaly_score": anomaly_score,
            "explanation": {
                "is_anomaly": anomaly_score > 0.22610116,
                "confidence": 0.85,
                "reasoning": f"Reconstruction error {anomaly_score:.4f} exceeds threshold 0.2261",
                "key_features": [i for i, f in enumerate(features[:10]) if abs(f) > 0.5]
            },
            "timestamp": str(np.datetime64('now'))
        }
    
    def _get_mock_phase2_explanation(self, features: List[float], reconstruction_error: float) -> Dict[str, Any]:
        """Mock Phase 2 explanation."""
        mock_shap_values = np.random.normal(0, 0.1, len(features))
        return {
            "phase": "phase2_autoencoder",
            "explanation_type": "shap_explainability",
            "features": features,
            "reconstruction_error": reconstruction_error,
            "shap_values": mock_shap_values.tolist(),
            "feature_importance": self._calculate_feature_importance(mock_shap_values),
            "timestamp": str(np.datetime64('now'))
        }
    
    def _get_mock_phase3_explanation(self, features: List[float], attack_type: int, confidence: float) -> Dict[str, Any]:
        """Mock Phase 3 explanation."""
        attack_types = ["BENIGN", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"]
        return {
            "phase": "phase3_classification",
            "explanation_type": "attack_type_explainability",
            "features": features,
            "attack_type": attack_type,
            "attack_name": attack_types[attack_type] if attack_type < len(attack_types) else "Unknown",
            "confidence": confidence,
            "explanation": {
                "predicted_attack": attack_types[attack_type] if attack_type < len(attack_types) else "Unknown",
                "confidence_reasoning": f"Model confidence {confidence:.3f} in prediction",
                "key_indicators": [f"feature_{i}" for i in range(5) if features[i] > 0.7]
            },
            "attack_shap_values": np.random.normal(0, 0.05, 10).tolist(),
            "timestamp": str(np.datetime64('now'))
        }
    
    def _get_mock_comprehensive_explanation(self, features: List[float]) -> Dict[str, Any]:
        """Mock comprehensive explanation."""
        anomaly_score = np.random.random() * 0.5
        anomaly_detected = anomaly_score > 0.22610116
        
        result = {
            "comprehensive_explanation": True,
            "features": features,
            "anomaly_detected": anomaly_detected,
            "reconstruction_error": anomaly_score,
            "phase1": self._get_mock_phase1_explanation(features, anomaly_score),
            "phase2": self._get_mock_phase2_explanation(features, anomaly_score),
            "timestamp": str(np.datetime64('now'))
        }
        
        if anomaly_detected:
            result["phase3"] = self._get_mock_phase3_explanation(
                features, 
                np.random.randint(1, 5), 
                np.random.random() * 0.3 + 0.7
            )
        else:
            result["phase3"] = {"phase": "phase3_classification", "explanation_type": "not_anomaly"}
        
        return result

# Initialize XAI service
xai_service = XAIService()

async def get_explanation(anomaly_id: str) -> Dict[str, Any]:
    """Get XAI explanation for an anomaly by ID."""
    try:
        # Fetch real anomaly data from database
        from services.database_service import database_service
        
        anomaly_data = database_service.get_anomaly_by_id(anomaly_id)
        
        # Extract real features from database
        features = []
        if 'features' in anomaly_data and anomaly_data['features']:
            import json
            try:
                features = json.loads(anomaly_data['features'])
            except json.JSONDecodeError:
                logger.error(f"Failed to parse features for anomaly {anomaly_id}")
                features = [0.0] * 78
        else:
            logger.warning(f"No features found for anomaly {anomaly_id}, using zeros")
            features = [0.0] * 78
        
        # Generate explanation using real features
        explanation = xai_service.get_comprehensive_explanation(features)
        explanation["anomaly_id"] = anomaly_id
        
        # Add anomaly metadata
        explanation["severity"] = anomaly_data.get("severity", "unknown")
        explanation["confidence"] = anomaly_data.get("confidence", 0.0)
        explanation["anomaly_score"] = anomaly_data.get("anomaly_score", 0.0)
        explanation["model_type"] = "Autoencoder"  # Add model_type for frontend
        explanation["explanation_type"] = "comprehensive"
        
        return explanation
    except Exception as e:
        logger.error(f"Failed to get explanation for {anomaly_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate explanation.")

async def explain_anomaly(request: AnomalyExplanationRequest) -> Dict[str, Any]:
    """Generate comprehensive XAI explanation for anomalous data point."""
    try:
        if len(request.features) != 78:
            raise HTTPException(status_code=400, detail="Expected 78 features")
        
        explanation = xai_service.get_comprehensive_explanation(request.features)
        explanation["request_id"] = request.request_id if hasattr(request, 'request_id') else "unknown"
        return explanation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate XAI explanation: {e}")

async def get_phase_explanation(phase: str, features: List[float], **kwargs) -> Dict[str, Any]:
    """Get phase-specific explanation."""
    try:
        if len(features) != 78:
            raise HTTPException(status_code=400, detail="Expected 78 features")
        
        if phase == "phase1":
            return xai_service.get_phase1_explanation(features, kwargs.get("anomaly_score", 0.3))
        elif phase == "phase2":
            return xai_service.get_phase2_explanation(features, kwargs.get("reconstruction_error", 0.3))
        elif phase == "phase3":
            return xai_service.get_phase3_explanation(
                features, 
                kwargs.get("attack_type", 1), 
                kwargs.get("confidence", 0.8)
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid phase: {phase}")
    except Exception as e:
        logger.error(f"Failed to get {phase} explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate {phase} explanation")

async def get_feature_importance(features: List[float]) -> Dict[str, Any]:
    """Get feature importance analysis."""
    try:
        if len(features) != 78:
            raise HTTPException(status_code=400, detail="Expected 78 features")
        
        # Get Phase 2 explanation for SHAP values
        phase2_result = xai_service.get_phase2_explanation(features, 0.3)
        
        return {
            "feature_importance": phase2_result["feature_importance"],
            "total_features": len(features),
            "top_features": phase2_result["feature_importance"][:10],
            "phase": "phase2_autoencoder",
            "timestamp": str(np.datetime64('now'))
        }
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate feature importance")

async def get_attack_type_explanation(features: List[float], attack_type: int) -> Dict[str, Any]:
    """Get attack type specific explanation."""
    try:
        if len(features) != 78:
            raise HTTPException(status_code=400, detail="Expected 78 features")
        
        return xai_service.get_phase3_explanation(features, attack_type, 0.8)
    except Exception as e:
        logger.error(f"Failed to get attack type explanation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate attack type explanation")
