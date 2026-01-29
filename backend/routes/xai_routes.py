"""
XAI (Explainable AI) related API routes.
"""

from fastapi import HTTPException

from models.schemas import AnomalyExplanationRequest
from services.xai_service import xai_service


def get_explanation(anomaly_id: str) -> dict:
    """Get XAI explanation for an anomaly."""
    try:
        # For now, return mock explanation
        explanation = xai_service.get_mock_explanation(anomaly_id)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to generate explanation.")


def explain_anomaly(request: AnomalyExplanationRequest) -> dict:
    """Generate SHAP explanation for a given anomalous data point."""
    try:
        explanation = xai_service.generate_shap_explanation(request.features)
        return explanation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP explanation: {e}")
