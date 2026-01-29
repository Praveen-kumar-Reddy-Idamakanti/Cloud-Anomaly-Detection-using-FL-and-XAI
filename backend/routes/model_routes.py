"""
Model-related API routes.
"""

import os
import numpy as np
from fastapi import HTTPException
from datetime import datetime

from models.schemas import (
    AnomalyDetectionRequest, 
    AnomalyDetectionResponse, 
    ModelInfo,
    HealthResponse,
    RootResponse
)
from services.model_service import model_service
from services.database_service import database_service


def get_model_info() -> ModelInfo:
    """Get information about the loaded model."""
    if not model_service.is_model_loaded():
        raise HTTPException(status_code=404, detail="No model loaded")
    
    info = model_service.get_model_info()
    return ModelInfo(**info)


def detect_anomalies(request: AnomalyDetectionRequest) -> AnomalyDetectionResponse:
    """Detect anomalies in input features."""
    if not model_service.is_model_loaded():
        raise HTTPException(status_code=404, detail="No model loaded")
    
    try:
        # Convert input to numpy array
        features = np.array(request.features, dtype=np.float32)
        
        # Detect anomalies
        results = model_service.detect_anomalies(features, request.threshold)
        
        return AnomalyDetectionResponse(
            predictions=results['is_anomaly'].tolist(),
            scores=results['anomaly_scores'].tolist(),
            threshold=request.threshold,
            confidence=results['anomaly_scores'].tolist()  # Using scores as confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def health_check() -> HealthResponse:
    """Health check endpoint for load balancers."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_service.is_model_loaded(),
        environment=os.getenv("ENVIRONMENT", "development")
    )


def root_endpoint() -> RootResponse:
    """Root endpoint with API information."""
    environment = os.getenv("ENVIRONMENT", "development")
    return RootResponse(
        message="Federated Anomaly Detection API",
        version="1.0.0",
        status="running",
        model_loaded=model_service.is_model_loaded(),
        environment=environment,
        system_type="production" if environment == "production" else "research_demonstration",
        warning="This is a research system not suitable for production use" if environment != "production" else None,
        processing_time="5-15 minutes for federated anomaly detection"
    )
