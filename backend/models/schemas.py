"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    features: List[List[float]] = Field(..., description="Input features for anomaly detection")
    threshold: Optional[float] = Field(0.4, description="Anomaly detection threshold")


class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection results."""
    predictions: List[int] = Field(..., description="Anomaly predictions (0=normal, 1=anomaly)")
    scores: List[float] = Field(..., description="Anomaly scores")
    threshold: float = Field(..., description="Threshold used for detection")
    confidence: List[float] = Field(..., description="Confidence scores")


class EnhancedDetectionRequest(BaseModel):
    """Enhanced request model for two-stage anomaly detection."""
    features: List[List[float]] = Field(..., description="Input features for anomaly detection (78 features)")
    threshold: Optional[float] = Field(0.22610116, description="Anomaly detection threshold")


class EnhancedDetectionResponse(BaseModel):
    """Enhanced response model for two-stage anomaly detection results."""
    anomaly_predictions: List[int] = Field(..., description="Anomaly predictions (0=normal, 1=anomaly)")
    reconstruction_errors: List[float] = Field(..., description="Reconstruction error scores")
    attack_type_predictions: List[int] = Field(..., description="Attack type predictions (only for anomalies)")
    attack_confidences: List[float] = Field(..., description="Attack type confidences")
    threshold: float = Field(..., description="Threshold used for detection")
    attack_types: List[str] = Field(default=["BENIGN", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"], description="Available attack types")


class ModelInfo(BaseModel):
    """Model information response."""
    model_path: str
    input_dim: int
    last_trained: str
    accuracy: Optional[float]
    status: str
    two_stage_enabled: Optional[bool] = Field(default=False, description="Whether two-stage prediction is enabled")
    attack_types: Optional[List[str]] = Field(default=[], description="Available attack types")


class TrainingRequest(BaseModel):
    """Request model for training initiation."""
    epochs: int = Field(5, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size for training")
    learning_rate: float = Field(0.001, description="Learning rate")


class StatsResponse(BaseModel):
    """System statistics response."""
    total_logs: int
    total_anomalies: int
    critical_anomalies: int
    high_anomalies: int
    medium_anomalies: int
    low_anomalies: int
    alert_rate: float
    avg_confidence: float


class TrainingHistoryEntry(BaseModel):
    """Training history entry model."""
    server_round: int
    avg_loss: Optional[float]
    std_loss: Optional[float]
    avg_accuracy: Optional[float]
    created_at: datetime


class AnomalyData(BaseModel):
    """Anomaly data model."""
    id: str
    timestamp: str
    severity: str
    source_ip: str
    destination_ip: str
    protocol: str
    action: str
    confidence: float
    reviewed: bool
    details: str


class LogData(BaseModel):
    """Log data model."""
    id: str
    timestamp: str
    source_ip: str
    destination_ip: str
    protocol: str
    encrypted: bool
    size: int


class AnomalyExplanationRequest(BaseModel):
    """Request model for anomaly explanation."""
    features: List[float] = Field(..., description="Single data instance (78 features) to explain.")
    threshold: Optional[float] = Field(0.22610116, description="Anomaly detection threshold")
    request_id: Optional[str] = Field(None, description="Request identifier")
    explanation_type: Optional[str] = Field("comprehensive", description="Type of explanation: phase1, phase2, phase3, or comprehensive")


class PhaseExplanationRequest(BaseModel):
    """Request model for phase-specific explanations."""
    phase: str = Field(..., description="XAI phase: phase1, phase2, or phase3")
    features: List[float] = Field(..., description="Input features (78 features)")
    anomaly_score: Optional[float] = Field(None, description="Anomaly score for phase1")
    reconstruction_error: Optional[float] = Field(None, description="Reconstruction error for phase2")
    attack_type: Optional[int] = Field(None, description="Attack type ID")
    confidence: Optional[float] = Field(None, description="Confidence score for phase3")


class FeatureImportanceRequest(BaseModel):
    """Request model for feature importance analysis."""
    features: List[float] = Field(..., description="Input features (78 features)")
    top_k: Optional[int] = Field(10, description="Number of top features to return")


class AttackTypeExplanationRequest(BaseModel):
    """Request model for attack type explanation."""
    features: List[float] = Field(..., description="Input features (78 features)")
    attack_type: int = Field(..., description="Attack type ID")
    confidence: Optional[float] = Field(None, description="Confidence score")


class AnomalyExplanationResponse(BaseModel):
    """Response model for anomaly explanation."""
    explanation_type: str = Field(..., description="Type of explanation provided")
    features: List[float] = Field(..., description="Input features that were explained")
    anomaly_detected: bool = Field(..., description="Whether anomaly was detected")
    reconstruction_error: Optional[float] = Field(None, description="Reconstruction error if applicable")
    confidence: Optional[float] = Field(None, description="Confidence score of detection")
    phase1: Optional[Dict[str, Any]] = Field(None, description="Phase 1 explanation data")
    phase2: Optional[Dict[str, Any]] = Field(None, description="Phase 2 explanation data")
    phase3: Optional[Dict[str, Any]] = Field(None, description="Phase 3 explanation data")
    timestamp: str = Field(..., description="Explanation generation timestamp")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model_loaded: bool
    environment: str


class RootResponse(BaseModel):
    """Root endpoint response."""
    message: str
    version: str
    status: str
    model_loaded: bool
    environment: str
    system_type: str
    warning: Optional[str]
    processing_time: str
