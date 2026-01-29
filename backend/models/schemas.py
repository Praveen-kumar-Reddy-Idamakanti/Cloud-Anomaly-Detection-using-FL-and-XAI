"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from typing import List, Optional
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


class ModelInfo(BaseModel):
    """Model information response."""
    model_path: str
    input_dim: int
    last_trained: str
    accuracy: Optional[float]
    status: str


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
    features: List[float] = Field(..., description="Single data instance (features) to explain.")


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
