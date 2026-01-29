"""
Main FastAPI application entry point.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from typing import List

# Import configuration
from config.app_config import create_app, is_production
from models.schemas import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    ModelInfo,
    TrainingRequest,
    StatsResponse,
    TrainingHistoryEntry,
    AnomalyData,
    LogData,
    AnomalyExplanationRequest
)

# Import route handlers
from routes.model_routes import (
    get_model_info,
    detect_anomalies,
    health_check,
    root_endpoint
)
from routes.anomaly_routes import (
    get_system_stats,
    get_training_history,
    get_anomalies,
    get_anomaly_by_id,
    review_anomaly,
    report_anomaly
)
from routes.xai_routes import (
    get_explanation,
    explain_anomaly
)
from routes.training_routes import (
    get_training_status,
    start_training,
    stop_training
)
from routes.data_routes import (
    get_logs,
    upload_log,
    stream_realtime_data
)

# Configure logging
logger = logging.getLogger(__name__)

# Create FastAPI application
app = create_app()

# Model and Health Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return root_endpoint()

@app.get("/health")
async def health():
    """Health check endpoint for load balancers."""
    return health_check()

@app.get("/model/info", response_model=ModelInfo)
async def model_info():
    """Get information about the loaded model."""
    return get_model_info()

@app.post("/model/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies_endpoint(request: AnomalyDetectionRequest):
    """Detect anomalies in input features."""
    return detect_anomalies(request)

# Statistics and History Endpoints
@app.get("/stats", response_model=StatsResponse)
async def stats():
    """Get system statistics from the database."""
    return get_system_stats()

@app.get("/history/training", response_model=List[TrainingHistoryEntry])
async def training_history():
    """Get historical training metrics from the database."""
    return get_training_history()

# Anomaly Management Endpoints
@app.get("/anomalies", response_model=List[AnomalyData])
async def anomalies(page: int = 1, limit: int = 10):
    """Get paginated list of anomalies from the database."""
    return get_anomalies(page, limit)

@app.get("/anomalies/{anomaly_id}", response_model=AnomalyData)
async def anomaly(anomaly_id: str):
    """Get specific anomaly by ID from the database."""
    return get_anomaly_by_id(anomaly_id)

@app.post("/anomalies/{anomaly_id}/review")
async def review_anomaly_endpoint(anomaly_id: str, reviewed: bool):
    """Mark anomaly as reviewed in the database."""
    return review_anomaly(anomaly_id, reviewed)

@app.post("/report_anomaly", status_code=201)
async def report_anomaly_endpoint(anomaly: AnomalyData):
    """Report a new anomaly from a client and store it in the database."""
    return report_anomaly(anomaly)

# Training Endpoints
@app.get("/training/status")
async def training_status():
    """Get current training status."""
    return get_training_status()

@app.post("/training/start")
async def start_training_endpoint(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training."""
    return start_training(request, background_tasks)

@app.post("/training/stop")
async def stop_training_endpoint():
    """Stop ongoing training process."""
    return stop_training()

# XAI (Explainable AI) Endpoints
@app.get("/explanations/{anomaly_id}")
async def explanation(anomaly_id: str):
    """Get XAI explanation for an anomaly."""
    return get_explanation(anomaly_id)

@app.post("/explain_anomaly")
async def explain_anomaly_endpoint(request: AnomalyExplanationRequest):
    """Generate SHAP explanation for a given anomalous data point."""
    return explain_anomaly(request)

# Data Management Endpoints
@app.get("/logs", response_model=List[LogData])
async def logs(page: int = 1, limit: int = 10):
    """Get paginated list of logs."""
    return get_logs(page, limit)

@app.post("/logs/upload")
async def upload_log_endpoint(file: UploadFile = File(...)):
    """Upload log file for federated anomaly detection processing."""
    return await upload_log(file)

@app.get("/realtime/stream")
async def realtime_stream():
    """Stream real-time data updates."""
    return stream_realtime_data()


if __name__ == "__main__":
    import uvicorn
    
    # Check for gunicorn availability
    try:
        import gunicorn
        GUNICORN_AVAILABLE = True
    except ImportError:
        GUNICORN_AVAILABLE = False
    
    # Get port from environment variable (required for Render)
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    # Use gunicorn in production if available, uvicorn otherwise
    if is_production():
        logger.info(f"Starting production server on {host}:{port}")
        if GUNICORN_AVAILABLE:
            logger.info("Gunicorn available - using uvicorn with production settings")
        else:
            logger.warning("Gunicorn not available - using uvicorn (install gunicorn for better production performance)")
        uvicorn.run(app, host=host, port=port, workers=1)
    else:
        logger.info(f"Starting development server on {host}:{port}")
        # For development with reload, use the app object directly but disable reload to avoid the warning
        uvicorn.run(app, host=host, port=port, reload=False)
