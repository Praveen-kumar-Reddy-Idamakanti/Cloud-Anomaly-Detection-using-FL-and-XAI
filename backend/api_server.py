"""
FastAPI server to bridge the federated anomaly detection model with the frontend.
This server exposes REST API endpoints for model inference, training status, and data management.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import traceback # Import traceback module
import shap # Import shap library
try:
    import gunicorn
    GUNICORN_AVAILABLE = True
except ImportError:
    GUNICORN_AVAILABLE = False

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from federated_anomaly_detection.models.autoencoder import create_model, AnomalyDetector
from use_model import load_model, detect_anomalies

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from federated_anomaly_detection.server.supabase_client import get_supabase_client
from supabase import Client

# Supabase client
supabase: Optional[Client] = None

# Global variables for model and data
model = None
model_info = {}
device = None
data_cache = {}
training_status = {"is_training": False, "progress": 0, "round": 0}

# Lifespan event handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    global supabase
    # Startup
    logger.info("Starting up API server...")
    
    # Initialize Supabase client
    try:
        supabase = get_supabase_client()
        logger.info("Supabase client initialized successfully.")
    except ValueError as e:
        logger.warning(f"Failed to initialize Supabase client: {e}. Some endpoints will not work.")

    # Try to load the latest model
    model_data = load_latest_model()
    if model_data:
        logger.info(f"Model loaded successfully: {model_data}")
    else:
        logger.warning("No model loaded - using mock data only")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")

# Initialize FastAPI app with lifespan handler
app = FastAPI(
    title="Federated Anomaly Detection API",
    description="API for federated anomaly detection model inference and management",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for production
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
if os.getenv("ENVIRONMENT") == "production":
    # In production, only allow specific origins
    cors_origins = [origin.strip() for origin in cors_origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class AnomalyDetectionRequest(BaseModel):
    features: List[List[float]] = Field(..., description="Input features for anomaly detection")
    threshold: Optional[float] = Field(0.4, description="Anomaly detection threshold")

class AnomalyDetectionResponse(BaseModel):
    predictions: List[int] = Field(..., description="Anomaly predictions (0=normal, 1=anomaly)")
    scores: List[float] = Field(..., description="Anomaly scores")
    threshold: float = Field(..., description="Threshold used for detection")
    confidence: List[float] = Field(..., description="Confidence scores")

class ModelInfo(BaseModel):
    model_path: str
    input_dim: int
    last_trained: str
    accuracy: Optional[float]
    status: str

class TrainingRequest(BaseModel):
    epochs: int = Field(5, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size for training")
    learning_rate: float = Field(0.001, description="Learning rate")

class StatsResponse(BaseModel):
    total_logs: int
    total_anomalies: int
    critical_anomalies: int
    high_anomalies: int
    medium_anomalies: int
    low_anomalies: int
    alert_rate: float
    avg_confidence: float

class TrainingHistoryEntry(BaseModel):
    server_round: int
    avg_loss: Optional[float]
    std_loss: Optional[float]
    avg_accuracy: Optional[float]
    created_at: datetime

class AnomalyData(BaseModel):
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

# Utility functions
def load_latest_model():
    """Load the latest trained model."""
    global model, model_info, device
    
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

# Function to wrap the model for SHAP
def shap_prediction_function(data: np.ndarray) -> np.ndarray:
    """
    Prediction function for SHAP: takes a numpy array and returns anomaly scores.
    """
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

# Mock data functions are removed as we are moving to a real database backend.


# API Endpoints

@app.middleware("http")
async def cors_handler(request, call_next):
    """Handle CORS preflight requests."""
    if request.method == "OPTIONS":
        response = JSONResponse(content={})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response
    
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.get("/")
async def root():
    """Root endpoint with API information."""
    environment = os.getenv("ENVIRONMENT", "development")
    return {
        "message": "Federated Anomaly Detection API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
        "environment": environment,
        "system_type": "production" if environment == "production" else "research_demonstration",
        "warning": "This is a research system not suitable for production use" if environment != "production" else None,
        "processing_time": "5-15 minutes for federated anomaly detection"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return ModelInfo(
        model_path=model_info.get("model_path", "unknown"),
        input_dim=model_info.get("input_dim", 9),
        last_trained=model_info.get("last_trained", "unknown"),
        accuracy=model_info.get("accuracy"),
        status="loaded"
    )

@app.post("/model/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies_endpoint(request: AnomalyDetectionRequest):
    """Detect anomalies in input features."""
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    try:
        # Convert input to numpy array
        features = np.array(request.features, dtype=np.float32)
        
        # Ensure correct input dimension
        expected_dim = model_info.get("input_dim", 9)
        if features.shape[1] != expected_dim:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {expected_dim} features, got {features.shape[1]}"
            )
        
        # Detect anomalies
        results = detect_anomalies(model, features, request.threshold, device)
        
        return AnomalyDetectionResponse(
            predictions=results['is_anomaly'].tolist(),
            scores=results['anomaly_scores'].tolist(),
            threshold=request.threshold,
            confidence=results['anomaly_scores'].tolist()  # Using scores as confidence
        )
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics from the database."""
    if supabase is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    try:
        # Get total logs from training runs for now
        # In a real system, this would come from a dedicated logs table
        logs_response = supabase.table("training_runs").select("count", count="exact").execute()
        total_logs = logs_response.count if logs_response.count is not None else 0

        # Get anomaly counts
        anomalies_response = supabase.table("anomalies").select("severity", "confidence").execute()
        
        anomalies_data = anomalies_response.data if anomalies_response.data is not None else []
        
        total_anomalies = len(anomalies_data)
        critical_anomalies = sum(1 for a in anomalies_data if a['severity'] == 'critical')
        high_anomalies = sum(1 for a in anomalies_data if a['severity'] == 'high')
        medium_anomalies = sum(1 for a in anomalies_data if a['severity'] == 'medium')
        low_anomalies = sum(1 for a in anomalies_data if a['severity'] == 'low')
        
        avg_confidence = np.mean([a['confidence'] for a in anomalies_data]) if total_anomalies > 0 else 0.0

        return StatsResponse(
            total_logs=total_logs,
            total_anomalies=total_anomalies,
            critical_anomalies=critical_anomalies,
            high_anomalies=high_anomalies,
            medium_anomalies=medium_anomalies,
            low_anomalies=low_anomalies,
            alert_rate=round((total_anomalies / total_logs) * 100, 2) if total_logs > 0 else 0.0,
            avg_confidence=round(float(avg_confidence), 2)
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        traceback.print_exc() # Print full traceback
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics.")

@app.get("/history/training", response_model=List[TrainingHistoryEntry])
async def get_training_history():
    """Get historical training metrics from the database."""
    if supabase is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    try:
        response = supabase.table("training_runs").select(
            "server_round, avg_loss, std_loss, avg_accuracy, created_at"
        ).order("server_round", desc=False).execute()

        if response.data is None:
            return []
        
        return response.data
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        traceback.print_exc() # Print full traceback
        raise HTTPException(status_code=500, detail="Failed to retrieve training history.")

@app.get("/anomalies", response_model=List[AnomalyData])
async def get_anomalies(page: int = 1, limit: int = 10):
    """Get paginated list of anomalies from the database."""
    if supabase is None:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    try:
        start_index = (page - 1) * limit
        response = supabase.table("anomalies").select("*").order("timestamp", desc=True).range(start_index, start_index + limit - 1).execute()
        
        if response.data is None:
            return []
            
        return response.data
    except Exception as e:
        logger.error(f"Failed to get anomalies: {e}")
        traceback.print_exc() # Print full traceback
        raise HTTPException(status_code=500, detail="Failed to retrieve anomalies.")

@app.get("/anomalies/{anomaly_id}", response_model=AnomalyData)
async def get_anomaly(anomaly_id: str):
    """Get specific anomaly by ID from the database."""
    if supabase is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    try:
        response = supabase.table("anomalies").select("*").eq("id", anomaly_id).single().execute()
        
        if response.data is None:
            raise HTTPException(status_code=404, detail="Anomaly not found")
            
        return response.data
    except Exception as e:
        logger.error(f"Failed to get anomaly {anomaly_id}: {e}")
        traceback.print_exc() # Print full traceback
        raise HTTPException(status_code=500, detail="Failed to retrieve anomaly.")

@app.post("/anomalies/{anomaly_id}/review")
async def review_anomaly(anomaly_id: str, reviewed: bool):
    """Mark anomaly as reviewed in the database."""
    if supabase is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    try:
        response = supabase.table("anomalies").update({"reviewed": reviewed}).eq("id", anomaly_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Anomaly with id {anomaly_id} not found.")

        return {"message": f"Anomaly {anomaly_id} marked as {'reviewed' if reviewed else 'unreviewed'}"}
    except Exception as e:
        logger.error(f"Failed to review anomaly {anomaly_id}: {e}")
        traceback.print_exc() # Print full traceback
        raise HTTPException(status_code=500, detail="Failed to update anomaly review status.")

@app.post("/report_anomaly", status_code=201)
async def report_anomaly(anomaly: AnomalyData):
    """Report a new anomaly from a client and store it in the database."""
    if supabase is None:
        raise HTTPException(status_code=503, detail="Database connection not available")

    try:
        response = supabase.table("anomalies").insert(anomaly.dict()).execute()

        if not response.data:
            raise HTTPException(status_code=500, detail="Failed to insert anomaly into database.")

        return {"message": "Anomaly reported successfully", "anomaly_id": response.data[0]['id']}
    except Exception as e:
        logger.error(f"Failed to report anomaly: {e}")
        traceback.print_exc() # Print full traceback
        raise HTTPException(status_code=500, detail="Failed to report anomaly.")

@app.get("/logs", response_model=List[LogData])
async def get_logs(page: int = 1, limit: int = 10):
    """Get paginated list of logs."""
    # This endpoint is a placeholder for a future implementation
    # that would fetch logs from a dedicated logging backend or database table.
    return []

@app.post("/logs/upload")
async def upload_log(file: UploadFile = File(...)):
    """Upload log file for federated anomaly detection processing."""
    # Simulate processing time for federated learning
    import time
    time.sleep(2)  # Simulate initial processing
    
    # In a real implementation, this would process the uploaded file
    return {
        "message": f"File {file.filename} uploaded successfully for federated processing",
        "size": file.size,
        "timestamp": datetime.now().isoformat(),
        "processing_note": "This is a research system. Processing may take 5-15 minutes for federated anomaly detection.",
        "status": "queued_for_analysis"
    }

@app.get("/training/status")
async def get_training_status():
    """Get current training status."""
    return training_status

@app.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training."""
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    # Start training in background
    background_tasks.add_task(simulate_training, request)
    
    return {"message": "Training started", "request": request.dict()}

async def simulate_training(request: TrainingRequest):
    """Simulate training process."""
    global training_status
    
    training_status["is_training"] = True
    training_status["progress"] = 0
    training_status["round"] = 0
    
    try:
        # Simulate training rounds
        for round_num in range(request.epochs):
            training_status["round"] = round_num + 1
            
            # Simulate progress within each round
            for progress in range(0, 101, 10):
                training_status["progress"] = progress
                await asyncio.sleep(0.5)  # Simulate work
        
        # Training completed
        training_status["is_training"] = False
        training_status["progress"] = 100
        
        logger.info("Training simulation completed")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_status["is_training"] = False
        training_status["progress"] = 0

@app.get("/realtime/stream")
async def stream_realtime_data():
    """Stream real-time data updates."""
    # This would typically use Server-Sent Events (SSE)
    # For now, return a simple endpoint
    return {
        "message": "Real-time streaming endpoint",
        "note": "Implement SSE for real-time updates"
    }

@app.get("/explanations/{anomaly_id}")
async def get_explanation(anomaly_id: str):
    """Get XAI explanation for an anomaly."""
    # Generate mock explanation
    explanation = {
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
    
    return explanation

@app.post("/explain_anomaly")
async def explain_anomaly(request: AnomalyExplanationRequest):
    """Generate SHAP explanation for a given anomalous data point."""
    global model, model_info, device

    if model is None:
        raise HTTPException(status_code=404, detail="Model not loaded. Cannot generate explanation.")

    try:
        # The input_dim from model_info is crucial for SHAP background data
        input_dim = model_info.get("input_dim", 9)
        if input_dim is None:
            raise HTTPException(status_code=500, detail="Model input dimension unknown.")

        # Create a background dataset for SHAP
        # For simplicity, using a small random background, but ideally,
        # this should be a representative sample of non-anomalous data.
        # Ensure the background data has the correct input_dim
        background_data = np.random.rand(100, input_dim).astype(np.float32)

        # Create a KernelExplainer
        explainer = shap.KernelExplainer(shap_prediction_function, background_data)

        # Convert the single instance to explain to a numpy array
        instance_to_explain = np.array([request.features], dtype=np.float32)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(instance_to_explain)

        # For autoencoders, shap_values will be an array of shap values for each output feature.
        # We want the explanation for the anomaly score, which is a single output from shap_prediction_function.
        # The explainer for KernelExplainer with a single output function returns shap_values as a single array.
        
        # Map SHAP values to features. Assuming feature names are not available,
        # use generic "feature_X"
        feature_importances = []
        for i, value in enumerate(shap_values[0]): # Assuming shap_values[0] for single output
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
        traceback.print_exc() # Print full traceback
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP explanation: {e}")

if __name__ == "__main__":
    # Get port from environment variable (required for Render)
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    # Use gunicorn in production if available, uvicorn otherwise
    if os.getenv("ENVIRONMENT") == "production":
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
