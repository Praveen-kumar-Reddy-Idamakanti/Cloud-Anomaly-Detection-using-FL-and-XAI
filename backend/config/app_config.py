"""
Application configuration and setup.
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path Configuration
class PathConfig:
    """Centralized path configuration from environment variables."""
    
    def __init__(self):
        self.project_root = Path(os.getenv("PROJECT_ROOT", "."))
        self.ai_root = Path(os.getenv("AI_ROOT", self.project_root / "AI"))
        self.model_development_path = Path(os.getenv("MODEL_DEVELOPMENT_PATH", self.ai_root / "model_development"))
        self.model_artifacts_path = Path(os.getenv("MODEL_ARTIFACTS_PATH", self.ai_root / "model_artifacts"))
        self.data_preprocessing_path = Path(os.getenv("DATA_PREPROCESSING_PATH", self.ai_root / "data_preprocessing"))
        self.federated_learning_path = Path(os.getenv("FEDERATED_LEARNING_PATH", self.project_root / "federated_anomaly_detection"))
        self.logs_path = Path(os.getenv("LOGS_PATH", self.project_root / "logs"))
    
    def get_model_input_dim(self) -> int:
        """Get model input dimension from environment."""
        return int(os.getenv("MODEL_INPUT_DIM", "78"))
    
    def get_attack_types(self) -> list:
        """Get attack types from environment."""
        attack_types_str = os.getenv("ATTACK_TYPES", '["BENIGN", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"]')
        try:
            import ast
            return ast.literal_eval(attack_types_str)
        except:
            return ["BENIGN", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"]
    
    def get_anomaly_threshold(self) -> float:
        """Get anomaly threshold from environment."""
        return float(os.getenv("ANOMALY_THRESHOLD", "0.22610116"))


# Global path configuration instance
path_config = PathConfig()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    # Import services here to avoid circular imports
    from services.model_service import model_service
    from services.database_service import database_service
    
    # Startup
    logger.info("Starting up API server...")
    
    # Initialize database connection
    if not database_service.is_connected():
        logger.warning("Database connection not available. Some endpoints will not work.")
    
    # Try to load the latest model
    model_data = model_service.load_latest_model()
    if model_data:
        logger.info(f"Model loaded successfully: {model_data}")
    else:
        logger.warning("No model loaded - using mock data only")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
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
    
    # Add CORS preflight handler
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
    
    return app


def get_environment() -> str:
    """Get the current environment."""
    return os.getenv("ENVIRONMENT", "development")


def is_production() -> bool:
    """Check if running in production environment."""
    return get_environment() == "production"


# Path Configuration
class PathConfig:
    """Centralized path configuration from environment variables."""
    
    def __init__(self):
        self.project_root = Path(os.getenv("PROJECT_ROOT", "."))
        self.ai_root = Path(os.getenv("AI_ROOT", self.project_root / "AI"))
        self.model_development_path = Path(os.getenv("MODEL_DEVELOPMENT_PATH", self.ai_root / "model_development"))
        self.model_artifacts_path = Path(os.getenv("MODEL_ARTIFACTS_PATH", self.ai_root / "model_artifacts"))
        self.data_preprocessing_path = Path(os.getenv("DATA_PREPROCESSING_PATH", self.ai_root / "data_preprocessing"))
        self.federated_learning_path = Path(os.getenv("FEDERATED_LEARNING_PATH", self.project_root / "federated_anomaly_detection"))
        self.logs_path = Path(os.getenv("LOGS_PATH", self.project_root / "logs"))
    
    def get_model_input_dim(self) -> int:
        """Get model input dimension from environment."""
        return int(os.getenv("MODEL_INPUT_DIM", "78"))
    
    def get_attack_types(self) -> list:
        """Get attack types from environment."""
        attack_types_str = os.getenv("ATTACK_TYPES", '["BENIGN", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"]')
        try:
            import ast
            return ast.literal_eval(attack_types_str)
        except:
            return ["BENIGN", "DoS GoldenEye", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris"]
    
    def get_anomaly_threshold(self) -> float:
        """Get anomaly threshold from environment."""
        return float(os.getenv("ANOMALY_THRESHOLD", "0.22610116"))


# Global path configuration instance
path_config = PathConfig()
