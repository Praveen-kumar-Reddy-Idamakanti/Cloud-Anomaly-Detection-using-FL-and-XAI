"""
Application configuration and setup.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.model_service import model_service
from services.database_service import database_service

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
