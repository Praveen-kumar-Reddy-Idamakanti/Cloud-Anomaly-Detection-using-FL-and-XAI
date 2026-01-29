"""
Training-related API routes.
"""

from fastapi import HTTPException, BackgroundTasks

from models.schemas import TrainingRequest
from services.training_service import training_service


def get_training_status() -> dict:
    """Get current training status."""
    return training_service.get_training_status()


def start_training(request: TrainingRequest, background_tasks: BackgroundTasks) -> dict:
    """Start model training."""
    try:
        result = training_service.start_training(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to start training.")


def stop_training() -> dict:
    """Stop ongoing training process."""
    try:
        result = training_service.stop_training()
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to stop training.")
