"""
Anomaly-related API routes.
"""

from fastapi import HTTPException
from typing import List

from models.schemas import (
    AnomalyData,
    StatsResponse,
    TrainingHistoryEntry
)
from services.database_service import database_service


def get_system_stats() -> StatsResponse:
    """Get system statistics from the database."""
    try:
        stats = database_service.get_system_stats()
        return StatsResponse(**stats)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics.")


def get_training_history() -> List[TrainingHistoryEntry]:
    """Get historical training metrics from the database."""
    try:
        history = database_service.get_training_history()
        return [TrainingHistoryEntry(**entry) for entry in history]
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve training history.")


def get_anomalies(page: int = 1, limit: int = 10) -> List[AnomalyData]:
    """Get paginated list of anomalies from the database."""
    try:
        anomalies = database_service.get_anomalies(page, limit)
        return [AnomalyData(**anomaly) for anomaly in anomalies]
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve anomalies.")


def get_anomaly_by_id(anomaly_id: str) -> AnomalyData:
    """Get specific anomaly by ID from the database."""
    try:
        anomaly = database_service.get_anomaly_by_id(anomaly_id)
        return AnomalyData(**anomaly)
    except ValueError as e:
        if "Anomaly not found" in str(e):
            raise HTTPException(status_code=404, detail="Anomaly not found")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve anomaly.")


def review_anomaly(anomaly_id: str, reviewed: bool) -> dict:
    """Mark anomaly as reviewed in the database."""
    try:
        result = database_service.review_anomaly(anomaly_id, reviewed)
        return result
    except ValueError as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update anomaly review status.")


def report_anomaly(anomaly: AnomalyData) -> dict:
    """Report a new anomaly from a client and store it in the database."""
    try:
        result = database_service.report_anomaly(anomaly.dict())
        return result
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to report anomaly.")
