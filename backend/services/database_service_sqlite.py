"""
Database service for handling SQLite operations (migrated from Supabase).
"""

import logging
import traceback
from typing import List, Dict, Any, Optional
import numpy as np

# Import SQLite database service
from .sqlite_database_service import sqlite_database_service

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service class for database operations (now using SQLite)."""
    
    def __init__(self):
        self.sqlite_service = sqlite_database_service
        logger.info("DatabaseService initialized with SQLite backend")
    
    def is_connected(self) -> bool:
        """Check if database connection is available."""
        return self.sqlite_service.is_connected()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics from the database.
        
        Returns:
            Dictionary containing system statistics
        """
        return self.sqlite_service.get_system_stats()
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get historical training metrics from the database.
        
        Returns:
            List of training history entries
        """
        return self.sqlite_service.get_training_history()
    
    def get_anomalies(self, page: int = 1, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get paginated list of anomalies from the database.
        
        Args:
            page: Page number (1-based)
            limit: Number of items per page
            
        Returns:
            List of anomaly records
        """
        return self.sqlite_service.get_anomalies(page, limit)
    
    def get_anomaly_by_id(self, anomaly_id: str) -> Dict[str, Any]:
        """
        Get specific anomaly by ID from the database.
        
        Args:
            anomaly_id: ID of the anomaly to retrieve
            
        Returns:
            Anomaly record
        """
        return self.sqlite_service.get_anomaly_by_id(anomaly_id)
    
    def review_anomaly(self, anomaly_id: str, reviewed: bool) -> Dict[str, str]:
        """
        Mark anomaly as reviewed in the database.
        
        Args:
            anomaly_id: ID of the anomaly to update
            reviewed: Whether the anomaly has been reviewed
            
        Returns:
            Success message
        """
        return self.sqlite_service.review_anomaly(anomaly_id, reviewed)
    
    def report_anomaly(self, anomaly_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Report a new anomaly from a client and store it in the database.
        
        Args:
            anomaly_data: Anomaly data to insert
            
        Returns:
            Success message with anomaly ID
        """
        return self.sqlite_service.report_anomaly(anomaly_data)
    
    def get_logs(self, page: int = 1, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get paginated list of logs from the database.
        
        Args:
            page: Page number (1-based)
            limit: Number of items per page
            
        Returns:
            List of log records
        """
        return self.sqlite_service.get_logs(page, limit)
    
    def upload_log(self, log_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Upload a new log entry to the database.
        
        Args:
            log_data: Log data to insert
            
        Returns:
            Success message with log ID
        """
        return self.sqlite_service.upload_log(log_data)
    
    def add_training_run(self, training_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Add a new training run record.
        
        Args:
            training_data: Training run data to insert
            
        Returns:
            Success message with training run ID
        """
        return self.sqlite_service.add_training_run(training_data)
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        return self.sqlite_service.get_database_info()


# Global database service instance
database_service = DatabaseService()
