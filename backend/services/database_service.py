"""
Database service for handling database operations.
"""

import logging
import traceback
import os
import sys
from typing import List, Dict, Any, Optional
import numpy as np

# Add project root to Python path to import database setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from database.sqlite_setup import SQLiteSetup

# Try to import Supabase client, fall back to mock if not available
try:
    from federated_anomaly_detection.server.supabase_client import get_supabase_client
    from supabase import Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("Supabase modules not available, using SQLite database")

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service class for database operations."""
    
    def __init__(self):
        self.supabase: Optional[Any] = None
        self.sqlite_db: Optional[SQLiteSetup] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the database client."""
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase not available - using SQLite database")
            self.sqlite_db = SQLiteSetup()
            return
        
        try:
            self.supabase = get_supabase_client()
            logger.info("Supabase client initialized successfully.")
        except ValueError as e:
            logger.warning(f"Failed to initialize Supabase client: {e}. Falling back to SQLite.")
            self.supabase = None
            self.sqlite_db = SQLiteSetup()
    
    def is_connected(self) -> bool:
        """Check if database connection is available."""
        return self.supabase is not None or self.sqlite_db is not None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics from the database.
        
        Returns:
            Dictionary containing system statistics
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        if self.sqlite_db:
            # Use SQLite database
            try:
                conn = self.sqlite_db.connect()
                cursor = conn.cursor()
                
                # Get system stats from the system_stats table
                cursor.execute('SELECT * FROM system_stats WHERE id = 1')
                system_stats = cursor.fetchone()
                
                if system_stats:
                    return {
                        "total_logs": system_stats['total_logs'],
                        "total_anomalies": system_stats['total_anomalies'],
                        "critical_anomalies": system_stats['critical_anomalies'],
                        "high_anomalies": system_stats['high_anomalies'],
                        "medium_anomalies": system_stats['medium_anomalies'],
                        "low_anomalies": system_stats['low_anomalies'],
                        "alert_rate": system_stats['alert_rate'],
                        "avg_confidence": system_stats['avg_confidence']
                    }
                else:
                    # If no system stats exist, calculate from anomalies table
                    cursor.execute('SELECT severity, confidence FROM anomalies')
                    anomalies = cursor.fetchall()
                    
                    total_anomalies = len(anomalies)
                    critical_anomalies = sum(1 for a in anomalies if a['severity'] == 'critical')
                    high_anomalies = sum(1 for a in anomalies if a['severity'] == 'high')
                    medium_anomalies = sum(1 for a in anomalies if a['severity'] == 'medium')
                    low_anomalies = sum(1 for a in anomalies if a['severity'] == 'low')
                    
                    # Get total logs from logs table
                    cursor.execute('SELECT COUNT(*) as count FROM logs')
                    logs_result = cursor.fetchone()
                    total_logs = logs_result['count'] if logs_result else 0
                    
                    avg_confidence = np.mean([a['confidence'] for a in anomalies]) if total_anomalies > 0 else 0.0
                    alert_rate = round((total_anomalies / total_logs) * 100, 2) if total_logs > 0 else 0.0
                    
                    return {
                        "total_logs": total_logs,
                        "total_anomalies": total_anomalies,
                        "critical_anomalies": critical_anomalies,
                        "high_anomalies": high_anomalies,
                        "medium_anomalies": medium_anomalies,
                        "low_anomalies": low_anomalies,
                        "alert_rate": alert_rate,
                        "avg_confidence": round(float(avg_confidence), 2)
                    }
                    
            except Exception as e:
                logger.error(f"Failed to get stats from SQLite: {e}")
                traceback.print_exc()
                raise ValueError("Failed to retrieve system statistics.")
            finally:
                self.sqlite_db.close()
        
        if not SUPABASE_AVAILABLE:
            # Mock implementation (should not reach here with SQLite)
            return {
                "total_logs": 1000,
                "total_anomalies": 50,
                "critical_anomalies": 5,
                "high_anomalies": 10,
                "medium_anomalies": 15,
                "low_anomalies": 20,
                "alert_rate": 5.0,
                "avg_confidence": 0.75
            }
        
        try:
            # Get total logs from training runs for now
            # In a real system, this would come from a dedicated logs table
            logs_response = self.supabase.table("training_runs").select("count", count="exact").execute()
            total_logs = logs_response.count if logs_response.count is not None else 0
            
            # Get anomaly counts
            anomalies_response = self.supabase.table("anomalies").select("severity", "confidence").execute()
            
            anomalies_data = anomalies_response.data if anomalies_response.data is not None else []
            
            total_anomalies = len(anomalies_data)
            critical_anomalies = sum(1 for a in anomalies_data if a['severity'] == 'critical')
            high_anomalies = sum(1 for a in anomalies_data if a['severity'] == 'high')
            medium_anomalies = sum(1 for a in anomalies_data if a['severity'] == 'medium')
            low_anomalies = sum(1 for a in anomalies_data if a['severity'] == 'low')
            
            avg_confidence = np.mean([a['confidence'] for a in anomalies_data]) if total_anomalies > 0 else 0.0
            
            return {
                "total_logs": total_logs,
                "total_anomalies": total_anomalies,
                "critical_anomalies": critical_anomalies,
                "high_anomalies": high_anomalies,
                "medium_anomalies": medium_anomalies,
                "low_anomalies": low_anomalies,
                "alert_rate": round((total_anomalies / total_logs) * 100, 2) if total_logs > 0 else 0.0,
                "avg_confidence": round(float(avg_confidence), 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            traceback.print_exc()
            raise ValueError("Failed to retrieve system statistics.")
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get historical training metrics from the database.
        
        Returns:
            List of training history entries
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        if self.sqlite_db:
            # Use SQLite database
            try:
                conn = self.sqlite_db.connect()
                cursor = conn.cursor()
                
                cursor.execute('SELECT server_round, avg_loss, std_loss, avg_accuracy, created_at FROM training_runs ORDER BY server_round ASC')
                training_runs = cursor.fetchall()
                
                # Convert Row objects to dictionaries
                result = []
                for run in training_runs:
                    result.append(dict(run))
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get training history from SQLite: {e}")
                traceback.print_exc()
                raise ValueError("Failed to retrieve training history.")
            finally:
                self.sqlite_db.close()
        
        if not SUPABASE_AVAILABLE:
            # Mock implementation
            return [
                {
                    "server_round": 1,
                    "avg_loss": 0.5,
                    "std_loss": 0.1,
                    "avg_accuracy": 0.85,
                    "created_at": "2025-01-01T00:00:00"
                }
            ]
        
        try:
            response = self.supabase.table("training_runs").select(
                "server_round, avg_loss, std_loss, avg_accuracy, created_at"
            ).order("server_round", desc=False).execute()
            
            return response.data if response.data is not None else []
            
        except Exception as e:
            logger.error(f"Failed to get training history: {e}")
            traceback.print_exc()
            raise ValueError("Failed to retrieve training history.")
    
    def get_anomalies(self, page: int = 1, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get paginated list of anomalies from the database.
        
        Args:
            page: Page number (1-based)
            limit: Number of items per page
            
        Returns:
            List of anomaly records
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        if self.sqlite_db:
            # Use SQLite database
            try:
                conn = self.sqlite_db.connect()
                cursor = conn.cursor()
                
                start_index = (page - 1) * limit
                cursor.execute('SELECT * FROM anomalies ORDER BY timestamp DESC LIMIT ? OFFSET ?', (limit, start_index))
                anomalies = cursor.fetchall()
                
                # Convert Row objects to dictionaries
                result = []
                for anomaly in anomalies:
                    result.append(dict(anomaly))
                
                return result
                
            except Exception as e:
                logger.error(f"Failed to get anomalies from SQLite: {e}")
                traceback.print_exc()
                raise ValueError("Failed to retrieve anomalies.")
            finally:
                self.sqlite_db.close()
        
        if not SUPABASE_AVAILABLE:
            # Mock implementation
            return []
        
        try:
            start_index = (page - 1) * limit
            response = self.supabase.table("anomalies").select("*").order("timestamp", desc=True).range(start_index, start_index + limit - 1).execute()
            
            return response.data if response.data is not None else []
            
        except Exception as e:
            logger.error(f"Failed to get anomalies: {e}")
            traceback.print_exc()
            raise ValueError("Failed to retrieve anomalies.")
    
    def get_anomaly_by_id(self, anomaly_id: str) -> Dict[str, Any]:
        """
        Get specific anomaly by ID from the database.
        
        Args:
            anomaly_id: ID of the anomaly to retrieve
            
        Returns:
            Anomaly record
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        if self.sqlite_db:
            # Use SQLite database
            try:
                conn = self.sqlite_db.connect()
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM anomalies WHERE id = ?', (anomaly_id,))
                anomaly = cursor.fetchone()
                
                if anomaly is None:
                    raise ValueError("Anomaly not found")
                
                return dict(anomaly)
                
            except Exception as e:
                logger.error(f"Failed to get anomaly {anomaly_id} from SQLite: {e}")
                traceback.print_exc()
                raise ValueError("Failed to retrieve anomaly.")
            finally:
                self.sqlite_db.close()
        
        if not SUPABASE_AVAILABLE:
            # Mock implementation (should not reach here with SQLite)
            raise ValueError("Anomaly not found")
        
        try:
            response = self.supabase.table("anomalies").select("*").eq("id", anomaly_id).single().execute()
            
            if response.data is None:
                raise ValueError("Anomaly not found")
                
            return response.data
            
        except Exception as e:
            logger.error(f"Failed to get anomaly {anomaly_id}: {e}")
            traceback.print_exc()
            raise ValueError("Failed to retrieve anomaly.")
    
    def review_anomaly(self, anomaly_id: str, reviewed: bool) -> Dict[str, str]:
        """
        Mark anomaly as reviewed in the database.
        
        Args:
            anomaly_id: ID of the anomaly to update
            reviewed: Whether the anomaly has been reviewed
            
        Returns:
            Success message
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        if not SUPABASE_AVAILABLE:
            # Mock implementation
            return {"message": f"Anomaly {anomaly_id} marked as {'reviewed' if reviewed else 'unreviewed'}"}
        
        try:
            response = self.supabase.table("anomalies").update({"reviewed": reviewed}).eq("id", anomaly_id).execute()
            
            if not response.data:
                raise ValueError(f"Anomaly with id {anomaly_id} not found.")
            
            return {"message": f"Anomaly {anomaly_id} marked as {'reviewed' if reviewed else 'unreviewed'}"}
            
        except Exception as e:
            logger.error(f"Failed to review anomaly {anomaly_id}: {e}")
            traceback.print_exc()
            raise ValueError("Failed to update anomaly review status.")
    
    def report_anomaly(self, anomaly_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Report a new anomaly from a client and store it in the database.
        
        Args:
            anomaly_data: Anomaly data to insert
            
        Returns:
            Success message with anomaly ID
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        if not SUPABASE_AVAILABLE:
            # Mock implementation
            return {"message": "Anomaly reported successfully", "anomaly_id": "mock_id"}
        
        try:
            response = self.supabase.table("anomalies").insert(anomaly_data).execute()
            
            if not response.data:
                raise ValueError("Failed to insert anomaly into database.")
            
            return {"message": "Anomaly reported successfully", "anomaly_id": response.data[0]['id']}
            
        except Exception as e:
            logger.error(f"Failed to report anomaly: {e}")
            traceback.print_exc()
            raise ValueError("Failed to report anomaly.")


# Global database service instance
database_service = DatabaseService()
