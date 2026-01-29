"""
SQLite database service for handling database operations.
"""

import logging
import traceback
import sqlite3
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# Import SQLite setup
from database.sqlite_setup import SQLiteSetup

logger = logging.getLogger(__name__)

class SQLiteDatabaseService:
    """Service class for SQLite database operations."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'anomaly_detection.db')
        self.conn = None
        self._ensure_database_directory()
        self._initialize_database()
    
    def _ensure_database_directory(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Database directory ensured: {db_dir}")
    
    def _initialize_database(self):
        """Initialize the database if it doesn't exist."""
        if not os.path.exists(self.db_path):
            logger.info("Database not found, initializing...")
            setup = SQLiteSetup(self.db_path)
            setup.initialize_database()
            logger.info("Database initialized successfully")
    
    def connect(self) -> sqlite3.Connection:
        """Connect to the SQLite database."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable dictionary-like row access
            logger.info(f"Connected to SQLite database: {self.db_path}")
        return self.conn
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def is_connected(self) -> bool:
        """Check if database connection is available."""
        try:
            self.connect()
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics from the database.
        
        Returns:
            Dictionary containing system statistics
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Get total logs count
            cursor.execute("SELECT COUNT(*) as count FROM logs")
            total_logs = cursor.fetchone()['count']
            
            # Get anomaly counts
            cursor.execute("SELECT severity, COUNT(*) as count FROM anomalies GROUP BY severity")
            anomaly_results = cursor.fetchall()
            
            anomaly_counts = {row['severity']: row['count'] for row in anomaly_results}
            total_anomalies = sum(anomaly_counts.values())
            
            critical_anomalies = anomaly_counts.get('critical', 0)
            high_anomalies = anomaly_counts.get('high', 0)
            medium_anomalies = anomaly_counts.get('medium', 0)
            low_anomalies = anomaly_counts.get('low', 0)
            
            # Get average confidence
            cursor.execute("SELECT AVG(confidence) as avg_conf FROM anomalies")
            avg_confidence_result = cursor.fetchone()
            avg_confidence = avg_confidence_result['avg_conf'] or 0.0
            
            # Calculate alert rate
            alert_rate = round((total_anomalies / total_logs) * 100, 2) if total_logs > 0 else 0.0
            
            # Update system stats table
            cursor.execute('''
                UPDATE system_stats 
                SET total_logs = ?, total_anomalies = ?, critical_anomalies = ?, 
                    high_anomalies = ?, medium_anomalies = ?, low_anomalies = ?,
                    avg_confidence = ?, alert_rate = ?, last_updated = CURRENT_TIMESTAMP
                WHERE id = 1
            ''', (total_logs, total_anomalies, critical_anomalies, high_anomalies, 
                  medium_anomalies, low_anomalies, avg_confidence, alert_rate))
            
            conn.commit()
            
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
            logger.error(f"Failed to get stats: {e}")
            traceback.print_exc()
            raise ValueError("Failed to retrieve system statistics.")
        finally:
            self.close()
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get historical training metrics from the database.
        
        Returns:
            List of training history entries
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT server_round, avg_loss, std_loss, avg_accuracy, created_at
                FROM training_runs 
                ORDER BY server_round ASC
            ''')
            
            results = cursor.fetchall()
            
            return [
                {
                    "server_round": row['server_round'],
                    "avg_loss": row['avg_loss'],
                    "std_loss": row['std_loss'],
                    "avg_accuracy": row['avg_accuracy'],
                    "created_at": row['created_at']
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get training history: {e}")
            traceback.print_exc()
            raise ValueError("Failed to retrieve training history.")
        finally:
            self.close()
    
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
        
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            start_index = (page - 1) * limit
            
            cursor.execute('''
                SELECT id, timestamp, severity, source_ip, destination_ip, protocol, 
                       action, confidence, reviewed, details, anomaly_score, 
                       attack_type_id, attack_confidence, created_at, updated_at
                FROM anomalies 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (limit, start_index))
            
            results = cursor.fetchall()
            
            return [
                {
                    "id": row['id'],
                    "timestamp": row['timestamp'],
                    "severity": row['severity'],
                    "source_ip": row['source_ip'],
                    "destination_ip": row['destination_ip'],
                    "protocol": row['protocol'],
                    "action": row['action'],
                    "confidence": row['confidence'],
                    "reviewed": bool(row['reviewed']),
                    "details": row['details'],
                    "anomaly_score": row['anomaly_score'],
                    "attack_type_id": row['attack_type_id'],
                    "attack_confidence": row['attack_confidence'],
                    "created_at": row['created_at'],
                    "updated_at": row['updated_at']
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get anomalies: {e}")
            traceback.print_exc()
            raise ValueError("Failed to retrieve anomalies.")
        finally:
            self.close()
    
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
        
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, timestamp, severity, source_ip, destination_ip, protocol, 
                       action, confidence, reviewed, details, features, anomaly_score, 
                       attack_type_id, attack_confidence, created_at, updated_at
                FROM anomalies 
                WHERE id = ?
            ''', (anomaly_id,))
            
            result = cursor.fetchone()
            
            if result is None:
                raise ValueError("Anomaly not found")
            
            return {
                "id": result['id'],
                "timestamp": result['timestamp'],
                "severity": result['severity'],
                "source_ip": result['source_ip'],
                "destination_ip": result['destination_ip'],
                "protocol": result['protocol'],
                "action": result['action'],
                "confidence": result['confidence'],
                "reviewed": bool(result['reviewed']),
                "details": result['details'],
                "features": json.loads(result['features']) if result['features'] else None,
                "anomaly_score": result['anomaly_score'],
                "attack_type_id": result['attack_type_id'],
                "attack_confidence": result['attack_confidence'],
                "created_at": result['created_at'],
                "updated_at": result['updated_at']
            }
            
        except Exception as e:
            logger.error(f"Failed to get anomaly {anomaly_id}: {e}")
            traceback.print_exc()
            raise ValueError("Failed to retrieve anomaly.")
        finally:
            self.close()
    
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
        
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE anomalies 
                SET reviewed = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (reviewed, anomaly_id))
            
            if cursor.rowcount == 0:
                raise ValueError(f"Anomaly with id {anomaly_id} not found.")
            
            conn.commit()
            
            return {"message": f"Anomaly {anomaly_id} marked as {'reviewed' if reviewed else 'unreviewed'}"}
            
        except Exception as e:
            logger.error(f"Failed to review anomaly {anomaly_id}: {e}")
            traceback.print_exc()
            raise ValueError("Failed to update anomaly review status.")
        finally:
            self.close()
    
    def report_anomaly(self, anomaly_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Report a new anomaly and store it in the database.
        
        Args:
            anomaly_data: Anomaly data to insert
            
        Returns:
            Success message with anomaly ID
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Generate ID if not provided
            if 'id' not in anomaly_data:
                anomaly_data['id'] = f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(anomaly_data)}"
            
            # Add timestamp if not provided
            if 'timestamp' not in anomaly_data:
                anomaly_data['timestamp'] = datetime.now().isoformat()
            
            # Convert features to JSON string if provided
            if 'features' in anomaly_data and isinstance(anomaly_data['features'], list):
                anomaly_data['features'] = json.dumps(anomaly_data['features'])
            
            # Prepare columns and values
            columns = ', '.join(anomaly_data.keys())
            placeholders = ', '.join(['?' for _ in anomaly_data.keys()])
            
            cursor.execute(f'INSERT INTO anomalies ({columns}) VALUES ({placeholders})', tuple(anomaly_data.values()))
            
            conn.commit()
            
            return {"message": "Anomaly reported successfully", "anomaly_id": anomaly_data['id']}
            
        except Exception as e:
            logger.error(f"Failed to report anomaly: {e}")
            traceback.print_exc()
            raise ValueError("Failed to report anomaly.")
        finally:
            self.close()
    
    def get_logs(self, page: int = 1, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get paginated list of logs from the database.
        
        Args:
            page: Page number (1-based)
            limit: Number of items per page
            
        Returns:
            List of log records
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            start_index = (page - 1) * limit
            
            cursor.execute('''
                SELECT id, timestamp, source_ip, destination_ip, protocol, encrypted, 
                       size, features, anomaly_score, created_at, updated_at
                FROM logs 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (limit, start_index))
            
            results = cursor.fetchall()
            
            return [
                {
                    "id": row['id'],
                    "timestamp": row['timestamp'],
                    "source_ip": row['source_ip'],
                    "destination_ip": row['destination_ip'],
                    "protocol": row['protocol'],
                    "encrypted": bool(row['encrypted']),
                    "size": row['size'],
                    "features": json.loads(row['features']) if row['features'] else None,
                    "anomaly_score": row['anomaly_score'],
                    "created_at": row['created_at'],
                    "updated_at": row['updated_at']
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get logs: {e}")
            traceback.print_exc()
            raise ValueError("Failed to retrieve logs.")
        finally:
            self.close()
    
    def upload_log(self, log_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Upload a new log entry to the database.
        
        Args:
            log_data: Log data to insert
            
        Returns:
            Success message with log ID
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Generate ID if not provided
            if 'id' not in log_data:
                log_data['id'] = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(log_data)}"
            
            # Add timestamp if not provided
            if 'timestamp' not in log_data:
                log_data['timestamp'] = datetime.now().isoformat()
            
            # Convert features to JSON string if provided
            if 'features' in log_data and isinstance(log_data['features'], list):
                log_data['features'] = json.dumps(log_data['features'])
            
            # Prepare columns and values
            columns = ', '.join(log_data.keys())
            placeholders = ', '.join(['?' for _ in log_data.keys()])
            
            cursor.execute(f'INSERT INTO logs ({columns}) VALUES ({placeholders})', tuple(log_data.values()))
            
            conn.commit()
            
            return {"message": "Log uploaded successfully", "log_id": log_data['id']}
            
        except Exception as e:
            logger.error(f"Failed to upload log: {e}")
            traceback.print_exc()
            raise ValueError("Failed to upload log.")
        finally:
            self.close()
    
    def add_training_run(self, training_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Add a new training run record.
        
        Args:
            training_data: Training run data to insert
            
        Returns:
            Success message with training run ID
        """
        if not self.is_connected():
            raise ValueError("Database connection not available")
        
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Prepare columns and values
            columns = ', '.join(training_data.keys())
            placeholders = ', '.join(['?' for _ in training_data.keys()])
            
            cursor.execute(f'INSERT INTO training_runs ({columns}) VALUES ({placeholders})', tuple(training_data.values()))
            
            conn.commit()
            
            return {"message": "Training run recorded successfully", "run_id": str(cursor.lastrowid)}
            
        except Exception as e:
            logger.error(f"Failed to add training run: {e}")
            traceback.print_exc()
            raise ValueError("Failed to add training run.")
        finally:
            self.close()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            
            # Get table counts
            tables_info = {}
            tables = ['anomalies', 'training_runs', 'logs', 'users', 'system_stats']
            
            for table in tables:
                cursor.execute(f'SELECT COUNT(*) as count FROM {table}')
                tables_info[table] = cursor.fetchone()['count']
            
            # Get database file size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'database_path': self.db_path,
                'database_size_mb': round(db_size / (1024 * 1024), 2),
                'tables': tables_info,
                'connection_status': 'connected'
            }
            
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {'error': str(e)}
        finally:
            self.close()


# Global SQLite database service instance
sqlite_database_service = SQLiteDatabaseService()
