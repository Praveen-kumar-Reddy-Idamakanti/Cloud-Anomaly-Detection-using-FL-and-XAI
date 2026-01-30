"""
SQLite database setup and initialization script.
"""

import sqlite3
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import random
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path - use backend/database directory
try:
    from config.app_config import path_config
    DB_PATH = str(path_config.project_root / "backend" / "database" / "anomaly_detection.db")
except ImportError:
    # Fallback to local directory if config not available
    DB_PATH = os.path.join(os.path.dirname(__file__), 'anomaly_detection.db')

class SQLiteSetup:
    """SQLite database setup and management."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.conn = None
        self._ensure_database_directory()
    
    def _ensure_database_directory(self):
        """Ensure the database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Database directory ensured: {db_dir}")
    
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
    
    def initialize_database(self):
        """Initialize the database with required tables."""
        logger.info("Initializing SQLite database...")
        
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            # Create tables
            self._create_anomalies_table(cursor)
            self._create_training_runs_table(cursor)
            self._create_logs_table(cursor)
            self._create_users_table(cursor)
            self._create_system_stats_table(cursor)
            
            # Insert initial data
            self._insert_initial_data(cursor)
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            conn.rollback()
            raise
        finally:
            self.close()
    
    def _create_anomalies_table(self, cursor: sqlite3.Cursor):
        """Create the anomalies table."""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low')),
                source_ip TEXT NOT NULL,
                destination_ip TEXT NOT NULL,
                protocol TEXT NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
                reviewed BOOLEAN DEFAULT FALSE,
                details TEXT,
                features TEXT,  -- JSON string of features
                anomaly_score REAL,
                attack_type_id INTEGER,
                attack_confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_severity ON anomalies(severity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomalies_reviewed ON anomalies(reviewed)')
        logger.info("Created anomalies table")
    
    def _create_training_runs_table(self, cursor: sqlite3.Cursor):
        """Create the training runs table."""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_round INTEGER NOT NULL,
                avg_loss REAL,
                std_loss REAL,
                avg_accuracy REAL,
                min_loss REAL,
                max_loss REAL,
                total_samples INTEGER,
                duration_seconds INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_runs_round ON training_runs(server_round)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_runs_created ON training_runs(created_at)')
        logger.info("Created training_runs table")
    
    def _create_logs_table(self, cursor: sqlite3.Cursor):
        """Create the logs table."""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                source_ip TEXT NOT NULL,
                destination_ip TEXT NOT NULL,
                protocol TEXT NOT NULL,
                encrypted BOOLEAN DEFAULT FALSE,
                size INTEGER NOT NULL,
                features TEXT,  -- JSON string of features
                anomaly_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_protocol ON logs(protocol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_encrypted ON logs(encrypted)')
        logger.info("Created logs table")
    
    def _create_users_table(self, cursor: sqlite3.Cursor):
        """Create the users table."""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user' CHECK (role IN ('admin', 'user', 'analyst')),
                is_active BOOLEAN DEFAULT TRUE,
                last_login DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)')
        logger.info("Created users table")
    
    def _create_system_stats_table(self, cursor: sqlite3.Cursor):
        """Create the system statistics table."""
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY DEFAULT 1,
                total_logs INTEGER DEFAULT 0,
                total_anomalies INTEGER DEFAULT 0,
                critical_anomalies INTEGER DEFAULT 0,
                high_anomalies INTEGER DEFAULT 0,
                medium_anomalies INTEGER DEFAULT 0,
                low_anomalies INTEGER DEFAULT 0,
                avg_confidence REAL DEFAULT 0.0,
                alert_rate REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('INSERT OR REPLACE INTO system_stats (id) VALUES (1)')
        logger.info("Created system_stats table")
    
    def _insert_initial_data(self, cursor: sqlite3.Cursor):
        """Insert initial data into the database."""
        # Insert sample anomalies
        sample_anomalies = [
            {
                'id': 'anomaly_001',
                'timestamp': datetime.now().isoformat(),
                'severity': 'high',
                'source_ip': '192.168.1.100',
                'destination_ip': '10.0.0.1',
                'protocol': 'TCP',
                'action': 'block',
                'confidence': 0.85,
                'reviewed': False,
                'details': 'Suspicious network traffic detected',
                'features': json.dumps([0.1, 0.2, 0.3, 0.4, 0.5] + [random.random() for _ in range(73)]),
                'anomaly_score': 0.75,
                'attack_type_id': 2,
                'attack_confidence': 0.92
            },
            {
                'id': 'anomaly_002',
                'timestamp': datetime.now().isoformat(),
                'severity': 'medium',
                'source_ip': '192.168.1.101',
                'destination_ip': '10.0.0.2',
                'protocol': 'UDP',
                'action': 'alert',
                'confidence': 0.72,
                'reviewed': False,
                'details': 'Unusual UDP traffic pattern',
                'features': json.dumps([0.2, 0.3, 0.1, 0.6, 0.4] + [random.random() for _ in range(73)]),
                'anomaly_score': 0.65,
                'attack_type_id': 3,
                'attack_confidence': 0.78
            },
            {
                'id': 'anomaly_003',
                'timestamp': datetime.now().isoformat(),
                'severity': 'critical',
                'source_ip': '192.168.1.102',
                'destination_ip': '10.0.0.3',
                'protocol': 'TCP',
                'action': 'block',
                'confidence': 0.95,
                'reviewed': True,
                'details': 'Critical security threat detected',
                'features': json.dumps([0.8, 0.9, 0.7, 0.2, 0.1] + [random.random() for _ in range(73)]),
                'anomaly_score': 0.92,
                'attack_type_id': 1,
                'attack_confidence': 0.98
            }
        ]
        
        # Add more diverse sample anomalies for demonstration
        additional_anomalies = [
            {
                'id': f'anomaly_{i:04d}',
                'timestamp': (datetime.now() - timedelta(minutes=random.randint(1, 1440))).isoformat(),
                'severity': random.choice(['critical', 'high', 'medium', 'low']),
                'source_ip': f'192.168.1.{random.randint(100, 200)}',
                'destination_ip': f'10.0.0.{random.randint(1, 50)}',
                'protocol': random.choice(['TCP', 'UDP', 'HTTP', 'HTTPS', 'DNS']),
                'action': random.choice(['block', 'alert', 'monitor', 'allow']),
                'confidence': round(random.uniform(0.6, 0.98), 2),
                'reviewed': random.choice([True, False]),
                'details': random.choice([
                    'Suspicious network traffic pattern detected',
                    'Potential DDoS attack identified',
                    'Unusual port scanning activity',
                    'Abnormal data transfer volume',
                    'Possible malware communication',
                    'Unauthorized access attempt',
                    'Data exfiltration suspected',
                    'Network reconnaissance activity'
                ]),
                'features': json.dumps([random.random() for _ in range(78)]),
                'anomaly_score': round(random.uniform(0.3, 0.95), 3),
                'attack_type_id': random.randint(0, 4),
                'attack_confidence': round(random.uniform(0.7, 0.99), 2) if random.random() > 0.3 else None
            }
            for i in range(4, 103)  # Add 100 more anomalies (total 103)
        ]
        
        all_anomalies = sample_anomalies + additional_anomalies
        
        for anomaly in all_anomalies:
            columns = ', '.join(anomaly.keys())
            placeholders = ', '.join(['?' for _ in anomaly.keys()])
            cursor.execute(f'INSERT OR IGNORE INTO anomalies ({columns}) VALUES ({placeholders})', tuple(anomaly.values()))
        
        # Insert sample training run
        cursor.execute('''
            INSERT OR IGNORE INTO training_runs (server_round, avg_loss, std_loss, avg_accuracy, total_samples, duration_seconds)
            VALUES (1, 0.45, 0.12, 0.87, 1000, 300)
        ''')
        
        # Insert sample logs
        sample_logs = [
            {
                'id': 'log_001',
                'timestamp': datetime.now().isoformat(),
                'source_ip': '192.168.1.200',
                'destination_ip': '10.0.0.10',
                'protocol': 'HTTP',
                'encrypted': False,
                'size': 1024,
                'features': json.dumps([0.3, 0.4, 0.2, 0.7, 0.1]),
                'anomaly_score': 0.45
            },
            {
                'id': 'log_002',
                'timestamp': datetime.now().isoformat(),
                'source_ip': '192.168.1.201',
                'destination_ip': '10.0.0.11',
                'protocol': 'HTTPS',
                'encrypted': True,
                'size': 2048,
                'features': json.dumps([0.5, 0.6, 0.4, 0.3, 0.2]),
                'anomaly_score': 0.35
            }
        ]
        
        for log in sample_logs:
            columns = ', '.join(log.keys())
            placeholders = ', '.join(['?' for _ in log.keys()])
            cursor.execute(f'INSERT OR IGNORE INTO logs ({columns}) VALUES ({placeholders})', tuple(log.values()))
        
        # Update system stats
        cursor.execute('''
            UPDATE system_stats 
            SET total_anomalies = 103,
                high_anomalies = 25,
                medium_anomalies = 26,
                critical_anomalies = 26,
                low_anomalies = 26,
                avg_confidence = 0.82,
                alert_rate = 10.3,
                last_updated = CURRENT_TIMESTAMP
            WHERE id = 1
        ''')
        
        logger.info("Inserted initial sample data")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database information and statistics."""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
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
    
    def backup_database(self, backup_path: str = None) -> str:
        """Create a backup of the database."""
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Use path configuration for backup directory
            try:
                from config.app_config import path_config
                backup_dir = path_config.project_root / "backend" / "database" / "backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / f'backup_{timestamp}.db'
            except ImportError:
                # Fallback to local directory if config not available
                backup_path = os.path.join(os.path.dirname(self.db_path), f'backup_{timestamp}.db')
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            raise
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup."""
        try:
            import shutil
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"Database restored from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error restoring database: {e}")
            return False

def main():
    """Main function to initialize the database."""
    print("🚀 Initializing SQLite Database for Anomaly Detection System")
    print("=" * 50)
    
    setup = SQLiteSetup()
    
    try:
        # Initialize database
        setup.initialize_database()
        
        # Get database info
        info = setup.get_database_info()
        
        print("\n📊 Database Information:")
        print(f"   Path: {info.get('database_path', 'N/A')}")
        print(f"   Size: {info.get('database_size_mb', 0)} MB")
        print(f"   Tables: {info.get('tables', {})}")
        print(f"   Status: {info.get('connection_status', 'unknown')}")
        
        print("\n✅ Database initialized successfully!")
        print("\n📝 Tables created:")
        print("   - anomalies: Anomaly detection records")
        print("   - training_runs: Federated learning training data")
        print("   - logs: Network traffic logs")
        print("   - users: User management")
        print("   - system_stats: System statistics")
        
        print("\n🔍 Sample data inserted:")
        print("   - 103 sample anomalies (diverse types and severities)")
        print("   - 1 training run")
        print("   - 2 network logs")
        print("   - System statistics updated")
        
        print("\n📁 Database file location:")
        print(f"   {setup.db_path}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
