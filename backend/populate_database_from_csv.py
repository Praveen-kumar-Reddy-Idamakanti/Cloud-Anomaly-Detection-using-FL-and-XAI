"""
Populate SQLite database from real anomalies CSV file.
"""

import pandas as pd
import sqlite3
import json
import random
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from database.sqlite_setup import SQLiteSetup

class DatabasePopulatorFromCSV:
    """Populate database with real anomalies from CSV file."""
    
    def __init__(self):
        self.db = SQLiteSetup()
        self.csv_file = project_root / "data" / "real_anomalies_consolidated.csv"
        
    def load_anomalies_from_csv(self):
        """Load anomalies from the consolidated CSV file."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")
        
        print(f"📁 Loading anomalies from: {self.csv_file}")
        df = pd.read_csv(self.csv_file)
        print(f"📊 Loaded {len(df)} anomalies")
        print(f"📈 Attack distribution:")
        print(df['original_attack_type'].value_counts())
        
        return df
    
    def convert_csv_row_to_db_record(self, row, index):
        """Convert CSV row to database record format."""
        # Parse features from string
        try:
            features = json.loads(row['features'])
            if isinstance(features, list):
                features = [float(x) for x in features]
            else:
                features = [0.0] * 78
        except:
            features = [0.0] * 78
        
        # Ensure exactly 78 features
        if len(features) > 78:
            features = features[:78]
        elif len(features) < 78:
            features.extend([0.0] * (78 - len(features)))
        
        # Use existing values from CSV
        return {
            'id': row['id'],
            'timestamp': row['timestamp'],
            'severity': row['severity'],
            'source_ip': row['source_ip'],
            'destination_ip': row['destination_ip'],
            'protocol': row['protocol'],
            'action': row['action'],
            'confidence': float(row['confidence']),
            'reviewed': False,  # Reset reviewed status
            'details': f"Real {row['original_attack_type']} attack from {row['source_file']}",
            'features': json.dumps(features),
            'anomaly_score': float(row['anomaly_score']),
            'attack_type_id': int(row['attack_type_id']),
            'attack_confidence': float(row['attack_confidence'])
        }
    
    def populate_database(self, max_anomalies=1000):
        """Populate database with anomalies from CSV."""
        try:
            # Load anomalies from CSV
            df = self.load_anomalies_from_csv()
            
            # Sample anomalies for performance (optional)
            if len(df) > max_anomalies:
                print(f"📊 Sampling {max_anomalies} anomalies from {len(df)} total")
                df = df.sample(n=max_anomalies, random_state=42)
            
            # Convert to database records
            print("🔄 Converting to database records...")
            anomaly_records = []
            for index, (_, row) in enumerate(df.iterrows()):
                record = self.convert_csv_row_to_db_record(row, index)
                anomaly_records.append(record)
            
            # Insert into database
            conn = self.db.connect()
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("DELETE FROM anomalies")
            print("🗑️  Cleared existing anomaly records")
            
            # Insert new records
            print("💾 Inserting new anomaly records...")
            for record in anomaly_records:
                columns = ', '.join(record.keys())
                placeholders = ', '.join(['?' for _ in record.keys()])
                cursor.execute(f'INSERT INTO anomalies ({columns}) VALUES ({placeholders})', 
                             tuple(record.values()))
            
            # Update system stats
            total_anomalies = len(anomaly_records)
            critical_count = sum(1 for r in anomaly_records if r['severity'] == 'critical')
            high_count = sum(1 for r in anomaly_records if r['severity'] == 'high')
            medium_count = sum(1 for r in anomaly_records if r['severity'] == 'medium')
            low_count = sum(1 for r in anomaly_records if r['severity'] == 'low')
            
            avg_confidence = sum(r['confidence'] for r in anomaly_records) / total_anomalies
            
            cursor.execute('''
                UPDATE system_stats 
                SET total_anomalies = ?,
                    critical_anomalies = ?,
                    high_anomalies = ?,
                    medium_anomalies = ?,
                    low_anomalies = ?,
                    avg_confidence = ?,
                    alert_rate = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE id = 1
            ''', (total_anomalies, critical_count, high_count, medium_count, low_count,
                  round(avg_confidence, 3), round((total_anomalies / 1000) * 100, 2)))
            
            conn.commit()
            conn.close()
            
            print(f"\n✅ Successfully populated database with {total_anomalies} real anomalies!")
            print(f"   - Critical: {critical_count}")
            print(f"   - High: {high_count}")
            print(f"   - Medium: {medium_count}")
            print(f"   - Low: {low_count}")
            print(f"   - Avg Confidence: {avg_confidence:.3f}")
            
            # Show attack type distribution
            attack_dist = df['original_attack_type'].value_counts()
            print(f"\n🎯 Attack Type Distribution:")
            for attack_type, count in attack_dist.items():
                print(f"   - {attack_type}: {count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error populating database: {e}")
            return False

def main():
    """Main function to populate database from CSV."""
    print("🚀 Populating SQLite Database from Real Anomalies CSV")
    print("=" * 60)
    
    populator = DatabasePopulatorFromCSV()
    
    # Check if CSV exists
    if not populator.csv_file.exists():
        print(f"❌ CSV file not found: {populator.csv_file}")
        print("📋 Please run extract_real_anomalies.py first!")
        return 1
    
    # Ask user for number of anomalies
    try:
        max_anomalies = int(input("📊 How many anomalies to populate (recommended: 500-2000): ") or "1000")
        max_anomalies = min(max_anomalies, 10000)  # Cap at 10k for performance
        max_anomalies = max(max_anomalies, 100)   # Minimum 100
    except:
        max_anomalies = 1000
    
    print(f"🎯 Populating database with {max_anomalies} real anomalies...")
    
    success = populator.populate_database(max_anomalies)
    
    if success:
        print("\n🎉 Database populated with real CICIDS2017 anomaly data!")
        print("\n📋 Next steps:")
        print("   1. Restart the backend server")
        print("   2. Check the frontend Dashboard")
        print("   3. Click on anomalies for XAI explanations")
        print("   4. Real network features are now ready!")
        print("\n🔬 This data contains:")
        print("   - Real network traffic features from CICIDS2017")
        print("   - Actual attack patterns (DDoS, PortScan, Web Attacks, etc.)")
        print("   - Legitimate research dataset for academic demonstration")
        return 0
    else:
        print("\n❌ Failed to populate database")
        return 1

if __name__ == "__main__":
    exit(main())
