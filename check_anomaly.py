import sqlite3
import json

conn = sqlite3.connect('data/anomaly_detection.db')
cursor = conn.cursor()

cursor.execute('SELECT id, features FROM anomalies WHERE id = ?', ('real_Unknown_212339',))
result = cursor.fetchone()

print('Anomaly features:')
if result:
    print(f'ID: {result[0]}')
    print(f'Features type: {type(result[1])}')
    print(f'Features length: {len(result[1]) if result[1] else 0}')
    print(f'Features preview: {result[1][:100] if result[1] else None}')
    
    # Try to parse features
    try:
        features = json.loads(result[1])
        print(f'Parsed features type: {type(features)}')
        print(f'Parsed features length: {len(features)}')
        print(f'First 5 features: {features[:5]}')
    except Exception as e:
        print(f'Error parsing features: {e}')

conn.close()
