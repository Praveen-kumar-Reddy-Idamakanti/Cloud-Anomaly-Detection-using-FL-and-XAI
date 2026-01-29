import pandas as pd
import glob

# Check processed data for attack types
files = glob.glob('data_preprocessing/processed_data/processed_*.csv')

print('🎯 ATTACK TYPES IN PROCESSED DATA:')
for file in files[:5]:
    df = pd.read_csv(file, nrows=1000)
    if 'Attack_Category' in df.columns:
        unique_attacks = df['Attack_Category'].value_counts()
        print(f'📁 {file.split("/")[-1]}:')
        for attack, count in unique_attacks.items():
            print(f'  {attack}: {count}')
        print()

# Also check all columns in one file
print('📊 ALL COLUMNS IN PROCESSED DATA:')
first_file = files[0]
df = pd.read_csv(first_file, nrows=10)
print(f'File: {first_file.split("/")[-1]}')
print('Columns:')
for i, col in enumerate(df.columns):
    print(f'  {i+1:2d}. {col}')
