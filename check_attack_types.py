import pandas as pd
import glob

# Check multiple files for different attack types
files = glob.glob('data/raw/CICIDS2017/MachineLearningCVE/*.csv')[:5]

for file in files:
    try:
        df = pd.read_csv(file, nrows=1000)
        unique_labels = df[' Label'].unique()
        if len(unique_labels) > 1:
            print(f'📁 {file.split("/")[-1]}:')
            print(f'  Labels: {unique_labels}')
            print()
    except:
        continue
