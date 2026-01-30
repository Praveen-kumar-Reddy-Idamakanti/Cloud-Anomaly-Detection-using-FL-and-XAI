import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_cloud_activity_data(n_samples=1000, n_features=20, anomaly_ratio=0.05, random_state=42):
    """
    Generate synthetic cloud activity data with some anomalies.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features in the dataset
        anomaly_ratio: Ratio of anomalies in the data
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (data, labels) where labels are 0 for normal and 1 for anomaly
    """
    np.random.seed(random_state)
    
    # Generate normal data (multivariate normal distribution)
    normal_data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Create anomalies by adding outliers to some samples
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Make anomalies by shifting some features significantly
    for idx in anomaly_indices:
        # Select random features to modify
        n_affected = np.random.randint(1, n_features//2 + 1)
        affected_features = np.random.choice(n_features, n_affected, replace=False)
        
        # Add significant deviation to selected features
        normal_data[idx, affected_features] += np.random.uniform(3, 10, n_affected)
    
    # Create labels (0: normal, 1: anomaly)
    labels = np.zeros(n_samples, dtype=int)
    labels[anomaly_indices] = 1
    
    # Scale data to [0, 1]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(normal_data)
    
    return scaled_data, labels

def save_node_data(node_id, data_dir='data', n_samples=1000, n_features=20):
    """
    Generate and save synthetic data for a specific node
    
    Args:
        node_id: ID of the node/client
        data_dir: Directory to save the data
        n_samples: Number of samples to generate
        n_features: Number of features in the dataset
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data with different random seeds for different nodes
    data, labels = generate_cloud_activity_data(
        n_samples=n_samples,
        n_features=n_features,
        random_state=42 + node_id  # Different seed for each node
    )
    
    # Create a DataFrame and save to CSV
    feature_columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_columns)
    df['is_anomaly'] = labels
    
    # Save to CSV
    file_path = os.path.join(data_dir, f'node_{node_id}.csv')
    df.to_csv(file_path, index=False)
    print(f'Saved data for node {node_id} to {file_path}')

def load_node_data(node_id, data_dir='data'):
    """
    Load data for a specific node
    
    Args:
        node_id: ID of the node/client
        data_dir: Directory containing the data files
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    file_path = os.path.join(data_dir, f'node_{node_id}.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No data file found for node {node_id} at {file_path}")
    
    df = pd.read_csv(file_path)
    features = df.drop('is_anomaly', axis=1).values
    labels = df['is_anomaly'].values
    
    return features, labels


def load_network_data(node_id, data_dir='data', dataset='BETH'):
    """
    Load preprocessed network data for a specific node
    
    Args:
        node_id: ID of the node/client (or file identifier)
        data_dir: Base directory containing the data files
        dataset: Name of the dataset to load (e.g., 'BETH', 'CICIDS2017')
        
    Returns:
        Tuple of (features, labels) as numpy arrays
    """
    # Look for preprocessed .npz files first
    processed_dir = os.path.join(data_dir, 'processed')
    
    # Try exact match first
    npz_path = os.path.join(processed_dir, f"node_{node_id}.npz")
    
    if not os.path.exists(npz_path):
        # Try to find a matching file in the raw data directory
        raw_dir = os.path.join(data_dir, 'raw', dataset)
        
        # Look for files that might match this node
        matching_files = []
        for root, _, files in os.walk(raw_dir):
            for file in files:
                if file.endswith('.csv') and str(node_id) in file:
                    matching_files.append(os.path.join(root, file))
        
        if not matching_files:
            raise FileNotFoundError(
                f"No network data files found for node {node_id} in {raw_dir}"
            )
        
        # Use the first matching file
        raw_file = matching_files[0]
        print(f"Found matching raw file: {raw_file}")
        
        # Process the raw file on the fly
        from preprocess_network_data import NetworkDataPreprocessor
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessor = NetworkDataPreprocessor(
                raw_data_dir=os.path.dirname(raw_file),
                output_dir=temp_dir
            )
            features, labels = preprocessor.process_file(Path(raw_file))
            
            if features is None or labels is None:
                raise ValueError(f"Failed to process {raw_file}")
            
            # Save the processed data for future use
            os.makedirs(processed_dir, exist_ok=True)
            npz_path = os.path.join(processed_dir, f"node_{node_id}.npz")
            np.savez_compressed(npz_path, features=features, labels=labels)
            print(f"Saved processed data to {npz_path}")
    
    # Load the processed data
    with np.load(npz_path) as data:
        features = data['features']
        labels = data['labels']
    
    return features, labels
