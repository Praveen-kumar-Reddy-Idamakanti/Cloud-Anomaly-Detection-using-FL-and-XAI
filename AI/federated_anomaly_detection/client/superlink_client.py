"""
Client launcher for Flower SuperLink.
This script launches a client that connects to a SuperLink server.
"""
import argparse
import os
import sys
import warnings
from typing import Dict, Optional, Tuple, List, Any

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import flwr as fl
from flwr.common import Scalar, Parameters, NDArrays, ndarrays_to_parameters
import numpy as np
import torch

from federated_anomaly_detection.client.anomaly_client import AnomalyDetectionClient

def get_parameters() -> List[np.ndarray]:
    """Get model parameters as a list of NumPy ndarrays."""
    # Create a dummy client to get the model architecture
    client = AnomalyDetectionClient(
        node_id=0,  # Dummy ID
        data_path="data/processed/client_1.npz"  # Default path
    )
    
    # Get model parameters as numpy arrays
    return [val.cpu().numpy() for _, val in client.model.state_dict().items()]

def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration dictionary for each round."""
    return {
        "epochs": 5,  # Number of local epochs
        "batch_size": 32,
        "server_round": server_round,  # The current round of federated learning
    }

def start_client(
    client_id: int,
    server_address: str,
    data_dir: str,
) -> None:
    """Start a client that connects to the Flower server.
    
    Args:
        client_id: The ID of the client
        server_address: Address of the Flower server
        data_dir: Path to the client's data file
    """
    # Create client
    client = AnomalyDetectionClient(
        node_id=client_id,
        data_path=data_dir
    )
    
    # Start client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )

def main():
    """Parse command line arguments and start client."""
    parser = argparse.ArgumentParser(description="Flower SuperLink Client")
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        help="A unique identifier for this client",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default="0.0.0.0:8080",
        help="Address of the SuperLink server (default: 0.0.0.0:8080)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the client's data file (e.g., data/processed/client_1.npz)",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=9,
        help="Input dimension (kept for backward compatibility, not used)",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Starting Flower SuperLink client with ID: {args.client_id}")
    print(f"- Server address: {args.server_address}")
    print(f"- Data file: {args.data_dir}")
    print("=" * 80 + "\n")
    
    try:
        start_client(
            client_id=args.client_id,
            server_address=args.server_address,
            data_dir=args.data_dir,
        )
    except KeyboardInterrupt:
        print("\nClient stopped by user")
    except Exception as e:
        print(f"Error in client {args.client_id}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
