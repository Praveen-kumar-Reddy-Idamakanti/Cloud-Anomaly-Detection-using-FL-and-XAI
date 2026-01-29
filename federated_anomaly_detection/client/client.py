import argparse
import sys
import os
import traceback
from typing import Dict, Optional

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import flwr as fl
from flwr.common import Parameters, Scalar
from federated_anomaly_detection.client.anomaly_client import AnomalyDetectionClient

def main():
    parser = argparse.ArgumentParser(description="Federated Anomaly Detection Client")
    parser.add_argument(
        "--node_id", 
        type=int, 
        required=True, 
        help="Unique ID for this client node"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/node_{node_id}.npz",
        help="Path to the node's data file (supports {node_id} formatting)",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address (default: 0.0.0.0:8080)",
    )
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Resume from latest checkpoint"
    )
    
    args = parser.parse_args()
    
    try:
        # Format data path with node_id if needed
        if "{node_id}" in args.data_path:
            args.data_path = args.data_path.format(node_id=args.node_id)

        # Create data directory if it doesn't exist
        data_dir = os.path.dirname(args.data_path)
        if data_dir:  # Only create directory if path contains a directory
            os.makedirs(data_dir, exist_ok=True)

        print(f"Starting client {args.node_id} connecting to {args.server_address}")
        print(f"Using data from: {os.path.abspath(args.data_path)}")
        
        # Initialize and start the client
        client = AnomalyDetectionClient(node_id=args.node_id, data_path=args.data_path)
        
        # Start the client using start_numpy_client
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client,
        )
        
    except KeyboardInterrupt:
        print("\nClient interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError in client {args.node_id}:")
        print(traceback.format_exc())
        print("\nClient failed. Make sure the server is running and accessible.")
        sys.exit(1)

if __name__ == "__main__":
    main()
