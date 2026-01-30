"""
Legacy server implementation for backward compatibility.
This file exists to maintain backward compatibility with existing scripts.
For new deployments, use superlink_config.py instead.
"""
import argparse
import warnings

# Show deprecation warning
warnings.warn(
    "The server.py script is deprecated. Please use superlink_config.py instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import the new implementation
from federated_anomaly_detection.server.superlink_config import start_server

def main():
    """Legacy main function that delegates to the new implementation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Federated Anomaly Detection Server (Legacy)")
    parser.add_argument(
        "--input_dim",
        type=int,
        default=9,
        help="Number of features in the input data",
    )
    parser.add_argument(
        "--min_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for training",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds",
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Server address (default: 0.0.0.0:8080)",
    )
    
    args = parser.parse_args()
    
    # Map legacy arguments to new implementation
    import sys
    sys.argv = [
        sys.argv[0],
        f"--input-dim={args.input_dim}",
        f"--min-clients={args.min_clients}",
        f"--num-rounds={args.num_rounds}",
        f"--address={args.server_address}",
    ]
    
    # Start the server using the new implementation
    start_server()
if __name__ == "__main__":
    main()
