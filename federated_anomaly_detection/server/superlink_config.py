"""
Configuration for Flower SuperLink server.
This file contains the configuration for running the federated learning server
using Flower's new SuperLink architecture.
"""
import os
import sys
from typing import Dict, Any, Optional, List, Tuple

import flwr as fl
from flwr.server.strategy import Strategy
from flwr.server.client_manager import SimpleClientManager

# Import ServerConfig based on Flower version
try:
    # For newer versions of Flower
    from flwr.server import ServerConfig
except ImportError:
    # Fallback for older versions
    from flwr.server.server import ServerConfig

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from federated_anomaly_detection.server.strategy import SaveModelStrategy
from federated_anomaly_detection.models.autoencoder import create_model

def get_strategy(
    input_dim: int = 9,
    min_clients: int = 2,
    num_rounds: int = 10,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.5,
) -> Strategy:
    """Create and return a strategy for federated learning.
    
    Args:
        input_dim: Number of input features
        min_clients: Minimum number of clients required for training
        num_rounds: Total number of federated learning rounds
        fraction_fit: Fraction of clients used for training in each round
        fraction_evaluate: Fraction of clients used for evaluation in each round
        
    Returns:
        Configured strategy instance
    """
    # Create initial model
    device = 'cpu'  # SuperLink will handle device placement
    model, _, _, _ = create_model(
        input_dim=input_dim,
        learning_rate=0.001,
        device=device
    )
    
    # Get initial parameters
    init_params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    # Define strategy
    return SaveModelStrategy(
        input_dim=input_dim,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_clients,
        min_evaluate_clients=max(1, min_clients // 2),
        min_available_clients=min_clients,
        on_fit_config_fn=lambda server_round: {
            "epochs": 5,
            "batch_size": 32,
            "server_round": server_round,
        },
        initial_parameters=fl.common.ndarrays_to_parameters(init_params),
    )

def get_superlink_config() -> Dict[str, Any]:
    """Return configuration for the SuperLink server."""
    return {
        "address": "0.0.0.0:8080",
        "ssl_keyfile": None,  # Path to SSL key file if using SSL
        "ssl_certfile": None,  # Path to SSL certificate if using SSL
        "ssl_ca_certfile": None,  # Path to CA certificate if using SSL
        "min_available_clients": 2,  # Minimum number of clients to start training
        "keep_alive_period_seconds": 60,  # Timeout for client connections
    }

def get_server_config() -> ServerConfig:
    """Return configuration for the Flower server."""
    return ServerConfig(
        num_rounds=10,  # Will be overridden by command line argument if provided
        round_timeout=600.0,  # Timeout for each round in seconds
    )

def start_server() -> None:
    """Start the Flower server."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Start Flower server")
    parser.add_argument(
        "--input-dim",
        type=int,
        default=9,
        help="Number of features in the input data",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for training",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of federated learning rounds",
    )
    parser.add_argument(
        "--address",
        type=str,
        default="localhost:8080",
        help="Server address (default: localhost:8080)",
    )
    
    args = parser.parse_args()
    
    # Configure strategy
    strategy = get_strategy(
        input_dim=args.input_dim,
        min_clients=args.min_clients,
        num_rounds=args.num_rounds,
    )
    
    # Configure server
    config = get_server_config()
    config.num_rounds = args.num_rounds
    
    print("=" * 80)
    print("Starting Flower server with configuration:")
    print(f"- Address: {args.address}")
    print(f"- Input dimension: {args.input_dim}")
    print(f"- Minimum clients: {args.min_clients}")
    print(f"- Number of rounds: {args.num_rounds}")
    print("=" * 80 + "\n")
    
    # Start server
    fl.server.start_server(
        server_address=args.address,
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    start_server()
