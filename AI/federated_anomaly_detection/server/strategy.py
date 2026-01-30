import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import flwr as fl
import numpy as np
import torch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg

from federated_anomaly_detection.models.autoencoder import create_model
from federated_anomaly_detection.server.supabase_client import get_supabase_client


def get_device() -> torch.device:
    """Get the device to run the model on"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define strategy for federated learning
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        input_dim: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn=None,
        on_fit_config_fn=None,
        initial_parameters=None,
    ) -> None:
        # Store custom parameters
        self.input_dim = input_dim
        self.device = get_device()
        self.best_loss = float('inf')
        self.log_dir = f"logs/server/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.models_dir = "saved_models"
        try:
            self.supabase_client = get_supabase_client()
        except ValueError as e:
            print(f"Failed to initialize Supabase client: {e}")
            self.supabase_client = None

        # Create necessary directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        self.round_num = 0
        self.best_model_path = os.path.join(self.models_dir, "best_model.pth")
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'anomaly_ratio': [],
            'learning_rate': []
        }

        # Initialize model for server-side evaluation
        self.model, _, _, _ = create_model(input_dim=input_dim, learning_rate=0.001, device=self.device)

        # Initialize parent class with compatible parameters
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config_fn or self._get_fit_config,
            initial_parameters=initial_parameters,
            evaluate_metrics_aggregation_fn=self.aggregate_metrics,
        )
        
        # Store eval_fn separately since it's not a direct parameter in newer Flower versions
        self.eval_fn = eval_fn
    
    def _get_fit_config(self, server_round: int) -> Dict[str, Any]:
        """Return training configuration for each round."""
        return {
            "server_round": server_round,  # Send as integer
            "epochs": 5,  # Default number of epochs as integer
            "learning_rate": str(0.001 * (0.9 ** (server_round // 5))),  # Learning rate decay
        }

    
    def _get_eval_fn(self):
        """Return an evaluation function for server-side evaluation."""
        def evaluate(
            server_round: int, parameters: Parameters, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            # This is a placeholder - in a real implementation, you would evaluate
            # the model on a server-side validation set
            return 0.0, {}
        return evaluate
    
    def aggregate_metrics(self, metrics: List[Tuple[int, Dict[str, Scalar]]]) -> Dict[str, Scalar]:
        """Aggregate evaluation metrics to provide a comprehensive view of model performance."""
        if not metrics:
            print("Warning: No metrics to aggregate")
            return {}

        # Initialize dictionaries to hold all metric values from all clients
        all_metrics = {key: [] for _, m in metrics for key in m}
        num_examples_list = [num_examples for num_examples, _ in metrics]
        total_examples = sum(num_examples_list)

        # Collect all metric values
        for num_examples, client_metrics in metrics:
            for key, value in client_metrics.items():
                try:
                    float_val = float(value)
                    if np.isfinite(float_val):
                        all_metrics[key].append((num_examples, float_val))
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert metric {key} value '{value}' to float, skipping.")

        aggregated_metrics = {}

        # Calculate weighted average and standard deviation for each metric
        for name, values in all_metrics.items():
            if not values:
                continue

            # Weighted average
            weighted_sum = sum(num_examples * value for num_examples, value in values)
            avg_metric = weighted_sum / total_examples if total_examples > 0 else 0.0
            aggregated_metrics[f"avg_{name}"] = avg_metric

            # Standard deviation for loss metrics
            if "loss" in name:
                mean = aggregated_metrics[f"avg_{name}"]
                # Weighted variance
                weighted_variance = sum(num_examples * ((value - mean) ** 2) for num_examples, value in values) / total_examples
                std_dev = np.sqrt(weighted_variance)
                aggregated_metrics[f"std_{name}"] = std_dev

        # Update history
        for name in ['val_loss', 'train_loss', 'anomaly_ratio', 'learning_rate']:
            if f'avg_{name}' in aggregated_metrics:
                self.history.setdefault(name, []).append(aggregated_metrics[f'avg_{name}'])

        # Print summary of aggregated metrics
        print("\n" + "="*80)
        print(f"ROUND {self.round_num} - AGGREGATED METRICS:")
        print("-" * 80)
        
        metric_groups = {
            'Loss Metrics': ['train_loss', 'val_loss', 'reconstruction_loss'],
            'Anomaly Detection': ['anomaly_ratio', 'threshold'],
            'Classification': ['accuracy', 'precision', 'recall', 'f1_score'],
            'Training': ['learning_rate', 'epochs']
        }
        
        for group_name, metric_list in metric_groups.items():
            group_has_metrics = any(f'avg_{m}' in aggregated_metrics for m in metric_list)
            if group_has_metrics:
                print(f"{group_name}:")
                for m in metric_list:
                    if f'avg_{m}' in aggregated_metrics:
                        value = aggregated_metrics[f'avg_{m}']
                        print(f"  - avg_{m}: {value:.6f}", end="")
                        if f'std_{m}' in aggregated_metrics:
                            std_value = aggregated_metrics[f'std_{m}']
                            print(f" (std: {std_value:.6f})")
                        else:
                            print()
        
        print("="*80 + "\n")

        # Save history to file
        self._save_history()

        return aggregated_metrics
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store the model"""
        # Call parent's aggregate_fit to handle the weighted averaging
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            # Update model with new parameters
            params_dict = zip(self.model.state_dict().keys(), parameters_to_ndarrays(aggregated_parameters))
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.model.load_state_dict(state_dict, strict=True)
            
            # Save the model
            model_path = os.path.join(self.log_dir, f"model_round_{server_round}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'round': server_round,
                'metrics': aggregated_metrics
            }, model_path)
            print(f"Model saved to {model_path}")
            
            # Track best model
            val_loss = aggregated_metrics.get('avg_val_loss', float('inf'))
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                best_model_path = os.path.join(self.log_dir, "best_model.pth")
                torch.save(self.model.state_dict(), best_model_path)
            
            # Log detailed metrics
            self._log_metrics(server_round, results, aggregated_metrics)
            
        return aggregated_parameters, aggregated_metrics
    
    def _log_metrics(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], aggregated_metrics: Dict[str, Scalar]):
        """Log detailed metrics from training round."""
        try:
            # Prepare metrics for logging
            round_metrics = {
                'server_round': server_round,
                'timestamp': datetime.now().isoformat(),
                'aggregated_metrics': aggregated_metrics,
                'client_metrics': {}
            }
            
            # Add per-client metrics
            for client, fit_res in results:
                if fit_res.metrics:
                    round_metrics['client_metrics'][str(client.cid)] = {
                        'num_samples': fit_res.num_examples,
                        'metrics': fit_res.metrics
                    }
            
            # Save detailed metrics to file
            metrics_path = os.path.join(self.log_dir, f"round_{server_round}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(round_metrics, f, indent=2)
            
            # Helper function to safely format metric values
            def format_metric(value, default='N/A', precision=6):
                if value is None:
                    return default
                try:
                    # Convert to float if it's a string
                    num = float(value) if isinstance(value, str) else value
                    return f"{num:.{precision}f}"
                except (ValueError, TypeError):
                    return str(value) if value is not None else default
            
            # Print summary with safe formatting
            print(f"\nRound {server_round} Summary:")
            print(f"- Clients: {len(results)}")
            # Print all aggregated metrics
            for key, value in aggregated_metrics.items():
                print(f"- {key}: {format_metric(value)}")
            print(f"- Best Val Loss: {self.best_loss:.6f}")
            
            # Helper function to safely convert and append metric values
            def safe_append(history_key, metric_key, default=0.0):
                try:
                    value = aggregated_metrics.get(metric_key)
                    if value is None:
                        print(f"Warning: Metric {metric_key} not found in aggregated_metrics")
                        value = default
                    
                    # Convert to float, handling both string and numeric types
                    try:
                        value_float = float(value)
                        self.history[history_key].append(value_float)
                        return value_float
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not convert {metric_key} value '{value}' to float: {e}")
                        self.history[history_key].append(default)
                        return default
                except Exception as e:
                    print(f"Error in safe_append for {metric_key}: {e}")
                    self.history[history_key].append(default)
                    return default
            
            # Update history with safe conversion
            train_loss = safe_append('train_loss', 'avg_train_loss')
            val_loss = safe_append('val_loss', 'avg_val_loss', float('inf'))
            anomaly_ratio = safe_append('anomaly_ratio', 'avg_anomaly_ratio')
            learning_rate = safe_append('learning_rate', 'avg_learning_rate')
            
            # Debug output
            print(f"\nMetrics being saved:")
            print(f"- Train Loss: {train_loss}")
            print(f"- Val Loss: {val_loss}")
            print(f"- Anomaly Ratio: {anomaly_ratio}")
            print(f"- Learning Rate: {learning_rate}")
            
            # Save best model if validation loss improved
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_best_model(server_round, val_loss)
            
            # Save history after each round
            self._save_history()

            # Log to Supabase
            self._log_to_supabase(server_round, aggregated_metrics)
            
            return aggregated_metrics
            
        except Exception as e:
            print(f"Error in _log_metrics: {e}")
            import traceback
            traceback.print_exc()
            return aggregated_metrics

    def _log_to_supabase(self, server_round: int, aggregated_metrics: Dict[str, Scalar]):
        """Log metrics to Supabase."""
        try:
            if self.supabase_client:
                # Prepare data for Supabase
                data_to_insert = {
                    "server_round": server_round,
                    "avg_loss": aggregated_metrics.get("avg_train_loss"),
                    "std_loss": aggregated_metrics.get("std_train_loss"),
                    "avg_accuracy": aggregated_metrics.get("avg_accuracy"),
                    "model_checkpoint_path": os.path.join(self.log_dir, f"model_round_{server_round}.pth")
                }
                
                # Filter out None values
                data_to_insert = {k: v for k, v in data_to_insert.items() if v is not None}

                if data_to_insert:
                    self.supabase_client.table("training_runs").insert(data_to_insert).execute()
                    print("Successfully logged metrics to Supabase.")
                else:
                    print("No data to log to Supabase.")

        except Exception as e:
            print(f"Error logging to Supabase: {e}")

    def _save_history(self) -> None:
        """Save training history to a JSON file with error handling."""
        try:
            history_path = os.path.join(self.log_dir, "training_history.json")
            
            # Create a copy of history with only serializable values
            serializable_history = {}
            for key, values in self.history.items():
                serializable_history[key] = [float(v) if isinstance(v, (int, float)) else 0.0 for v in values]
            
            # Write to file atomically using a temporary file
            temp_path = f"{history_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(serializable_history, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Rename temp file to final file (atomic on Unix, not on Windows but good enough for our case)
            if os.path.exists(history_path):
                os.remove(history_path)
            os.rename(temp_path, history_path)
            
            print(f"Training history saved to {os.path.abspath(history_path)}")
            
        except Exception as e:
            print(f"Error saving training history: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_best_model(self, round_num: int, val_loss: float) -> None:
        """Save the current model as the best model if it has the lowest validation loss."""
        # Save the model state
        torch.save({
            'round': round_num,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'input_dim': self.input_dim
        }, self.best_model_path)
        
        print(f"\nNew best model saved at round {round_num} with validation loss: {val_loss:.6f}")
        print(f"Model saved to: {os.path.abspath(self.best_model_path)}")
        
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        
        # Sample clients for evaluation
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Create the evaluation instruction
        evaluate_ins = EvaluateIns(parameters, {})
        
        # Return instructions for each sampled client
        return [(client, evaluate_ins) for client in clients]

    def evaluate(
        self, server_round: int, parameters: fl.common.NDArrays
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None

        # Evaluate model
        result = self.eval_fn(server_round, parameters, {})
        if result is None:
            return None
        return result
