#!/usr/bin/env python3
"""
Optimized Federated Learning Server with enhanced metrics and precision focus
"""
import os
import sys
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import time
import json

# Disable TensorFlow imports to avoid protobuf issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.modules['tensorflow'] = None
sys.modules['tensorflow.python'] = None

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import flwr as fl
    from flwr.server import ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from shared_model import create_shared_model
    print("✅ Flower imported successfully (TensorFlow bypassed)")
except ImportError as e:
    print(f"❌ Flower import error: {e}")
    sys.exit(1)

class OptimizedFederatedAutoencoder(nn.Module):
    """Optimized autoencoder with better architecture for precision"""
    def __init__(self, input_dim=79, encoding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_optimized_strategy(input_dim=79, min_clients=2, num_rounds=10):
    """Create optimized federated learning strategy with precision focus"""
    
    # Create shared model to ensure consistency
    model = create_shared_model(input_dim=input_dim)
    
    # Get initial parameters
    initial_params = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    # Enhanced training history
    training_history = {
        'rounds': [],
        'train_losses': [],
        'val_losses': [],
        'accuracies': [],
        'precisions': [],
        'recalls': [],
        'f1_scores': [],
        'client_counts': [],
        'aggregation_times': [],
        'learning_rates': []
    }
    
    # Custom optimized strategy
    class OptimizedFedAvg(FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.round_num = 0
            self.best_accuracy = 0.0
            self.best_precision = 0.0
            self.best_f1 = 0.0
            self.training_history = training_history
            self.start_time = time.time()
        
        def aggregate_fit(self, server_round, results, failures):
            """Enhanced aggregation with precision-focused monitoring"""
            self.round_num = server_round
            round_start_time = time.time()
            
            # Log client participation
            print(f"\n{'='*80}")
            print(f"🔄 ROUND {server_round} - OPTIMIZED FEDERATED LEARNING")
            print(f"{'='*80}")
            print(f"📊 Clients participated: {len(results)}")
            print(f"❌ Client failures: {len(failures)}")
            
            # Client details with enhanced metrics
            for client_proxy, fit_res in results:
                try:
                    # Handle different metric formats
                    if hasattr(fit_res, 'metrics') and fit_res.metrics:
                        client_id = fit_res.metrics.get('client_id', 'Unknown')
                        train_loss = fit_res.metrics.get('train_loss', 0)
                        epochs = fit_res.metrics.get('epochs', 0)
                        learning_rate = fit_res.metrics.get('learning_rate', 0)
                    elif isinstance(fit_res, (list, tuple)) and len(fit_res) > 1:
                        # Handle tuple format (num_samples, metrics_dict)
                        metrics_dict = fit_res[1] if hasattr(fit_res[1], '__dict__') else fit_res[1]
                        if hasattr(metrics_dict, '__dict__'):
                            client_id = getattr(metrics_dict, 'client_id', 'Unknown')
                            train_loss = getattr(metrics_dict, 'train_loss', 0)
                            epochs = getattr(metrics_dict, 'epochs', 0)
                            learning_rate = getattr(metrics_dict, 'learning_rate', 0)
                        elif isinstance(metrics_dict, dict):
                            client_id = metrics_dict.get('client_id', 'Unknown')
                            train_loss = metrics_dict.get('train_loss', 0)
                            epochs = metrics_dict.get('epochs', 0)
                            learning_rate = metrics_dict.get('learning_rate', 0)
                        else:
                            client_id = 'Unknown'
                            train_loss = 0
                            epochs = 0
                            learning_rate = 0
                    else:
                        client_id = 'Unknown'
                        train_loss = 0
                        epochs = 0
                        learning_rate = 0
                    
                    print(f"   👤 Client {client_id}: Loss={train_loss:.6f}, Epochs={epochs}, LR={learning_rate:.6f}")
                    
                except Exception as e:
                    print(f"   ⚠️  Client metrics extraction error: {e}")
                    print(f"   👤 Client Unknown: Loss=0.000000, Epochs=0, LR=0.000000")
            
            # Call parent aggregation
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            
            if aggregated_parameters is not None:
                aggregation_time = time.time() - round_start_time
                
                # Enhanced metrics logging - extract from aggregated_metrics (dictionary format)
                if aggregated_metrics and isinstance(aggregated_metrics, dict):
                    train_loss = float(aggregated_metrics.get("train_loss", 0.0))
                    val_loss = float(aggregated_metrics.get("val_loss", 0.0))
                    accuracy = float(aggregated_metrics.get("accuracy", 0.0))
                    precision = float(aggregated_metrics.get("precision", 0.0))
                    recall = float(aggregated_metrics.get("recall", 0.0))
                    f1_score = float(aggregated_metrics.get("f1_score", 0.0))
                else:
                    train_loss = 0.0
                    val_loss = 0.0
                    accuracy = 0.0
                    precision = 0.0
                    recall = 0.0
                    f1_score = 0.0
                
                # Store in history
                self.training_history['rounds'].append(server_round)
                self.training_history['train_losses'].append(train_loss)
                self.training_history['val_losses'].append(val_loss)
                self.training_history['accuracies'].append(accuracy)
                self.training_history['precisions'].append(precision)
                self.training_history['recalls'].append(recall)
                self.training_history['f1_scores'].append(f1_score)
                self.training_history['client_counts'].append(len(results))
                self.training_history['aggregation_times'].append(aggregation_time)
                
                # Performance analysis with precision focus
                improvements = []
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    improvements.append("🟢 NEW BEST ACCURACY")
                if precision > self.best_precision:
                    self.best_precision = precision
                    improvements.append("🎯 NEW BEST PRECISION")
                if f1_score > self.best_f1:
                    self.best_f1 = f1_score
                    improvements.append("📊 NEW BEST F1")
                
                if not improvements:
                    improvements.append("🔴 No improvement")
                
                # Calculate training progress
                elapsed_time = time.time() - self.start_time
                avg_time_per_round = elapsed_time / server_round
                eta = avg_time_per_round * (num_rounds - server_round)
                
                print(f"📈 OPTIMIZED AGGREGATION RESULTS:")
                print(f"   🎯 Train Loss: {train_loss:.6f}")
                print(f"   🔍 Val Loss: {val_loss:.6f}")
                print(f"   📊 Accuracy: {accuracy * 100:.2f}% {'🟢' if accuracy > 0.75 else '🔴'}")
                print(f"   🎯 Precision: {precision * 100:.2f}% {'🟢' if precision > 0.5 else '🔴'}")
                print(f"   🔍 Recall: {recall * 100:.2f}%")
                print(f"   📈 F1-Score: {f1_score * 100:.2f}%")
                print(f"   ⏱️  Aggregation Time: {aggregation_time:.2f}s")
                print(f"   🏆 Best Accuracy: {self.best_accuracy * 100:.2f}%")
                print(f"   🎯 Best Precision: {self.best_precision * 100:.2f}%")
                print(f"   📊 Best F1-Score: {self.best_f1 * 100:.2f}%")
                print(f"   ⏰ Elapsed: {elapsed_time:.1f}s, ETA: {eta:.1f}s")
                print(f"   {'   '.join(improvements)}")
                
                # Save checkpoint
                self.save_checkpoint(server_round, aggregated_parameters, aggregated_metrics)
                
                print(f"{'='*80}\n")
            
            return aggregated_parameters, aggregated_metrics
        
        def save_checkpoint(self, round_num, parameters, metrics):
            """Save model checkpoint and training history"""
            try:
                # Create checkpoints directory
                checkpoint_dir = Path("checkpoints")
                checkpoint_dir.mkdir(exist_ok=True)
                
                # Save model parameters - unpack Parameters object
                param_arrays = parameters_to_ndarrays(parameters)
                checkpoint_file = checkpoint_dir / f"round_{round_num}_optimized_model.npz"
                np.savez_compressed(checkpoint_file, *param_arrays)
                
                # Save training history
                history_file = checkpoint_dir / "optimized_training_history.json"
                with open(history_file, 'w') as f:
                    json.dump(self.training_history, f, indent=2)
                
                print(f"💾 Optimized checkpoint saved: {checkpoint_file}")
                
            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")
    
    strategy = OptimizedFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.8,
        min_fit_clients=min_clients,
        min_evaluate_clients=max(1, min_clients // 2),
        min_available_clients=min_clients,
        on_fit_config_fn=lambda server_round: {
            "epochs": 5,  # Reduced from 10 to 5 for faster training
            "batch_size": 64,
            "server_round": server_round,
            "learning_rate": 0.001 * (0.95 ** (server_round // 3)),
        },
        fit_metrics_aggregation_fn=lambda metrics: {
            "train_loss": float(np.mean([m[1].train_loss if isinstance(m, tuple) and len(m) > 1 and hasattr(m[1], 'train_loss') else 
                                      (m[1].get("train_loss", 0) if isinstance(m, tuple) and len(m) > 1 else 
                                       m.get("train_loss", 0)) for m in metrics])) if metrics else 0.0
        },
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "val_loss": float(np.mean([m[1].val_loss if isinstance(m, tuple) and len(m) > 1 and hasattr(m[1], 'val_loss') else 
                                    (m[1].get("val_loss", 0) if isinstance(m, tuple) and len(m) > 1 else 
                                     m.get("val_loss", 0)) for m in metrics])) if metrics else 0.0,
            "accuracy": float(np.mean([m[1].accuracy if isinstance(m, tuple) and len(m) > 1 and hasattr(m[1], 'accuracy') else 
                                    (m[1].get("accuracy", 0) if isinstance(m, tuple) and len(m) > 1 else 
                                     m.get("accuracy", 0)) for m in metrics])) if metrics else 0.0,
            "precision": float(np.mean([m[1].precision if isinstance(m, tuple) and len(m) > 1 and hasattr(m[1], 'precision') else 
                                      (m[1].get("precision", 0) if isinstance(m, tuple) and len(m) > 1 else 
                                       m.get("precision", 0)) for m in metrics])) if metrics else 0.0,
            "recall": float(np.mean([m[1].recall if isinstance(m, tuple) and len(m) > 1 and hasattr(m[1], 'recall') else 
                                  (m[1].get("recall", 0) if isinstance(m, tuple) and len(m) > 1 else 
                                   m.get("recall", 0)) for m in metrics])) if metrics else 0.0,
            "f1_score": float(np.mean([m[1].f1_score if isinstance(m, tuple) and len(m) > 1 and hasattr(m[1], 'f1_score') else 
                                    (m[1].get("f1_score", 0) if isinstance(m, tuple) and len(m) > 1 else 
                                     m.get("f1_score", 0)) for m in metrics])) if metrics else 0.0,
            "threshold": float(np.mean([m[1].threshold if isinstance(m, tuple) and len(m) > 1 and hasattr(m[1], 'threshold') else 
                                    (m[1].get("threshold", 0) if isinstance(m, tuple) and len(m) > 1 else 
                                     m.get("threshold", 0)) for m in metrics])) if metrics else 0.0,
        },
        initial_parameters=ndarrays_to_parameters(initial_params),
    )
    
    return strategy

def main():
    parser = argparse.ArgumentParser(description="Optimized Federated Learning Server")
    parser.add_argument("--input_dim", type=int, default=79)
    parser.add_argument("--min_clients", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--address", type=str, default="localhost:8080")
    parser.add_argument("--round-timeout", type=float, default=1800.0,
                       help="Round timeout in seconds (increased for large datasets)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 OPTIMIZED FEDERATED LEARNING SERVER")
    print("=" * 80)
    print(f"📊 Input dimension: {args.input_dim}")
    print(f"👥 Minimum clients: {args.min_clients}")
    print(f"🔄 Number of rounds: {args.num_rounds}")
    print(f"🌐 Server address: {args.address}")
    print(f"⏱️  Round timeout: {args.round_timeout}s")
    print(f"🎯 Optimized for precision and accuracy")
    print(f"🔧 TensorFlow bypassed to avoid protobuf issues")
    print("=" * 80)
    
    # Create optimized strategy
    strategy = create_optimized_strategy(
        input_dim=args.input_dim,
        min_clients=args.min_clients,
        num_rounds=args.num_rounds
    )
    
    # Create server config
    config = ServerConfig(
        num_rounds=args.num_rounds,
        round_timeout=args.round_timeout,
    )
    
    try:
        print(f"\n🎯 Starting optimized server on {args.address}...")
        print(f"⏳ Waiting for {args.min_clients} clients to connect...")
        print(f"📈 Training with precision-focused optimization")
        
        # Start server
        fl.server.start_server(
            server_address=args.address,
            config=config,
            strategy=strategy,
        )
        
        print(f"\n🎉 Optimized server completed {args.num_rounds} rounds successfully!")
        print(f"📊 Checkpoints saved in 'checkpoints/' directory")
        print(f"📈 Enhanced training history available for analysis")
        
        # Final summary
        if hasattr(strategy, 'training_history'):
            history = strategy.training_history
            if history['accuracies']:
                print(f"\n🏆 FINAL PERFORMANCE SUMMARY:")
                print(f"   📊 Best Accuracy: {max(history['accuracies']) * 100:.2f}%")
                print(f"   🎯 Best Precision: {max(history['precisions']) * 100:.2f}%")
                print(f"   🔍 Best Recall: {max(history['recalls']) * 100:.2f}%")
                print(f"   📈 Best F1-Score: {max(history['f1_scores']) * 100:.2f}%")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
