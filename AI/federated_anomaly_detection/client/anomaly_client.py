import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from typing import Dict, List, Tuple, Optional, Any, Union
import flwr as fl
from flwr.common import Scalar, NDArrays, Parameters, FitRes, EvaluateRes
from datetime import datetime
import json
import time
import signal
import sys
import uuid
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() # Load environment variables

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from federated_anomaly_detection.models.autoencoder import create_model
from federated_anomaly_detection.utils.data_utils import load_node_data, generate_cloud_activity_data, load_network_data

API_SERVER_URL = os.getenv("API_SERVER_URL", "http://localhost:8000") # Default to localhost

def get_device() -> torch.device:
    """Get the device to run the model on"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AnomalyDetectionClient(fl.client.NumPyClient):
    """Flower client implementing federated anomaly detection using an autoencoder."""
    def __init__(self, node_id: int, data_path: str):
        self.node_id = node_id
        self.data_path = data_path
        self.device = get_device()
        self.current_round = 0
        self.best_loss = float('inf')
        
        # Set up signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        # Initialize x and y
        x = None
        y = None
        
        # Load network data
        if os.path.exists(data_path):
            # Load from specified .npz file
            data = np.load(data_path)
            x = data['features']
            y = data.get('labels')  # Use .get() for safe access
            if y is None:
                print("Warning: 'labels' not found in data file. Using dummy labels.")
                y = np.zeros(x.shape[0], dtype=int)
            print(f"Loaded data from {data_path} with shape {x.shape} and {len(y)} labels")
            
            # Ensure features are in [0,1] range for autoencoder
            x_min = np.min(x, axis=0, keepdims=True)
            x_max = np.max(x, axis=0, keepdims=True)
            x = (x - x_min) / (x_max - x_min + 1e-8)  # Add small epsilon to avoid division by zero
            
            # Check for NaN or inf values
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                print("Warning: Data contains NaN or inf values after normalization. Replacing with 0.")
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            # Fallback to other methods if file not found
            try:
                x, y = load_network_data(node_id, os.path.dirname(data_path))
                if y is None:
                    y = np.zeros(len(x), dtype=int)
            except FileNotFoundError:
                print(f"No data file found. Using default data loading...")
                try:
                    x, y = load_node_data(node_id, os.path.dirname(data_path))
                    if y is None:
                        y = np.zeros(len(x), dtype=int)
                except FileNotFoundError:
                    print(f"No data file found. Generating synthetic data...")
                    x, y = generate_cloud_activity_data(n_samples=1000, n_features=9, random_state=42+node_id)
                    if y is None:
                        y = np.zeros(len(x), dtype=int)
                    
                    # Normalize synthetic data
                    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
        
        # Ensure y is defined and has the same length as x
        if y is None or len(y) != len(x):
            print("Warning: Invalid labels. Using dummy labels.")
            y = np.zeros(len(x), dtype=int)
        
        # Convert to PyTorch tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Split data into train/validation (80/20)
        split_idx = int(0.8 * len(x_tensor))
        self.train_data = x_tensor[:split_idx]
        self.val_data = x_tensor[split_idx:]
        self.train_labels = y_tensor[:split_idx]
        self.val_labels = y_tensor[split_idx:]
        
        # Create data loaders with optimized settings
        self.batch_size = 128  # Reduced from 256 to 128 for better stability
        self.gradient_accumulation_steps = 2  # Reduced accumulation steps
        
        # Enable automatic mixed precision if CUDA is available
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Pin memory if using CUDA for faster data transfer
        pin_memory = torch.cuda.is_available()
        
        # For Windows, we need to set num_workers=0 to avoid multiprocessing issues
        num_workers = 0 if os.name == 'nt' else 2
        
        self.train_loader = DataLoader(
            TensorDataset(self.train_data, self.train_data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        self.val_loader = DataLoader(
            TensorDataset(self.val_data, self.val_labels),
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        # Calculate class weights for imbalanced data
        class_counts = np.bincount(self.train_labels.numpy())
        total_samples = len(self.train_labels)
        class_weights = torch.tensor([total_samples / (len(class_counts) * count) if count > 0 else 1.0 
                                   for count in class_counts], 
                                   dtype=torch.float32, device=self.device)
        print(f"Class counts: {class_counts}")
        print(f"Class weights: {class_weights.cpu().numpy()}")
        
        # Create model, optimizer, and scheduler with class weights
        self.input_dim = x.shape[1]
        self.model, self.optimizer, self.scheduler, self.class_weights = create_model(
            input_dim=self.input_dim,
            learning_rate=0.001,
            device=self.device,
            class_weights=class_weights
        )
        
        # Track best validation loss
        self.best_val_loss = float('inf')
        self.patience = 5
        self.counter = 0
        
        # Create log directory
        self.log_dir = f"logs/client_{node_id}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"Client {node_id} initialized with {len(self.train_data)} training samples on {self.device}")
    
    def _handle_interrupt(self, signum, frame):
        """Handle interruption signals"""
        print(f"\nReceived interrupt signal. Exiting gracefully.")
        sys.exit(0)
        
    # Remove to_client method as it's not needed with direct NumPyClient implementation
    
    def get_parameters(self, config: Optional[Dict[str, Scalar]] = None) -> NDArrays:
        """Get model parameters as a list of NumPy arrays.
        
        Args:
            config: Configuration parameters (unused).
            
        Returns:
            List[np.ndarray]: List of model parameters as NumPy arrays.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from a list of NumPy arrays.
        
        Args:
            parameters: List of NumPy arrays containing model parameters.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)
    
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Train the model on the local data with mixed precision and gradient accumulation."""
        # Get current round from config
        server_round = config.get("server_round", 0)
        self.current_round = int(server_round) if isinstance(server_round, str) else server_round
        
        # Server manages the global model state
        print(f"\nClient {self.node_id} - Starting at round {self.current_round}")
        
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Get training config
        epochs = int(config.get("epochs", 5))  # Reduced default epochs since we're using larger batches
        
        # Training variables
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        # Enable cudnn benchmarking for better performance (if using CUDA)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Training loop with mixed precision and gradient accumulation
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', 
                                 dtype=torch.float16 if self.scaler is not None else None):
                for batch_idx, (data, _) in enumerate(self.train_loader):
                    data = data.to(self.device, non_blocking=True)
                    
                    # Forward pass with class weights
                    self.optimizer.zero_grad()
                    reconstructed = self.model(data)
                    
                    # Calculate reconstruction errors
                    reconstruction_errors = F.mse_loss(reconstructed, data, reduction='none').mean(dim=1)
                    
                    # Calculate loss with class weights if available
                    if hasattr(self, 'class_weights') and self.class_weights is not None:
                        # Get batch labels (use all zeros if not available)
                        batch_labels = torch.zeros(data.size(0), dtype=torch.long, device=self.device)
                        if hasattr(self, 'train_labels'):
                            batch_indices = torch.arange(len(self.train_loader.dataset))[
                                batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size
                            ]
                            batch_labels = self.train_labels[batch_indices].to(self.device)
                        
                        # Apply class weights to reconstruction errors
                        weights = self.class_weights[batch_labels]
                        loss = (reconstruction_errors * weights).mean()
                    else:
                        loss = reconstruction_errors.mean()
                    
                    # Scale loss and backpropagate
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        
                        # Gradient accumulation
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                    else:
                        # No mixed precision - standard backprop
                        (loss / self.gradient_accumulation_steps).backward()
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
            
            # Step the scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Evaluate on validation set for ReduceLROnPlateau
                    val_loss = self.model.evaluate(self.val_loader, self.device)
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Calculate average loss
            avg_train_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            
            # Evaluate on validation set
            val_loss = self.model.evaluate(self.val_loader, self.device)
            val_losses.append(val_loss)
            
            # No need to save checkpoints - server manages the global model state
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Log metrics
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Client {self.node_id} - Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f} - "
                  f"LR: {lr:.6f}")
        
        # Calculate final metrics
        avg_train_loss = sum(train_losses) / len(train_losses)
        final_val_loss = val_losses[-1]
        
        # Store the last training loss for evaluation
        self.last_train_loss = avg_train_loss
        
        # Save model checkpoint
        os.makedirs(self.log_dir, exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': final_val_loss,
                'epoch': epochs,
                'best_val_loss': best_val_loss
            }, 
            os.path.join(self.log_dir, f'model_round_{config.get("server_round", 0)}.pth')
        )
        
        # Log training summary
        round_num = config.get("server_round", 0)
        print(f"\nClient {self.node_id} Training Summary (Round {round_num}):")
        print(f"  - Epochs: {epochs}")
        print(f"  - Final Train Loss: {avg_train_loss:.6f}")
        print(f"  - Final Val Loss: {final_val_loss:.6f}")
        print(f"  - Best Val Loss: {best_val_loss:.6f}")
        print(f"  - Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Return updated parameters and metrics in format: (parameters, num_examples, metrics_dict)
        metrics = {
            "train_loss": float(avg_train_loss),
            "val_loss": float(final_val_loss),
            "best_val_loss": float(best_val_loss),
            "learning_rate": float(self.optimizer.param_groups[0]['lr']),
            "epochs": int(epochs)
        }
        
        return self.get_parameters(), len(self.train_loader.dataset), metrics
    
    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on the local test set with comprehensive metrics."""
        self.set_parameters(parameters)
        self.model.eval()
        
        # Initialize metrics
        total_loss = 0.0
        all_errors = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                reconstructed = self.model(features)
                
                # Calculate reconstruction error
                loss = F.smooth_l1_loss(reconstructed, features, reduction='sum')
                total_loss += loss.item()
                
                # Store errors and labels for anomaly detection
                mse = F.mse_loss(reconstructed, features, reduction='none').mean(dim=1)
                all_errors.append(mse.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        val_loss = total_loss / len(self.val_loader.dataset) if len(self.val_loader.dataset) > 0 else float('inf')
        all_errors = np.concatenate(all_errors) if all_errors else np.array([])
        all_labels = np.concatenate(all_labels) if all_labels else np.array([])
        
        # Initialize default values
        threshold = 0.0
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        anomaly_ratio = 0.0
        
        # Only calculate metrics if we have data
        if len(all_errors) > 0 and len(all_labels) > 0:
            try:
                # Dynamic threshold based on validation errors (95th percentile)
                threshold = np.percentile(all_errors, 95) if len(all_errors) > 0 else 0.0
                predicted_anomalies = (all_errors > threshold).astype(int)
                
                # Calculate metrics only if we have both positive and negative samples
                if len(np.unique(all_labels)) > 1 and len(np.unique(predicted_anomalies)) > 1:
                    accuracy = accuracy_score(all_labels, predicted_anomalies)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        all_labels, predicted_anomalies, average='binary', zero_division=0
                    )
                
                # Calculate anomaly ratio (ratio of samples detected as anomalies)
                anomaly_ratio = np.mean(predicted_anomalies)  # Keep as a ratio (0 to 1)
                
            except Exception as e:
                print(f"Error calculating metrics: {e}")
        
        # --- Anomaly Reporting ---
        # If anomalies are detected, send them to the API server
        if len(all_errors) > 0 and len(all_labels) > 0 and 'predicted_anomalies' in locals():
            predicted_anomalies_indices = np.where(predicted_anomalies == 1)[0]
            for idx in predicted_anomalies_indices:
                # We need to get the original feature values for the anomaly.
                # Assuming `self.val_data` holds the original pre-normalized data
                # For now, we will use the index to reconstruct a simple version
                # In a real system, you'd store the original raw data for better reporting.
                anomaly_details_mock = {
                    "feature_values": self.val_data[idx].cpu().numpy().tolist(),
                    "reconstruction_error": all_errors[idx].item(),
                }

                # Construct the AnomalyData object matching the FastAPI Pydantic model
                anomaly_data = {
                    "id": str(uuid.uuid4()),  # Generate a unique ID for the anomaly
                    "timestamp": datetime.now().isoformat(),
                    "severity": "high",  # Placeholder, could be dynamic based on error magnitude
                    "source_ip": f"client_{self.node_id}_src",  # Placeholder
                    "destination_ip": "unknown_dest",  # Placeholder
                    "protocol": "unknown",  # Placeholder
                    "action": "flagged",  # Placeholder
                    "confidence": float(all_errors[idx].item()),  # Using error as confidence
                    "reviewed": False,
                    "details": json.dumps(anomaly_details_mock),
                }
                self._report_anomaly_to_server(anomaly_data)
        # --- End Anomaly Reporting ---
        
        # Get learning rate safely
        try:
            learning_rate = float(self.optimizer.param_groups[0]['lr'])
        except (IndexError, KeyError, AttributeError):
            learning_rate = 0.0
        
        # Log detailed results
        print(f"\nClient {self.node_id} Evaluation:")
        print(f"  - Val Loss: {val_loss:.6f}")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1-Score: {f1:.4f}")
        print(f"  - Anomaly Ratio: {anomaly_ratio * 100:.2f}%")
        print(f"  - Threshold (95th %ile): {threshold:.6f}")
        
        # Return comprehensive metrics with safe defaults
        results = {
            "val_loss": float(val_loss) if np.isfinite(val_loss) else float('inf'),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "anomaly_ratio": float(anomaly_ratio),
            "threshold": float(threshold),
            "train_loss": float(getattr(self, 'last_train_loss', 0.0)),
            "learning_rate": learning_rate,
            "num_samples": len(all_errors)
        }
        
        # Ensure all metrics are finite
        for k, v in results.items():
            if isinstance(v, float) and not np.isfinite(v):
                results[k] = 0.0
        
        # Save evaluation results
        round_num = config.get("server_round", 0)
        with open(os.path.join(self.log_dir, f'eval_results_round_{round_num}.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Log metrics
        print(f"\nClient {self.node_id} Evaluation Results (Round {round_num}):")
        print(f"  - Val Loss: {val_loss:.6f}")
        print(f"  - Train Loss: {results['train_loss']:.6f}")
        print(f"  - Accuracy: {results['accuracy']:.4f}")
        print(f"  - Precision: {results['precision']:.4f}")
        print(f"  - Recall: {results['recall']:.4f}")
        print(f"  - F1-Score: {results['f1_score']:.4f}")
        print(f"  - Anomaly Ratio: {results['anomaly_ratio'] * 100:.2f}%")
        print(f"  - Learning Rate: {results['learning_rate']:.6f}")
        print(f"  - Threshold: {results['threshold']:.6f}\n")
        
        # Return metrics in format: (loss, num_examples, metrics_dict)
        return float(val_loss), len(self.val_loader.dataset), results
    
    def _report_anomaly_to_server(self, anomaly_data: Dict[str, Any]):
        """Sends detected anomaly data to the API server."""
        try:
            response = requests.post(
                f"{API_SERVER_URL}/report_anomaly",
                json=anomaly_data,
                timeout=5  # 5-second timeout
            )
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            print(f"Anomaly {anomaly_data['id']} reported successfully to server.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to report anomaly {anomaly_data['id']} to server: {e}")
