#!/usr/bin/env python3
"""
Enhanced client with optimized anomaly detection thresholds
"""
import os
import sys
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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
    from flwr.common import Parameters, Scalar
    from shared_model import create_shared_model
    print("✅ Flower imported successfully (TensorFlow bypassed)")
except ImportError as e:
    print(f"❌ Flower import error: {e}")
    sys.exit(1)

class OptimizedFederatedClient(fl.client.NumPyClient):
    """Enhanced federated learning client with optimized anomaly detection"""
    
    def __init__(self, client_id, data_path):
        self.client_id = client_id
        self.data_path = data_path
        
        print(f"🔧 Initializing Optimized Client {client_id}...")
        
        # Load and validate data
        self._load_data()
        
        # Create enhanced model
        self._create_model()
        
        # Setup enhanced training
        self._setup_training()
        
        print(f"✅ Optimized Client {client_id} initialized successfully")
        print(f"   📊 Training samples: {len(self.train_features)}")
        print(f"   📊 Validation samples: {len(self.val_features)}")
        print(f"   🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_data(self):
        """Load and validate client data"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        data = np.load(self.data_path)
        
        # Load features and labels
        self.train_features = data['features'].astype(np.float32)
        self.train_labels = data.get('labels', np.zeros(len(self.train_features)))
        
        # Use validation data if available, otherwise split from training
        if 'val_features' in data:
            self.val_features = data['val_features'].astype(np.float32)
            self.val_labels = data.get('val_labels', np.zeros(len(self.val_features)))
        else:
            # Split training data for validation
            split_idx = int(0.8 * len(self.train_features))
            self.val_features = self.train_features[split_idx:].astype(np.float32)
            self.val_labels = self.train_labels[split_idx:]
            self.train_features = self.train_features[:split_idx].astype(np.float32)
            self.train_labels = self.train_labels[:split_idx]
        
        # Validate data
        if np.isnan(self.train_features).any():
            print(f"⚠️  Client {self.client_id}: Training data contains NaN, replacing with 0")
            self.train_features = np.nan_to_num(self.train_features, nan=0.0)
        
        if np.isnan(self.val_features).any():
            print(f"⚠️  Client {self.client_id}: Validation data contains NaN, replacing with 0")
            self.val_features = np.nan_to_num(self.val_features, nan=0.0)
        
        # Normalize to [0, 1] range
        self._normalize_data()
        
        print(f"📈 Data Distribution - Normal: {np.sum(self.train_labels == 0):,}, Anomaly: {np.sum(self.train_labels == 1):,}")
    
    def _normalize_data(self):
        """Normalize data to [0, 1] range"""
        # Fit on training data
        self.data_min = self.train_features.min(axis=0, keepdims=True)
        self.data_max = self.train_features.max(axis=0, keepdims=True)
        self.data_range = self.data_max - self.data_min + 1e-8
        
        # Normalize both training and validation
        self.train_features = (self.train_features - self.data_min) / self.data_range
        self.val_features = (self.val_features - self.data_min) / self.data_range
    
    def _create_model(self):
        """Create optimized model using shared architecture"""
        input_dim = self.train_features.shape[1]
        
        # Use shared model to ensure consistency with server
        self.model = create_shared_model(input_dim=input_dim)
        self.device = torch.device('cpu')
        self.model.to(self.device)
    
    def _setup_training(self):
        """Setup enhanced training components"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.8,
            patience=2,
            min_lr=1e-6
        )
        
        # Create enhanced data loaders
        train_dataset = TensorDataset(
            torch.tensor(self.train_features, dtype=torch.float32),
            torch.tensor(self.train_features, dtype=torch.float32)
        )
        
        val_dataset = TensorDataset(
            torch.tensor(self.val_features, dtype=torch.float32),
            torch.tensor(self.val_labels, dtype=torch.float32)
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=64,
            shuffle=True,
            num_workers=0
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=128,
            shuffle=False,
            num_workers=0
        )
    
    def get_parameters(self, config):
        """Get model parameters"""
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def optimize_threshold(self, errors, labels):
        """Optimize anomaly detection threshold for maximum precision with minimum recall requirement"""
        # Precision-focused: test higher percentiles (85-98th)
        thresholds = np.percentile(errors, np.arange(85, 99, 1))  # Test 85-98th percentiles
        
        best_threshold = np.percentile(errors, 95)  # Default to 95th percentile for higher precision
        best_score = 0
        
        print(f"\n🎯 Precision-Optimized Threshold Search:")
        print(f"📊 Testing {len(thresholds)} threshold candidates...")
        
        for i, threshold in enumerate(thresholds):
            predictions = (errors > threshold).astype(int)
            
            # Calculate metrics
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
            
            # Precision-focused scoring with minimum recall requirement
            if recall >= 0.15:  # Minimum 15% recall requirement
                # Weight precision more heavily (70% precision, 30% recall)
                custom_score = 0.7 * precision + 0.3 * recall
                
                if custom_score > best_score:
                    best_score = custom_score
                    best_threshold = threshold
                
                anomaly_rate = np.mean(predictions) * 100
                percentile = 85 + i
                print(f"   🎯 {percentile}th percentile: {threshold:.6f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Anomaly Rate: {anomaly_rate:.1f}%")
        
        print(f"✅ Best threshold: {best_threshold:.6f} (Score: {best_score:.3f})")
        return best_threshold
    
    def fit(self, parameters, config):
        """Enhanced training with better monitoring"""
        self.set_parameters(parameters)
        
        # Get training config (optimized for large datasets)
        epochs = int(config.get("epochs", 5))  # Reduced from 10 to 5 for faster training
        server_round = int(config.get("server_round", 0))
        learning_rate = float(config.get("learning_rate", 0.001))
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        print(f"\n🏃 Client {self.client_id} - Round {server_round} Optimized Training")
        print(f"   📚 Epochs: {epochs}")
        print(f"   📈 Learning Rate: {learning_rate:.6f}")
        
        # Enhanced training loop
        self.model.train()
        total_loss = 0.0
        epoch_losses = []
        best_epoch_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.to(self.device)
                
                self.optimizer.zero_grad()
                reconstructed = self.model(data)
                loss = F.mse_loss(reconstructed, data)
                loss.backward()
                
                # Enhanced gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            total_loss += avg_epoch_loss
            
            # Learning rate scheduling
            self.scheduler.step(avg_epoch_loss)
            
            # Progress reporting
            if epoch % 2 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"   📊 Epoch {epoch+1:2d}/{epochs}: Loss = {avg_epoch_loss:.6f}, LR = {current_lr:.6f}")
            
            if avg_epoch_loss < best_epoch_loss:
                best_epoch_loss = avg_epoch_loss
        
        avg_train_loss = total_loss / epochs
        
        # Enhanced metrics
        metrics = {
            "train_loss": float(avg_train_loss),
            "best_epoch_loss": float(best_epoch_loss),
            "client_id": self.client_id,
            "server_round": server_round,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "num_batches": len(self.train_loader),
            "samples_processed": len(self.train_loader.dataset)
        }
        
        print(f"✅ Client {self.client_id} - Round {server_round} Optimized Training Completed")
        print(f"   📈 Average Training Loss: {avg_train_loss:.6f}")
        print(f"   🎯 Best Epoch Loss: {best_epoch_loss:.6f}")
        
        return self.get_parameters(config), len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters, config):
        """Enhanced evaluation with optimized thresholds and detailed metrics"""
        self.set_parameters(parameters)
        
        server_round = int(config.get("server_round", 0))
        
        print(f"\n🔍 Client {self.client_id} - Round {server_round} Optimized Evaluation")
        
        self.model.eval()
        total_loss = 0.0
        all_errors = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in self.val_loader:
                data = data.to(self.device)
                reconstructed = self.model(data)
                
                # Calculate reconstruction loss
                loss = F.mse_loss(reconstructed, data, reduction='sum')
                total_loss += loss.item()
                
                # Calculate per-sample errors for anomaly detection
                sample_errors = F.mse_loss(reconstructed, data, reduction='none').mean(dim=1)
                all_errors.append(sample_errors.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        all_errors = np.concatenate(all_errors) if all_errors else np.array([])
        all_labels = np.concatenate(all_labels) if all_labels else np.array([])
        
        # Optimize threshold for maximum precision
        optimal_threshold = self.optimize_threshold(all_errors, all_labels)
        predictions = (all_errors > optimal_threshold).astype(int)
        
        # Calculate comprehensive metrics
        precision = precision_score(all_labels, predictions, zero_division=0)
        recall = recall_score(all_labels, predictions, zero_division=0)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        accuracy = np.mean(predictions == all_labels)
        
        # Calculate ROC-AUC if we have both classes
        try:
            roc_auc = roc_auc_score(all_labels, all_errors)
        except:
            roc_auc = 0.0
        
        # Additional statistics
        anomaly_ratio = np.mean(predictions)
        error_std = np.std(all_errors)
        error_max = np.max(all_errors)
        error_min = np.min(all_errors)
        
        print(f"📊 Client {self.client_id} - Optimized Evaluation Results:")
        print(f"   📉 Validation Loss: {avg_loss:.6f}")
        print(f"   🎯 Optimal Threshold: {optimal_threshold:.6f}")
        print(f"   🚨 Anomaly Ratio: {anomaly_ratio * 100:.2f}%")
        print(f"   🎯 Accuracy: {accuracy * 100:.2f}%")
        print(f"   📈 Precision: {precision * 100:.2f}%")
        print(f"   🔍 Recall: {recall * 100:.2f}%")
        print(f"   📊 F1-Score: {f1 * 100:.2f}%")
        print(f"   📈 ROC-AUC: {roc_auc * 100:.2f}%")
        print(f"   📈 Error Std: {error_std:.6f}")
        
        metrics = {
            "val_loss": float(avg_loss),
            "threshold": float(optimal_threshold),
            "anomaly_ratio": float(anomaly_ratio),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "error_std": float(error_std),
            "error_max": float(error_max),
            "error_min": float(error_min),
            "client_id": self.client_id,
            "server_round": server_round,
            "val_samples": len(self.val_loader.dataset)
        }
        
        return float(avg_loss), len(self.val_loader.dataset), metrics

def main():
    parser = argparse.ArgumentParser(description="Optimized Federated Learning Client")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--data-dir", type=str, default="data/enhanced")
    parser.add_argument("--server-address", type=str, default="localhost:8080")
    
    args = parser.parse_args()
    
    # Construct data path
    data_path = Path(args.data_dir) / f"client_{args.client_id}.npz"
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return 1
    
    print("=" * 80)
    print(f"🚀 OPTIMIZED FEDERATED LEARNING CLIENT {args.client_id}")
    print("=" * 80)
    print(f"📂 Data path: {data_path}")
    print(f"🌐 Server address: {args.server_address}")
    print(f"🔧 TensorFlow bypassed to avoid protobuf issues")
    print(f"🎯 Optimized for precision and accuracy")
    print("=" * 80)
    
    # Create and start optimized client
    try:
        client = OptimizedFederatedClient(
            client_id=args.client_id,
            data_path=str(data_path)
        )
        
        print(f"\n🎯 Connecting to optimized server at {args.server_address}...")
        print("⏳ Waiting for server to start optimized training rounds...")
        
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client,
        )
        
        print(f"\n🎉 Optimized Client {args.client_id} completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Client {args.client_id} stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Client {args.client_id} error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
