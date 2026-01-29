"""
Calculate Classification Metrics for Trained Autoencoder
Convert reconstruction errors to anomaly detection performance metrics
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Import our modules
from autoencoder_model import AutoencoderConfig
from training_pipeline_fixed import FixedDataPreparation  # ✅ Use fixed data preparation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_classification_metrics():
    """Calculate comprehensive classification metrics for the trained model"""
    logger.info("🎯 Calculating Classification Metrics...")
    
    try:
        # Step 1: Load trained model
        logger.info("📥 Loading trained model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the best model
        checkpoint_path = Path("model_artifacts/best_autoencoder_fixed.pth")
        if not checkpoint_path.exists():
            logger.error("❌ Trained model not found! Run training_pipeline_fixed.py first")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config_dict = checkpoint['config']
        
        # Reconstruct config with CORRECT dimensions from model weights
        encoder_first_weight = checkpoint['model_state_dict']['encoder.0.weight']
        actual_input_dim = encoder_first_weight.shape[1]  # ✅ Get actual input dim from weights
        
        config = AutoencoderConfig()
        config.input_dim = actual_input_dim  # ✅ Use actual input_dim (78)
        config.encoding_dims = config_dict['encoding_dims']
        config.bottleneck_dim = config_dict['bottleneck_dim']
        config.dropout_rate = config_dict['dropout_rate']
        
        logger.info(f"✅ Using actual input dimension: {actual_input_dim}")
        
        # Load model
        from autoencoder_model import CloudAnomalyAutoencoder
        model = CloudAnomalyAutoencoder(
            input_dim=config.input_dim,
            encoding_dims=config.encoding_dims,
            bottleneck_dim=config.bottleneck_dim,
            dropout_rate=config.dropout_rate
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        logger.info(f"✅ Model loaded: {config.input_dim} → {config.bottleneck_dim} → {config.input_dim}")
        
        # Step 2: Load test data
        logger.info("📊 Loading test data...")
        data_prep = FixedDataPreparation()  # ✅ Use fixed data preparation
        
        # Prepare data first to populate all_data
        data_results = data_prep.prepare_data(batch_size=128)
        test_loader = data_results['test_loader']
        
        logger.info(f"✅ Test data loaded: {len(test_loader)} batches")
        
        # Step 3: Get predictions and reconstruction errors
        logger.info("🔍 Getting model predictions...")
        all_reconstruction_errors = []
        all_true_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.cpu().numpy()
                
                # Forward pass
                reconstructed, encoded = model(features)
                
                # Calculate reconstruction errors per sample
                batch_errors = torch.mean((reconstructed - features) ** 2, dim=1)
                
                all_reconstruction_errors.extend(batch_errors.cpu().numpy())
                all_true_labels.extend(labels)
        
        all_reconstruction_errors = np.array(all_reconstruction_errors)
        all_true_labels = np.array(all_true_labels)
        
        logger.info(f"✅ Processed {len(all_reconstruction_errors)} test samples")
        logger.info(f"  Normal samples: {np.sum(all_true_labels == 0)}")
        logger.info(f"  Anomaly samples: {np.sum(all_true_labels == 1)}")
        
        # Step 4: Determine optimal threshold
        logger.info("🎚️ Determining optimal threshold...")
        
        # Use 95th percentile of normal samples as threshold
        normal_errors = all_reconstruction_errors[all_true_labels == 0]
        threshold_95 = np.percentile(normal_errors, 95)
        
        # Also try different thresholds
        thresholds = {
            '95th_percentile': threshold_95,
            'mean_plus_2std': np.mean(normal_errors) + 2 * np.std(normal_errors),
            'mean_plus_3std': np.mean(normal_errors) + 3 * np.std(normal_errors),
            'median_plus_mad': np.median(normal_errors) + 2 * np.median(np.abs(normal_errors - np.median(normal_errors)))
        }
        
        logger.info(f"Thresholds: {thresholds}")
        
        # Step 5: Calculate metrics for each threshold
        logger.info("📈 Calculating classification metrics...")
        
        best_metrics = None
        best_threshold = None
        best_f1 = 0
        
        all_metrics = {}
        
        for threshold_name, threshold in thresholds.items():
            # Make predictions
            predictions = (all_reconstruction_errors > threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(all_true_labels, predictions)
            precision = precision_score(all_true_labels, predictions, zero_division=0)
            recall = recall_score(all_true_labels, predictions, zero_division=0)
            f1 = f1_score(all_true_labels, predictions, zero_division=0)
            
            # ROC-AUC
            try:
                roc_auc = roc_auc_score(all_true_labels, all_reconstruction_errors)
            except:
                roc_auc = 0.5
            
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': confusion_matrix(all_true_labels, predictions).tolist()
            }
            
            all_metrics[threshold_name] = metrics
            
            logger.info(f"  {threshold_name}:")
            logger.info(f"    Threshold: {threshold:.6f}")
            logger.info(f"    Accuracy: {accuracy:.4f}")
            logger.info(f"    Precision: {precision:.4f}")
            logger.info(f"    Recall: {recall:.4f}")
            logger.info(f"    F1-Score: {f1:.4f}")
            logger.info(f"    ROC-AUC: {roc_auc:.4f}")
            
            # Track best F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_metrics = metrics
                best_threshold = threshold_name
        
        # Step 6: Generate detailed report for best threshold
        logger.info(f"🏆 Best performing threshold: {best_threshold}")
        logger.info(f"   F1-Score: {best_metrics['f1_score']:.4f}")
        
        # Step 7: Create visualizations
        logger.info("📊 Creating visualizations...")
        
        # Plot 1: ROC Curve
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        fpr, tpr, _ = roc_curve(all_true_labels, all_reconstruction_errors)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {best_metrics["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Precision-Recall Curve
        plt.subplot(2, 3, 2)
        precision_curve, recall_curve, _ = precision_recall_curve(all_true_labels, all_reconstruction_errors)
        plt.plot(recall_curve, precision_curve, color='green', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        
        # Plot 3: Reconstruction Error Distribution
        plt.subplot(2, 3, 3)
        normal_errors = all_reconstruction_errors[all_true_labels == 0]
        anomaly_errors = all_reconstruction_errors[all_true_labels == 1]
        
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.axvline(best_metrics['threshold'], color='black', linestyle='--', label=f'Threshold = {best_metrics["threshold"]:.3f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Reconstruction Error Distribution')
        plt.legend()
        plt.grid(True)
        
        # Plot 4: Confusion Matrix
        plt.subplot(2, 3, 4)
        cm = np.array(best_metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'], 
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix\n({best_threshold})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Plot 5: Metrics Comparison
        plt.subplot(2, 3, 5)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [best_metrics['accuracy'], best_metrics['precision'], 
                         best_metrics['recall'], best_metrics['f1_score']]
        
        bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.ylabel('Score')
        plt.title('Classification Metrics')
        plt.ylim(0, 1)
        plt.grid(True, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 6: Threshold Comparison
        plt.subplot(2, 3, 6)
        threshold_names = list(all_metrics.keys())
        f1_scores = [all_metrics[name]['f1_score'] for name in threshold_names]
        
        plt.bar(threshold_names, f1_scores, color='lightsteelblue')
        plt.ylabel('F1-Score')
        plt.title('F1-Score by Threshold Method')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig('model_artifacts/classification_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Step 8: Save comprehensive results
        logger.info("💾 Saving metrics results...")
        
        results = {
            'best_threshold_method': best_threshold,
            'best_metrics': best_metrics,
            'all_thresholds': all_metrics,
            'dataset_info': {
                'total_samples': len(all_true_labels),
                'normal_samples': int(np.sum(all_true_labels == 0)),
                'anomaly_samples': int(np.sum(all_true_labels == 1)),
                'normal_percentage': float(np.sum(all_true_labels == 0) / len(all_true_labels) * 100),
                'anomaly_percentage': float(np.sum(all_true_labels == 1) / len(all_true_labels) * 100)
            },
            'reconstruction_error_stats': {
                'mean': float(np.mean(all_reconstruction_errors)),
                'std': float(np.std(all_reconstruction_errors)),
                'min': float(np.min(all_reconstruction_errors)),
                'max': float(np.max(all_reconstruction_errors)),
                'normal_mean': float(np.mean(normal_errors)),
                'normal_std': float(np.std(normal_errors)),
                'anomaly_mean': float(np.mean(anomaly_errors)),
                'anomaly_std': float(np.std(anomaly_errors))
            },
            'model_info': {
                'input_dim': config.input_dim,
                'bottleneck_dim': config.bottleneck_dim,
                'total_parameters': model.count_parameters()
            }
        }
        
        results_path = Path("model_artifacts/classification_metrics.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)  # ✅ Add default=str to handle numpy types
        
        logger.info(f"✅ Results saved to: {results_path}")
        logger.info(f"✅ Visualizations saved to: model_artifacts/classification_metrics.png")
        
        # Step 9: Print summary
        logger.info("🎉 CLASSIFICATION METRICS SUMMARY:")
        logger.info("=" * 50)
        logger.info(f"🏆 Best Method: {best_threshold}")
        logger.info(f"📊 Threshold: {best_metrics['threshold']:.6f}")
        logger.info(f"🎯 Accuracy: {best_metrics['accuracy']:.4f}")
        logger.info(f"🎯 Precision: {best_metrics['precision']:.4f}")
        logger.info(f"🎯 Recall: {best_metrics['recall']:.4f}")
        logger.info(f"🎯 F1-Score: {best_metrics['f1_score']:.4f}")
        logger.info(f"🎯 ROC-AUC: {best_metrics['roc_auc']:.4f}")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = calculate_classification_metrics()
    
    if success:
        logger.info("🎉 Classification metrics calculated successfully!")
        logger.info("✅ Check model_artifacts/classification_metrics.json for detailed results")
        logger.info("✅ Check model_artifacts/classification_metrics.png for visualizations")
    else:
        logger.error("❌ Metrics calculation failed")
