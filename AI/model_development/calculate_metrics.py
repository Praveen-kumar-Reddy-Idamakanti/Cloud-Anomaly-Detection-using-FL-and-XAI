"""
ENHANCED Calculate Classification Metrics for Two-Stage System
Convert reconstruction errors to anomaly detection performance metrics
PLUS attack type classification metrics for detected anomalies
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
from train import FixedDataPreparation, FixedAutoencoderTrainer, AttackTypeClassifier
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_preprocessing.lookup_functions.attack_type_lookup import get_attack_type, get_attack_type_id

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "model_artifacts"


def calculate_two_stage_classification_metrics():
    """Calculate comprehensive classification metrics for the ENHANCED two-stage system"""
    logger.info("🎯 Calculating ENHANCED Two-Stage Classification Metrics...")
    
    try:
        # Step 1: Load trained models (both stages)
        logger.info("📥 Loading trained models...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the best checkpoint
        checkpoint_path = ARTIFACTS_DIR / "latest_checkpoint_fixed.pth"
        if not checkpoint_path.exists():
            logger.error("❌ Trained models not found! Using existing model artifacts...")
            # Try to use the existing model
            return calculate_single_stage_metrics()
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config_dict = checkpoint['config']
        
        # Check if two-stage models are available
        two_stage_enabled = checkpoint.get('two_stage_enabled', False)
        if not two_stage_enabled:
            logger.warning("⚠️ Two-stage models not found in checkpoint. Using single-stage metrics.")
            return calculate_single_stage_metrics()
        
        # Reconstruct config with ACTUAL dimensions from checkpoint
        config = AutoencoderConfig()
        
        # Get actual input dimension from model weights
        encoder_first_weight = checkpoint['model_state_dict']['encoder.0.weight']
        actual_input_dim = encoder_first_weight.shape[1]  # Get actual input dim from weights
        
        config.input_dim = actual_input_dim
        config.encoding_dims = config_dict['encoding_dims']
        config.bottleneck_dim = config_dict['bottleneck_dim']
        config.dropout_rate = config_dict['dropout_rate']
        
        logger.info(f"✅ Using input dimension: {config.input_dim}")
        
        # Load Stage 1: Autoencoder
        from autoencoder_model import CloudAnomalyAutoencoder
        autoencoder = CloudAnomalyAutoencoder(
            input_dim=config.input_dim,
            encoding_dims=config.encoding_dims,
            bottleneck_dim=config.bottleneck_dim,
            dropout_rate=config.dropout_rate
        ).to(device)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        autoencoder.eval()
        
        # Load Stage 2: Attack Category Classifier
        attack_types = checkpoint.get('attack_types')
        if not attack_types:
            attack_types = checkpoint.get('attack_category_classes')
        if not attack_types:
            attack_types = ['DoS', 'PortScan', 'WebAttack', 'Infiltration', 'Botnet', 'BruteForce', 'Heartbleed', 'Other']

        attack_classifier = AttackTypeClassifier(
            input_dim=config.input_dim,
            num_classes=len(attack_types)
        ).to(device)
        attack_classifier.load_state_dict(checkpoint['attack_classifier_state_dict'])
        attack_classifier.eval()
        
        logger.info(f"✅ Two-stage models loaded:")
        logger.info(f"  Autoencoder: {config.input_dim} → {config.bottleneck_dim} → {config.input_dim}")
        logger.info(f"  Attack Classifier: {config.input_dim} → {len(attack_types)} attack categories")
        logger.info(f"  Attack types: {attack_types}")
        
        # Step 2: Load test data
        logger.info("📊 Loading test data...")
        data_prep = FixedDataPreparation()
        data_results = data_prep.prepare_data(batch_size=128)
        test_loader = data_results.get('test_two_stage_loader') or data_results['test_loader']
        
        logger.info(f"✅ Test data loaded: {len(test_loader)} batches")
        
        # Step 3: Get two-stage predictions
        logger.info("🔍 Getting two-stage model predictions...")
        
        # Initialize trainer for prediction
        trainer = FixedAutoencoderTrainer(
            config,
            attack_category_classes=attack_types,
            attack_category_encoder=None,
        )
        trainer.model = autoencoder
        trainer.attack_classifier = attack_classifier
        trainer.attack_types = attack_types
        
        # Collect predictions
        all_reconstruction_errors = []
        all_true_labels = []
        all_true_categories = []
        all_anomaly_predictions = []
        all_attack_predictions = []
        all_attack_confidences = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    features, labels, category_labels = batch
                else:
                    features, labels = batch
                    category_labels = None
                # Get two-stage predictions
                results = trainer.predict_two_stage(features)
                
                all_reconstruction_errors.extend(results['reconstruction_errors'])
                all_true_labels.extend(labels.cpu().numpy())
                if category_labels is not None:
                    all_true_categories.extend(category_labels.cpu().numpy())
                all_anomaly_predictions.extend(results['anomaly_predictions'])
                all_attack_predictions.extend(results['attack_type_predictions'])
                all_attack_confidences.extend(results['attack_confidences'])
        
        all_reconstruction_errors = np.array(all_reconstruction_errors)
        all_true_labels = np.array(all_true_labels)
        all_true_categories = np.array(all_true_categories) if all_true_categories else None
        all_anomaly_predictions = np.array(all_anomaly_predictions)
        all_attack_predictions = np.array(all_attack_predictions)
        all_attack_confidences = np.array(all_attack_confidences)
        
        logger.info(f"✅ Processed {len(all_reconstruction_errors)} test samples")
        logger.info(f"  Normal samples: {np.sum(all_true_labels == 0)}")
        logger.info(f"  Anomaly samples: {np.sum(all_true_labels == 1)}")
        logger.info(f"  Detected anomalies: {np.sum(all_anomaly_predictions == 1)}")
        
        # Step 4: Stage 1 - Anomaly Detection Metrics
        logger.info("📈 Stage 1: Calculating Anomaly Detection Metrics...")
        
        # Calculate binary classification metrics
        accuracy = accuracy_score(all_true_labels, all_anomaly_predictions)
        precision = precision_score(all_true_labels, all_anomaly_predictions, zero_division=0)
        recall = recall_score(all_true_labels, all_anomaly_predictions, zero_division=0)
        f1 = f1_score(all_true_labels, all_anomaly_predictions, zero_division=0)
        
        # ROC-AUC
        try:
            roc_auc = roc_auc_score(all_true_labels, all_reconstruction_errors)
        except:
            roc_auc = 0.5
        
        stage1_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(all_true_labels, all_anomaly_predictions).tolist()
        }
        
        logger.info(f"🎯 Stage 1 Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        
        # Step 5: Stage 2 - Attack Category Classification Metrics
        logger.info("📈 Stage 2: Calculating Attack Category Classification Metrics...")

        stage2_metrics = {
            'oracle_true_anomalies': None,
            'end_to_end_detected_samples': None,
        }

        if all_true_categories is None:
            logger.warning("⚠️ Two-stage category labels not available; cannot compute stage-2 metrics")
        else:
            # (A) Oracle evaluation on true anomalies (independent of stage-1)
            true_anomaly_mask = all_true_labels == 1
            X_true_anom = []
            y_true_cat = []
            with torch.no_grad():
                for batch in test_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        features, labels, category_labels = batch
                    else:
                        continue
                    mask = labels == 1
                    if mask.any():
                        X_true_anom.append(features[mask].to(device))
                        y_true_cat.append(category_labels[mask].cpu().numpy())

            if X_true_anom:
                X_true_anom = torch.cat(X_true_anom, dim=0)
                y_true_cat = np.concatenate(y_true_cat)
                outputs = attack_classifier(X_true_anom)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

                stage2_metrics['oracle_true_anomalies'] = {
                    'accuracy': float(accuracy_score(y_true_cat, preds)),
                    'precision_macro': float(precision_score(y_true_cat, preds, average='macro', zero_division=0)),
                    'recall_macro': float(recall_score(y_true_cat, preds, average='macro', zero_division=0)),
                    'f1_macro': float(f1_score(y_true_cat, preds, average='macro', zero_division=0)),
                    'precision_weighted': float(precision_score(y_true_cat, preds, average='weighted', zero_division=0)),
                    'recall_weighted': float(recall_score(y_true_cat, preds, average='weighted', zero_division=0)),
                    'f1_weighted': float(f1_score(y_true_cat, preds, average='weighted', zero_division=0)),
                    'support': int(len(y_true_cat)),
                    'confusion_matrix': confusion_matrix(y_true_cat, preds).tolist(),
                    'classification_report': classification_report(
                        y_true_cat,
                        preds,
                        labels=list(range(len(attack_types))),
                        target_names=attack_types,
                        output_dict=True,
                        zero_division=0,
                    ),
                }

            # (B) End-to-end evaluation on samples passed as anomalies by stage-1
            detected_mask = (all_anomaly_predictions == 1) & (all_true_labels == 1)
            if np.sum(detected_mask) > 0:
                y_true_detected = all_true_categories[detected_mask]
                y_pred_detected = all_attack_predictions[detected_mask]
                stage2_metrics['end_to_end_detected_samples'] = {
                    'accuracy': float(accuracy_score(y_true_detected, y_pred_detected)),
                    'precision_macro': float(precision_score(y_true_detected, y_pred_detected, average='macro', zero_division=0)),
                    'recall_macro': float(recall_score(y_true_detected, y_pred_detected, average='macro', zero_division=0)),
                    'f1_macro': float(f1_score(y_true_detected, y_pred_detected, average='macro', zero_division=0)),
                    'precision_weighted': float(precision_score(y_true_detected, y_pred_detected, average='weighted', zero_division=0)),
                    'recall_weighted': float(recall_score(y_true_detected, y_pred_detected, average='weighted', zero_division=0)),
                    'f1_weighted': float(f1_score(y_true_detected, y_pred_detected, average='weighted', zero_division=0)),
                    'support': int(np.sum(detected_mask)),
                    'confusion_matrix': confusion_matrix(y_true_detected, y_pred_detected).tolist(),
                }
        
        # Step 6: Create enhanced visualizations
        logger.info("📊 Creating two-stage visualizations...")
        
        plt.figure(figsize=(20, 12))
        
        # Plot 1: ROC Curve (Stage 1)
        plt.subplot(2, 4, 1)
        fpr, tpr, _ = roc_curve(all_true_labels, all_reconstruction_errors)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Stage 1: ROC Curve')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Confusion Matrix (Stage 1)
        plt.subplot(2, 4, 2)
        cm1 = np.array(stage1_metrics['confusion_matrix'])
        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'], 
                   yticklabels=['Normal', 'Anomaly'])
        plt.title('Stage 1: Anomaly Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Plot 3: Stage 1 Metrics
        plt.subplot(2, 4, 3)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [stage1_metrics['accuracy'], stage1_metrics['precision'], 
                         stage1_metrics['recall'], stage1_metrics['f1_score']]
        
        bars = plt.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        plt.ylabel('Score')
        plt.title('Stage 1: Anomaly Detection Metrics')
        plt.ylim(0, 1)
        plt.grid(True, axis='y')
        
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 4: Reconstruction Error Distribution
        plt.subplot(2, 4, 4)
        normal_errors = all_reconstruction_errors[all_true_labels == 0]
        anomaly_errors = all_reconstruction_errors[all_true_labels == 1]
        
        plt.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color='blue', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color='red', density=True)
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Density')
        plt.title('Reconstruction Error Distribution')
        plt.legend()
        plt.grid(True)
        
        # Plot 5: Attack Category Distribution (Stage 2)
        detected_anomalies = all_attack_predictions[all_anomaly_predictions == 1]
        if len(detected_anomalies) > 0:
            plt.subplot(2, 4, 5)
            attack_counts = [np.sum(detected_anomalies == i) for i in range(len(attack_types))]
            plt.bar(attack_types, attack_counts, color='lightcoral')
            plt.ylabel('Count')
            plt.title('Stage 2: Detected Attack Categories')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
        
        # Plot 6: Attack Confidence Distribution
        detected_mask_plot = all_anomaly_predictions == 1
        if len(detected_anomalies) > 0:
            plt.subplot(2, 4, 6)
            plt.hist(all_attack_confidences[detected_mask_plot], bins=20, alpha=0.7, color='orange')
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
            plt.title('Stage 2: Attack Classification Confidence')
            plt.grid(True)
        
        # Plot 7: Stage 2 Metrics (if available)
        if stage2_metrics.get('end_to_end_detected_samples'):
            plt.subplot(2, 4, 7)
            attack_metric_names = ['Acc', 'P(macro)', 'R(macro)', 'F1(macro)']
            vals = stage2_metrics['end_to_end_detected_samples']
            attack_metric_values = [vals['accuracy'], vals['precision_macro'], vals['recall_macro'], vals['f1_macro']]
            
            bars = plt.bar(attack_metric_names, attack_metric_values, color=['lightgreen', 'lightblue', 'lightyellow', 'lightpink'])
            plt.ylabel('Score')
            plt.title('Stage 2: Attack Category Metrics (End-to-End)')
            plt.ylim(0, 1)
            plt.grid(True, axis='y')
            
            for bar, value in zip(bars, attack_metric_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 8: Two-Stage Summary
        plt.subplot(2, 4, 8)
        stages = ['Stage 1\n(Anomaly)', 'Stage 2\n(Attack Category)']
        stage2_f1 = 0.0
        if stage2_metrics.get('end_to_end_detected_samples'):
            stage2_f1 = stage2_metrics['end_to_end_detected_samples']['f1_macro']
        f1_scores = [stage1_metrics['f1_score'], stage2_f1]
        colors = ['blue', 'red']
        
        bars = plt.bar(stages, f1_scores, color=colors, alpha=0.7)
        plt.ylabel('F1-Score')
        plt.title('Two-Stage System Performance')
        plt.ylim(0, 1)
        plt.grid(True, axis='y')
        
        for bar, value in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(ARTIFACTS_DIR / 'two_stage_classification_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Step 7: Save comprehensive results
        logger.info("💾 Saving two-stage metrics results...")
        
        results = {
            'two_stage_enabled': True,
            'stage1_anomaly_detection': stage1_metrics,
            'stage2_attack_category_classification': stage2_metrics,
            'dataset_info': {
                'total_samples': int(len(all_true_labels)),
                'normal_samples': int(np.sum(all_true_labels == 0)),
                'anomaly_samples': int(np.sum(all_true_labels == 1)),
                'detected_anomalies': int(np.sum(all_anomaly_predictions == 1)),
                'attack_classifications': int(len(detected_anomalies)),
            },
            'model_info': {
                'autoencoder_params': int(autoencoder.count_parameters()),
                'attack_classifier_params': int(attack_classifier.count_parameters()),
                'total_params': int(autoencoder.count_parameters() + attack_classifier.count_parameters()),
                'attack_categories': attack_types,
            },
            'reconstruction_error_stats': {
                'mean': float(np.mean(all_reconstruction_errors)),
                'std': float(np.std(all_reconstruction_errors)),
                'min': float(np.min(all_reconstruction_errors)),
                'max': float(np.max(all_reconstruction_errors)),
            },
        }
        
        results_path = ARTIFACTS_DIR / "two_stage_classification_metrics.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Two-stage results saved to: {results_path}")
        logger.info(f"✅ Visualizations saved to: {ARTIFACTS_DIR / 'two_stage_classification_metrics.png'}")
        
        # Step 8: Print comprehensive summary
        logger.info("🎉 TWO-STAGE CLASSIFICATION METRICS SUMMARY:")
        logger.info("=" * 60)
        logger.info(f"� STAGE 1 - Anomaly Detection:")
        logger.info(f"   Accuracy: {stage1_metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {stage1_metrics['precision']:.4f}")
        logger.info(f"   Recall: {stage1_metrics['recall']:.4f}")
        logger.info(f"   F1-Score: {stage1_metrics['f1_score']:.4f}")
        logger.info(f"   ROC-AUC: {stage1_metrics['roc_auc']:.4f}")
        logger.info("🎯 STAGE 2 - Attack Category Classification:")
        oracle = stage2_metrics.get('oracle_true_anomalies')
        e2e = stage2_metrics.get('end_to_end_detected_samples')
        if oracle:
            logger.info(f"   Oracle (true anomalies) - Acc: {oracle['accuracy']:.4f}, F1(macro): {oracle['f1_macro']:.4f}, Support: {oracle['support']}")
        else:
            logger.info("   Oracle (true anomalies): N/A")
        if e2e:
            logger.info(f"   End-to-End (detected & true anomalies) - Acc: {e2e['accuracy']:.4f}, F1(macro): {e2e['f1_macro']:.4f}, Support: {e2e['support']}")
        else:
            logger.info("   End-to-End: N/A")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Two-stage metrics calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def calculate_single_stage_metrics():
    """Fallback to single-stage metrics if two-stage not available"""
    logger.info("🔄 Calculating single-stage metrics from existing model...")
    
    try:
        # Load existing model and calculate metrics
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the autoencoder model
        model_path = ARTIFACTS_DIR / "best_autoencoder_fixed.pth"
        if not model_path.exists():
            logger.info("📂 Looking for available model files...")
            # List available model files in parent directory
            model_files = list(ARTIFACTS_DIR.glob("*.pth"))
            if model_files:
                logger.info(f"✅ Found model files: {[f.name for f in model_files]}")
                model_path = model_files[0]  # Use first available
            else:
                logger.error("❌ No trained model found!")
                return False
            
        # Load model info
        with open(ARTIFACTS_DIR / "classification_metrics.json", 'r') as f:
            existing_metrics = json.load(f)
        
        logger.info("✅ Using existing classification metrics:")
        logger.info(f"   Accuracy: {existing_metrics['best_metrics']['accuracy']:.4f}")
        logger.info(f"   Precision: {existing_metrics['best_metrics']['precision']:.4f}")
        logger.info(f"   Recall: {existing_metrics['best_metrics']['recall']:.4f}")
        logger.info(f"   F1-Score: {existing_metrics['best_metrics']['f1_score']:.4f}")
        logger.info(f"   ROC-AUC: {existing_metrics['best_metrics']['roc_auc']:.4f}")
        
        # Display confusion matrix
        cm = existing_metrics['best_metrics']['confusion_matrix']
        logger.info(f"   Confusion Matrix:")
        logger.info(f"     True Normal: {cm[0][0]}, False Anomaly: {cm[0][1]}")
        logger.info(f"     False Normal: {cm[1][0]}, True Anomaly: {cm[1][1]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Single-stage metrics calculation failed: {e}")
        return False


if __name__ == "__main__":
    success = calculate_two_stage_classification_metrics()
    
    if success:
        logger.info("🎉 ENHANCED Two-Stage classification metrics calculated successfully!")
        logger.info(f"✅ Check {ARTIFACTS_DIR / 'two_stage_classification_metrics.json'} for detailed results")
        logger.info(f"✅ Check {ARTIFACTS_DIR / 'two_stage_classification_metrics.png'} for visualizations")
    else:
        logger.error("❌ Two-stage metrics calculation failed")
