#!/usr/bin/env python3
"""
True Full-Test Macro F1 Calculator
Properly computes end-to-end performance by combining Stage-1 and Stage-2 predictions
and calculating macro-F1 across all 6 classes (Normal + 5 attack types).

This is the CORRECT way to evaluate true end-to-end performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrueEndToEndEvaluator:
    """Computes True Full-Test Macro F1 by properly combining Stage-1 and Stage-2 predictions."""
    
    def __init__(self):
        self.output_dir = Path(__file__).parent / "AI" / "model_artifacts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class definitions
        self.classes = ['Normal', 'Botnet', 'DoS', 'Infiltration', 'Other', 'PortScan']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        logger.info("🎯 True End-to-End Evaluator initialized")
        logger.info(f"📊 Classes: {self.classes}")
    
    def load_seeded_results(self):
        """Load the actual seeded experiment results."""
        logger.info("📂 Loading seeded experiment results...")
        
        results_file = self.output_dir / "fixed_seeded_evaluation_detailed_20260213_001306.json"
        
        if not results_file.exists():
            logger.error(f"❌ Results file not found: {results_file}")
            return None
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        # The file contains a list of results, take the first one (they're all identical)
        if isinstance(data, list):
            seeded_results = data[0]
            logger.info(f"✅ Loaded seeded results from {len(data)} seeds (using first seed)")
        else:
            seeded_results = data
            logger.info("✅ Loaded single seeded result")
        
        return seeded_results
    
    def reconstruct_full_predictions(self, seeded_results):
        """
        Reconstruct the complete test set predictions by combining Stage-1 and Stage-2.
        
        Logic:
        1. Stage-1 predicts: Normal vs Anomaly
        2. For samples predicted as Anomaly → Stage-2 predicts attack type
        3. For samples predicted as Normal → Class stays as Normal
        4. Missed anomalies (false negatives) → Predicted as Normal (incorrectly)
        """
        logger.info("🔄 Reconstructing full test set predictions...")
        
        # Extract data from seeded results
        dataset_info = seeded_results['full_metrics']['dataset_info']
        stage1_metrics = seeded_results['full_metrics']['stage1_anomaly_detection']
        stage2_metrics = seeded_results['full_metrics']['stage2_attack_category_classification']['oracle_true_anomalies']
        end_to_end_metrics = seeded_results['full_metrics']['stage2_attack_category_classification']['end_to_end_detected_samples']
        
        # Get dataset sizes
        total_samples = dataset_info['total_samples']
        normal_samples = dataset_info['normal_samples']
        anomaly_samples = dataset_info['anomaly_samples']
        detected_anomalies = dataset_info['detected_anomalies']
        
        logger.info(f"📊 Dataset: {total_samples} total, {normal_samples} normal, {anomaly_samples} anomalies")
        logger.info(f"🔍 Stage-1 detected: {detected_anomalies} anomalies")
        
        # Reconstruct confusion matrices
        stage1_cm = np.array(stage1_metrics['confusion_matrix'])
        # Stage-1 CM format: [[TN, FP], [FN, TP]]
        tn, fp = stage1_cm[0]
        fn, tp = stage1_cm[1]
        
        logger.info(f"📈 Stage-1: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Stage-2 confusion matrix (for detected anomalies only)
        stage2_cm = np.array(end_to_end_metrics['confusion_matrix'])
        # Stage-2 CM format: 5x5 for attack types (Botnet, DoS, Infiltration, Other, PortScan)
        
        # Build full 6x6 confusion matrix
        full_cm = np.zeros((6, 6), dtype=int)
        
        # Fill in the confusion matrix
        
        # Row 0: True Normal samples
        # - True Normal predicted as Normal: TN
        # - True Normal predicted as attacks: FP (distributed across attack types)
        full_cm[0, 0] = tn  # True Normal -> Predicted Normal
        
        # Distribute FP across attack types proportionally to Stage-2 predictions
        if fp > 0:
            # Get Stage-2 prediction distribution for detected anomalies
            stage2_predictions = stage2_cm.sum(axis=0)  # Sum columns = predictions per class
            stage2_total = stage2_predictions.sum()
            
            if stage2_total > 0:
                # Proportionally distribute FP across attack types
                fp_distribution = (stage2_predictions / stage2_total) * fp
                for i, fp_count in enumerate(fp_distribution):
                    full_cm[0, i+1] = int(fp_count)  # True Normal -> Predicted Attack i+1
        
        # Rows 1-5: True Anomaly samples
        # For each true attack type:
        # - Correctly detected (TP) → Go through Stage-2 classification
        # - Missed (FN) → Predicted as Normal
        
        # Get attack type distribution from dataset
        attack_support = [
            stage2_metrics['classification_report']['Botnet']['support'],
            stage2_metrics['classification_report']['DoS']['support'],
            stage2_metrics['classification_report']['Infiltration']['support'],
            stage2_metrics['classification_report']['Other']['support'],
            stage2_metrics['classification_report']['PortScan']['support']
        ]
        
        # For each attack type (rows 1-5)
        for attack_idx in range(5):
            attack_type = self.classes[attack_idx + 1]
            true_count = attack_support[attack_idx]
            
            # Use Stage-2 confusion matrix for this attack type
            attack_row = stage2_cm[attack_idx]  # Predictions for this true attack type
            
            # These are the correctly detected anomalies that went through Stage-2
            detected_count = attack_row.sum()
            
            # Missed anomalies (FN) = true_count - detected_count
            missed_count = true_count - detected_count
            
            # Fill in full confusion matrix row
            full_cm[attack_idx + 1, 0] = missed_count  # True Attack -> Predicted Normal (missed)
            
            # Fill in Stage-2 predictions
            for pred_idx in range(5):
                full_cm[attack_idx + 1, pred_idx + 1] = attack_row[pred_idx]
        
        return full_cm, {
            'total_samples': total_samples,
            'normal_samples': normal_samples,
            'anomaly_samples': anomaly_samples,
            'detected_anomalies': detected_anomalies,
            'stage1_cm': stage1_cm.tolist(),
            'stage2_cm': stage2_cm.tolist()
        }
    
    def compute_true_macro_f1(self, full_cm):
        """Compute macro F1 from the full 6x6 confusion matrix."""
        logger.info("🧮 Computing True Macro F1 from full confusion matrix...")
        
        # Compute per-class precision, recall, F1
        per_class_metrics = {}
        
        for i, class_name in enumerate(self.classes):
            tp = full_cm[i, i]
            fp = full_cm[:, i].sum() - tp
            fn = full_cm[i, :].sum() - tp
            tn = full_cm.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': full_cm[i, :].sum(),
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
        
        # Compute macro averages
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in per_class_metrics.values()])
        
        # Compute weighted averages
        total_support = sum(m['support'] for m in per_class_metrics.values())
        weighted_precision = sum(m['precision'] * m['support'] for m in per_class_metrics.values()) / total_support
        weighted_recall = sum(m['recall'] * m['support'] for m in per_class_metrics.values()) / total_support
        weighted_f1 = sum(m['f1_score'] * m['support'] for m in per_class_metrics.values()) / total_support
        
        # Compute overall accuracy
        overall_accuracy = full_cm.diagonal().sum() / full_cm.sum()
        
        results = {
            'per_class_metrics': per_class_metrics,
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1_score': weighted_f1
            },
            'overall_accuracy': overall_accuracy,
            'total_samples': full_cm.sum()
        }
        
        logger.info(f"✅ True Macro F1: {macro_f1:.4f}")
        logger.info(f"📊 Weighted F1: {weighted_f1:.4f}")
        logger.info(f"🎯 Overall Accuracy: {overall_accuracy:.4f}")
        
        return results, full_cm
    
    def create_confusion_matrix_visualization(self, full_cm, results):
        """Create a comprehensive visualization of the full confusion matrix."""
        logger.info("📈 Creating confusion matrix visualization...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Full Confusion Matrix (Normalized)
        cm_normalized = full_cm.astype('float') / full_cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.classes, yticklabels=self.classes, ax=ax1)
        ax1.set_title('Full Confusion Matrix (Normalized)\nTrue End-to-End Performance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # 2. Per-Class F1 Scores
        class_names = list(results['per_class_metrics'].keys())
        f1_scores = [results['per_class_metrics'][cls]['f1_score'] for cls in class_names]
        
        bars = ax2.bar(class_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
        ax2.set_title('Per-Class F1 Scores\n(True End-to-End)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # 3. Macro vs Weighted Comparison
        metrics = ['Precision', 'Recall', 'F1-Score']
        macro_vals = [results['macro_avg']['precision'], results['macro_avg']['recall'], results['macro_avg']['f1_score']]
        weighted_vals = [results['weighted_avg']['precision'], results['weighted_avg']['recall'], results['weighted_avg']['f1_score']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax3.bar(x - width/2, macro_vals, width, label='Macro Avg', color='skyblue')
        ax3.bar(x + width/2, weighted_vals, width, label='Weighted Avg', color='lightcoral')
        
        ax3.set_title('Macro vs Weighted Averages', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. Summary Statistics
        ax4.axis('off')
        
        summary_text = f"""
TRUE END-TO-END PERFORMANCE SUMMARY

📊 Overall Metrics:
• Accuracy: {results['overall_accuracy']:.4f}
• Total Samples: {results['total_samples']:,}

🎯 Macro Averages (Equal Weight):
• Precision: {results['macro_avg']['precision']:.4f}
• Recall: {results['macro_avg']['recall']:.4f}
• F1-Score: {results['macro_avg']['f1_score']:.4f}

⚖️ Weighted Averages (Support Weighted):
• Precision: {results['weighted_avg']['precision']:.4f}
• Recall: {results['weighted_avg']['recall']:.4f}
• F1-Score: {results['weighted_avg']['f1_score']:.4f}

📈 Per-Class Performance:
"""
        for class_name in self.classes:
            metrics = results['per_class_metrics'][class_name]
            summary_text += f"• {class_name}: F1={metrics['f1_score']:.3f}, Support={metrics['support']:,}\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the figure
        output_file = self.output_dir / "true_end_to_end_confusion_matrix.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Confusion matrix visualization saved to: {output_file}")
        return output_file
    
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def save_results(self, results, full_cm, reconstruction_info):
        """Save the complete true end-to-end results."""
        logger.info("💾 Saving true end-to-end results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare results data with numpy types converted
        results_data = {
            'evaluation_type': 'True End-to-End Full Test Evaluation',
            'timestamp': timestamp,
            'description': 'Properly computed end-to-end performance combining Stage-1 and Stage-2 predictions',
            'classes': self.classes,
            'reconstruction_info': reconstruction_info,
            'full_confusion_matrix': full_cm.tolist(),
            'results': self.convert_numpy_types(results),
            'key_findings': {
                'true_macro_f1': float(results['macro_avg']['f1_score']),
                'true_weighted_f1': float(results['weighted_avg']['f1_score']),
                'overall_accuracy': float(results['overall_accuracy']),
                'normal_class_f1': float(results['per_class_metrics']['Normal']['f1_score']),
                'best_attack_class': max(self.classes[1:], key=lambda x: results['per_class_metrics'][x]['f1_score']),
                'worst_attack_class': min(self.classes[1:], key=lambda x: results['per_class_metrics'][x]['f1_score'])
            }
        }
        
        # Save detailed results
        results_file = self.output_dir / f"true_end_to_end_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"✅ Results saved to: {results_file}")
        
        # Create summary report
        summary_file = self.output_dir / f"true_end_to_end_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("TRUE END-TO-END FULL TEST EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Evaluation Type: Proper end-to-end with 6 classes (Normal + 5 attacks)\n\n")
            
            f.write("KEY RESULTS:\n")
            f.write(f"• True Macro F1: {results['macro_avg']['f1_score']:.4f}\n")
            f.write(f"• True Weighted F1: {results['weighted_avg']['f1_score']:.4f}\n")
            f.write(f"• Overall Accuracy: {results['overall_accuracy']:.4f}\n\n")
            
            f.write("PER-CLASS PERFORMANCE:\n")
            for class_name in self.classes:
                metrics = results['per_class_metrics'][class_name]
                f.write(f"• {class_name:12}: F1={metrics['f1_score']:.3f}, P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, Support={metrics['support']:6,}\n")
            
            f.write(f"\nFULL CONFUSION MATRIX:\n")
            f.write("Predicted ->\n")
            f.write("True v     " + "  ".join(f"{cls:8}" for cls in self.classes) + "\n")
            for i, row in enumerate(full_cm):
                f.write(f"{self.classes[i]:8} " + "  ".join(f"{val:8}" for val in row) + "\n")
        
        logger.info(f"📄 Summary saved to: {summary_file}")
        
        return results_file, summary_file
    
    def evaluate_true_end_to_end(self):
        """Main evaluation pipeline."""
        logger.info("🚀 Starting True End-to-End Evaluation...")
        logger.info("=" * 80)
        
        # Load seeded results
        seeded_results = self.load_seeded_results()
        if not seeded_results:
            logger.error("❌ Failed to load seeded results")
            return None
        
        # Reconstruct full predictions
        full_cm, reconstruction_info = self.reconstruct_full_predictions(seeded_results)
        
        # Compute true macro F1
        results, full_cm = self.compute_true_macro_f1(full_cm)
        
        # Create visualization
        viz_file = self.create_confusion_matrix_visualization(full_cm, results)
        
        # Save results
        results_file, summary_file = self.save_results(results, full_cm, reconstruction_info)
        
        # Final summary
        logger.info("=" * 80)
        logger.info("🎯 TRUE END-TO-END EVALUATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"📊 True Macro F1: {results['macro_avg']['f1_score']:.4f}")
        logger.info(f"⚖️ True Weighted F1: {results['weighted_avg']['f1_score']:.4f}")
        logger.info(f"🎯 Overall Accuracy: {results['overall_accuracy']:.4f}")
        logger.info("=" * 80)
        logger.info("📁 Generated Files:")
        logger.info(f"   • {viz_file.name}")
        logger.info(f"   • {results_file.name}")
        logger.info(f"   • {summary_file.name}")
        logger.info("=" * 80)
        
        return results

def main():
    """Main execution function."""
    logger.info("🎓 TRUE END-TO-END MACRO F1 CALCULATOR")
    logger.info("=" * 80)
    logger.info("📄 Properly computes end-to-end performance:")
    logger.info("   • Combines Stage-1 and Stage-2 predictions")
    logger.info("   • Handles missed anomalies as Normal predictions")
    logger.info("   • Computes macro-F1 across 6 classes (Normal + 5 attacks)")
    logger.info("   • This is the CORRECT true end-to-end evaluation")
    logger.info("=" * 80)
    
    evaluator = TrueEndToEndEvaluator()
    results = evaluator.evaluate_true_end_to_end()
    
    if results:
        logger.info("\n🎉 True End-to-End evaluation completed!")
        logger.info("🔍 This is the REAL end-to-end performance metric!")
    else:
        logger.error("❌ Evaluation failed!")

if __name__ == "__main__":
    main()
