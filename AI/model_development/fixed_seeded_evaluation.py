#!/usr/bin/env python3
"""
Fixed Seeded Evaluation Script
Uses existing metrics calculation with different seeds for robust evaluation.

Usage:
    python fixed_seeded_evaluation.py
"""

import os
import sys
import json
import csv
import logging
from datetime import datetime
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "model_development"))

# PyTorch and ML imports
import torch
import numpy as np
import random

# Import existing modules
try:
    from calculate_metrics import calculate_two_stage_classification_metrics
    EVALUATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Evaluation modules not available: {e}")
    EVALUATION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FixedSeededEvaluator:
    """Fixed evaluation using existing metrics function with different seeds."""
    
    def __init__(self):
        self.seeds = [42, 123, 999]
        self.results = []
        self.output_dir = Path(__file__).parent.parent / "model_artifacts"
        self.output_dir.mkdir(exist_ok=True)
        
    def set_seed(self, seed):
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Set CUDA seed if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"🎲 Set evaluation seed to {seed}")
    
    def evaluate_with_seed(self, seed):
        """Evaluate models with given seed using existing metrics function."""
        logger.info(f"🚀 Starting evaluation with seed {seed}")
        
        try:
            # Set seed before evaluation
            self.set_seed(seed)
            
            logger.info("📊 Running two-stage metrics calculation...")
            
            # Use existing metrics calculation function
            success = calculate_two_stage_classification_metrics()
            
            if not success:
                logger.error("❌ Metrics calculation returned False")
                return None
            
            # Load the saved metrics from JSON file
            metrics_file = self.output_dir / "two_stage_classification_metrics.json"
            if not metrics_file.exists():
                logger.error("❌ Metrics file not found after calculation")
                return None
                
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            # Extract key metrics from the existing structure
            stage1_metrics = metrics.get('stage1_anomaly_detection', {})
            stage2_metrics = metrics.get('stage2_attack_category_classification', {})
            
            # Extract the values we need
            stage1_recall = stage1_metrics.get('recall', 0.0)
            stage2_oracle_acc = stage2_metrics.get('oracle_true_anomalies', {}).get('accuracy', 0.0)
            true_end_to_end_f1 = stage2_metrics.get('end_to_end_detected_samples', {}).get('f1_macro', 0.0)
            
            experiment_result = {
                'seed': seed,
                'stage1_recall': stage1_recall,
                'stage2_oracle_accuracy': stage2_oracle_acc,
                'true_end_to_end_f1': true_end_to_end_f1,
                'timestamp': datetime.now().isoformat(),
                'full_metrics': metrics
            }
            
            logger.info(f"✅ Seed {seed} evaluation completed:")
            logger.info(f"   Stage-1 Recall: {stage1_recall:.4f}")
            logger.info(f"   Stage-2 Oracle Acc: {stage2_oracle_acc:.4f} ({stage2_oracle_acc*100:.2f}%)")
            logger.info(f"   True End-to-End F1: {true_end_to_end_f1:.4f}")
            
            return experiment_result
            
        except Exception as e:
            logger.error(f"❌ Evaluation with seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_evaluations(self):
        """Run evaluations with all seeds."""
        logger.info("🎯 Starting seeded evaluations...")
        logger.info(f"📋 Seeds to test: {self.seeds}")
        
        if not EVALUATION_AVAILABLE:
            logger.error("❌ Evaluation modules not available. Cannot proceed.")
            return
        
        for i, seed in enumerate(self.seeds, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 Evaluation {i}/{len(self.seeds)}")
            logger.info(f"{'='*60}")
            
            result = self.evaluate_with_seed(seed)
            if result:
                self.results.append(result)
            else:
                logger.error(f"❌ Failed to get results for seed {seed}")
        
        # Calculate summary statistics
        self.calculate_summary_statistics()
        
        # Save results
        self.save_results()
        
        logger.info("🎉 All evaluations completed!")
    
    def calculate_summary_statistics(self):
        """Calculate mean and std for key metrics."""
        if len(self.results) < 2:
            logger.warning("⚠️ Need at least 2 successful runs for statistics")
            return
        
        metrics = ['stage1_recall', 'stage2_oracle_accuracy', 'true_end_to_end_f1']
        
        logger.info(f"\n📊 Summary Statistics:")
        logger.info(f"{'='*50}")
        
        for metric in metrics:
            values = [r[metric] for r in self.results if r and metric in r]
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Format for display
                if metric == 'stage2_oracle_accuracy':
                    mean_fmt = f"{mean_val*100:.2f}%"
                    std_fmt = f"{std_val*100:.2f}%"
                    metric_name = "Stage-2 Oracle Accuracy"
                elif metric == 'stage1_recall':
                    mean_fmt = f"{mean_val:.3f}"
                    std_fmt = f"{std_val:.3f}"
                    metric_name = "Stage-1 Recall"
                else:  # true_end_to_end_f1
                    mean_fmt = f"{mean_val:.3f}"
                    std_fmt = f"{std_val:.3f}"
                    metric_name = "True End-to-End F1"
                
                logger.info(f"{metric_name}:")
                logger.info(f"   Mean ± Std: {mean_fmt} ± {std_fmt}")
    
    def save_results(self):
        """Save results to CSV and JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        json_file = self.output_dir / f"fixed_seeded_evaluation_detailed_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"💾 Detailed results saved to: {json_file}")
        
        # Save summary as CSV
        csv_file = self.output_dir / f"fixed_seeded_evaluation_summary_{timestamp}.csv"
        
        # Prepare CSV data
        csv_data = []
        for result in self.results:
            if result:
                csv_data.append({
                    'Run': len(csv_data) + 1,
                    'Seed': result['seed'],
                    'Stage-1 Recall': f"{result['stage1_recall']:.3f}",
                    'Stage-2 Oracle Acc': f"{result['stage2_oracle_accuracy']*100:.2f}%",
                    'True End-to-End F1': f"{result['true_end_to_end_f1']:.3f}"
                })
        
        # Write CSV
        if csv_data:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
                writer.writeheader()
                writer.writerows(csv_data)
            logger.info(f"📊 Summary CSV saved to: {csv_file}")
            
            # Display table
            self.display_results_table(csv_data)
    
    def display_results_table(self, csv_data):
        """Display results in a formatted table."""
        logger.info(f"\n📋 Results Table:")
        logger.info(f"{'='*70}")
        
        # Header
        header = f"{'Run':<5} {'Seed':<6} {'Stage-1 Recall':<15} {'Stage-2 Acc':<13} {'True F1':<10}"
        logger.info(header)
        logger.info('-'*70)
        
        # Data rows
        for row in csv_data:
            line = f"{row['Run']:<5} {row['Seed']:<6} {row['Stage-1 Recall']:<15} {row['Stage-2 Oracle Acc']:<13} {row['True End-to-End F1']:<10}"
            logger.info(line)
        
        logger.info('='*70)

def main():
    """Main execution function."""
    logger.info("🎲 Fixed Seeded Evaluation for Two-Stage Anomaly Detection")
    logger.info("="*60)
    logger.info("📝 Using existing trained models with different evaluation seeds")
    
    evaluator = FixedSeededEvaluator()
    evaluator.run_all_evaluations()
    
    logger.info("\n🏁 Evaluation completed!")
    logger.info("📁 Check model_artifacts directory for detailed results")

if __name__ == "__main__":
    main()
