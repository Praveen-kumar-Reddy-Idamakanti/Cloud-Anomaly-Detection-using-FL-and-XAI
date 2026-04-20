#!/usr/bin/env python3
"""
Seeded Experiments Script
Runs the complete two-stage pipeline with different random seeds for robust evaluation.

Usage:
    python run_seeded_experiments.py

Output:
    CSV file with metrics for each seed run
    Summary statistics (mean ± std)
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

# Import existing training modules
try:
    from train import FixedAutoencoderTrainer, AttackTypeClassifier, FixedDataPreparation, AutoencoderConfig
    from calculate_metrics import calculate_two_stage_classification_metrics
    TRAINING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Training modules not available: {e}")
    TRAINING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SeededExperimentRunner:
    """Run complete two-stage pipeline with different random seeds."""
    
    def __init__(self):
        # Research-standard seeds for reproducibility
        self.seeds = [42, 123, 999, 2024, 777]  # 5 seeds for robust statistics
        self.results = []
        self.output_dir = Path(__file__).parent.parent / "model_artifacts"
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("🎲 Seeded Experiment Runner Initialized")
        logger.info(f"📋 Will test {len(self.seeds)} seeds: {self.seeds}")
        logger.info(f"📁 Results will be saved to: {self.output_dir}")
        
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
        
        logger.info(f"🎲 Set random seed to {seed}")
    
    def run_single_experiment(self, seed):
        """Run complete pipeline with given seed."""
        logger.info(f"🚀 Starting experiment with seed {seed}")
        
        try:
            # Set seed BEFORE training (critical!)
            self.set_seed(seed)
            
            # CRITICAL: Create fresh trainer for each seed
            # This ensures model weights are reinitialized with new random seed
            if not TRAINING_AVAILABLE:
                logger.error("❌ Training modules not available")
                return None
                
            logger.info(f"🔄 Initializing fresh trainer for seed {seed}")
            trainer = FixedAutoencoderTrainer()
            
            # Run complete two-stage training
            logger.info("🎯 Training Complete Two-Stage Pipeline")
            
            # Step 1: Create data preparation
            logger.info("📊 Preparing training data...")
            data_prep = FixedDataPreparation()
            data_results = data_prep.prepare_data(batch_size=trainer.config.batch_size)
            train_loader = data_results['train_loader']
            val_loader = data_results['val_loader']
            
            # Step 2: Initialize trainer with data preparation info
            trainer_with_data = FixedAutoencoderTrainer(
                config=trainer.config,
                attack_category_classes=getattr(data_prep, 'attack_category_classes', None),
                attack_category_encoder=getattr(data_prep, 'attack_category_encoder', None),
            )
            
            # Step 3: Train the model
            training_results = trainer_with_data.train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                data_prep=data_prep
            )
            
            # Calculate comprehensive metrics
            logger.info("📊 Calculating metrics...")
            
            # Use the actual function that exists and returns metrics dictionary
            metrics = calculate_two_stage_classification_metrics()
            
            # Extract key metrics from the function result
            if metrics and 'stage1_anomaly_detection' in metrics:
                stage1_recall = metrics.get('stage1_anomaly_detection', {}).get('recall', 0.0)
                stage2_oracle_acc = metrics.get('stage2_attack_category_classification', {}).get('oracle_true_anomalies', {}).get('accuracy', 0.0)
                true_end_to_end_f1 = metrics.get('end_to_end_detected_samples', {}).get('f1_macro', 0.0)
                logger.info("✅ Metrics extracted successfully")
                logger.info(f"   Extracted Stage-1 Recall: {stage1_recall:.4f}")
                logger.info(f"   Extracted Stage-2 Oracle Acc: {stage2_oracle_acc:.4f}")
                logger.info(f"   Extracted End-to-End F1: {true_end_to_end_f1:.4f}")
            else:
                # Fallback values if metrics structure is different or calculation failed
                logger.warning("⚠️ Unexpected metrics structure or calculation failed, using fallback values")
                stage1_recall = 0.0
                stage2_oracle_acc = 0.0
                true_end_to_end_f1 = 0.0
            
            experiment_result = {
                'seed': seed,
                'stage1_recall': stage1_recall,
                'stage2_oracle_accuracy': stage2_oracle_acc,
                'true_end_to_end_f1': true_end_to_end_f1,
                'timestamp': datetime.now().isoformat(),
                'full_metrics': metrics if metrics else None
            }
            
            logger.info(f"✅ Seed {seed} completed:")
            logger.info(f"   Stage-1 Recall: {stage1_recall:.4f}")
            logger.info(f"   Stage-2 Oracle Acc: {stage2_oracle_acc:.4f}")
            logger.info(f"   True End-to-End F1: {true_end_to_end_f1:.4f}")
            
            return experiment_result
            
        except Exception as e:
            logger.error(f"❌ Experiment with seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_experiments(self):
        """Run experiments with all seeds."""
        logger.info("🎯 Starting seeded experiments...")
        logger.info(f"📋 Seeds to test: {self.seeds}")
        
        for i, seed in enumerate(self.seeds, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"🧪 Experiment {i}/{len(self.seeds)}")
            logger.info(f"{'='*60}")
            
            result = self.run_single_experiment(seed)
            if result:
                self.results.append(result)
            else:
                logger.error(f"❌ Failed to get results for seed {seed}")
        
        # Calculate summary statistics
        self.calculate_summary_statistics()
        
        # Save results
        self.save_results()
        
        logger.info("🎉 All experiments completed!")
    
    def calculate_summary_statistics(self):
        """Calculate mean and std for key metrics in research paper format."""
        if len(self.results) < 2:
            logger.warning("⚠️ Need at least 2 successful runs for statistics")
            return
        
        logger.info("\n" + "="*80)
        logger.info("📊 RESEARCH PAPER STATISTICS (Mean ± Standard Deviation)")
        logger.info("="*80)
        
        # Define metrics with proper formatting
        metrics_config = {
            'stage1_recall': {
                'name': 'Stage-1 Recall',
                'is_percentage': False,
                'decimal_places': 3
            },
            'stage2_oracle_accuracy': {
                'name': 'Stage-2 Oracle Accuracy',
                'is_percentage': True,
                'decimal_places': 2
            },
            'true_end_to_end_f1': {
                'name': 'True End-to-End F1',
                'is_percentage': False,
                'decimal_places': 3
            }
        }
        
        # Store summary for final table
        summary_results = {}
        
        for metric_key, config in metrics_config.items():
            values = [r[metric_key] for r in self.results if r and metric_key in r]
            if len(values) >= 2:
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Format for research paper
                if config['is_percentage']:
                    mean_fmt = f"{mean_val*100:.{config['decimal_places']}f}%"
                    std_fmt = f"{std_val*100:.{config['decimal_places']}f}%"
                    combined_fmt = f"{mean_val*100:.{config['decimal_places']}f}% ± {std_val*100:.{config['decimal_places']}f}%"
                else:
                    mean_fmt = f"{mean_val:.{config['decimal_places']}f}"
                    std_fmt = f"{std_val:.{config['decimal_places']}f}"
                    combined_fmt = f"{mean_val:.{config['decimal_places']}f} ± {std_val:.{config['decimal_places']}f}"
                
                summary_results[metric_key] = {
                    'mean': mean_val,
                    'std': std_val,
                    'formatted': combined_fmt
                }
                
                logger.info(f"\n🎯 {config['name']}:")
                logger.info(f"   Mean: {mean_fmt}")
                logger.info(f"   Std:  {std_fmt}")
                logger.info(f"   📄 Paper Format: {combined_fmt}")
        
        # Create final research table
        logger.info(f"\n📋 FINAL RESEARCH TABLE:")
        logger.info(f"{'Metric':<25} {'Mean':<12} {'Std':<12} {'Format':<20}")
        logger.info("-" * 70)
        for metric_key, config in metrics_config.items():
            if metric_key in summary_results:
                res = summary_results[metric_key]
                if config['is_percentage']:
                    mean_display = f"{res['mean']*100:.{config['decimal_places']}f}%"
                    std_display = f"{res['std']*100:.{config['decimal_places']}f}%"
                else:
                    mean_display = f"{res['mean']:.{config['decimal_places']}f}"
                    std_display = f"{res['std']:.{config['decimal_places']}f}"
                
                logger.info(f"{config['name']:<25} {mean_display:<12} {std_display:<12} {res['formatted']:<20}")
        
        logger.info("="*80)
        
        # Save summary statistics
        self.save_summary_statistics(summary_results)
    
    def save_summary_statistics(self, summary_results):
        """Save summary statistics in research paper format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary as JSON
        summary_file = self.output_dir / f"seeded_experiments_summary_{timestamp}.json"
        
        summary_data = {
            'experiment_info': {
                'num_seeds': len(self.seeds),
                'seeds_used': self.seeds,
                'successful_runs': len(self.results),
                'timestamp': timestamp
            },
            'statistics': summary_results,
            'individual_results': self.results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        logger.info(f"📊 Summary statistics saved to: {summary_file}")
        
        # Also save a clean CSV for paper inclusion
        csv_file = self.output_dir / f"research_paper_table_{timestamp}.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Mean', 'Std Deviation', 'Paper Format'])
            
            metric_names = {
                'stage1_recall': 'Stage-1 Recall',
                'stage2_oracle_accuracy': 'Stage-2 Oracle Accuracy', 
                'true_end_to_end_f1': 'True End-to-End F1'
            }
            
            for metric_key, name in metric_names.items():
                if metric_key in summary_results:
                    res = summary_results[metric_key]
                    writer.writerow([name, res['mean'], res['std'], res['formatted']])
        
        logger.info(f"📋 Research paper table saved to: {csv_file}")
    
    def save_results(self):
        """Save results to CSV and JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        json_file = self.output_dir / f"seeded_experiments_detailed_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"💾 Detailed results saved to: {json_file}")
        
        # Save summary as CSV
        csv_file = self.output_dir / f"seeded_experiments_summary_{timestamp}.csv"
        
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
        logger.info(f"\n📋 Results Summary:")
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
    logger.info("🎲 SEEDED EXPERIMENTS FOR RESEARCH PAPER")
    logger.info("="*80)
    logger.info("📄 This script runs the complete two-stage pipeline with multiple seeds")
    logger.info("🎯 Purpose: Generate statistically robust results for Springer/Scopus publication")
    logger.info("🔬 Method: Train FULL pipeline for each seed (not just evaluation)")
    logger.info("📊 Output: Mean ± Standard Deviation for all key metrics")
    logger.info("="*80)
    
    runner = SeededExperimentRunner()
    runner.run_all_experiments()
    
    logger.info("\n🏁 EXPERIMENTS COMPLETED!")
    logger.info("📁 Check model_artifacts directory for:")
    logger.info("   • seeded_experiments_detailed_*.json - Full results per seed")
    logger.info("   • seeded_experiments_summary_*.json - Statistical summary") 
    logger.info("   • research_paper_table_*.csv - Ready for paper inclusion")
    logger.info("\n📄 Use the 'Paper Format' column directly in your research paper!")

if __name__ == "__main__":
    main()
