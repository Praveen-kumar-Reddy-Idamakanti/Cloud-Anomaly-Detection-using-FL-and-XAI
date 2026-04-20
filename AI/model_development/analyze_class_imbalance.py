#!/usr/bin/env python3
"""
Class Imbalance Analysis and Visualization
Creates comprehensive visualizations of the class imbalance problem
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_metrics_data():
    """Load the two-stage classification metrics"""
    metrics_path = Path(__file__).parent.parent / "model_artifacts" / "two_stage_classification_metrics.json"
    
    if not metrics_path.exists():
        print(f"❌ Metrics file not found: {metrics_path}")
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)

def create_class_distribution_plot(data):
    """Create comprehensive class imbalance visualizations"""
    
    # Extract classification report data
    oracle_report = data['stage2_attack_category_classification']['oracle_true_anomalies']['classification_report']
    
    # Prepare data for visualization
    classes = ['Botnet', 'DoS', 'Infiltration', 'Other', 'PortScan']
    support = [oracle_report[cls]['support'] for cls in classes]
    precision = [oracle_report[cls]['precision'] for cls in classes]
    recall = [oracle_report[cls]['recall'] for cls in classes]
    f1_scores = [oracle_report[cls]['f1-score'] for cls in classes]
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Class Imbalance Analysis - Two-Stage Anomaly Detection', fontsize=16, fontweight='bold')
    
    # 1. Class Distribution (Log Scale)
    ax1 = axes[0, 0]
    bars = ax1.bar(classes, support, color=['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6'])
    ax1.set_yscale('log')
    ax1.set_title('Class Distribution (Log Scale)', fontweight='bold')
    ax1.set_ylabel('Number of Samples (log scale)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, support):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Class Distribution (Percentage)
    ax2 = axes[0, 1]
    total_samples = sum(support)
    percentages = [s/total_samples * 100 for s in support]
    colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
    
    wedges, texts, autotexts = ax2.pie(percentages, labels=classes, autopct='%1.2f%%', 
                                      colors=colors, startangle=90)
    ax2.set_title('Class Distribution (%)', fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 3. Performance Metrics Comparison
    ax3 = axes[0, 2]
    x = np.arange(len(classes))
    width = 0.25
    
    ax3.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#3498db')
    ax3.bar(x, recall, width, label='Recall', alpha=0.8, color='#e74c3c')
    ax3.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='#2ecc71')
    
    ax3.set_xlabel('Attack Classes')
    ax3.set_ylabel('Score')
    ax3.set_title('Performance Metrics by Class', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1_scores)):
        ax3.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i + width, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Support vs Performance Scatter
    ax4 = axes[1, 0]
    scatter = ax4.scatter(support, f1_scores, s=200, alpha=0.7, c=range(len(classes)), 
                          cmap='viridis', edgecolors='black')
    
    # Add class labels to points
    for i, cls in enumerate(classes):
        ax4.annotate(cls, (support[i], f1_scores[i]), xytext=(5, 5), 
                    textcoords='offset points', fontweight='bold')
    
    ax4.set_xlabel('Class Support (Number of Samples)')
    ax4.set_ylabel('F1-Score')
    ax4.set_title('Support vs F1-Score Relationship', fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    # 5. Confusion Matrix Heatmap
    ax5 = axes[1, 1]
    cm = np.array(data['stage2_attack_category_classification']['oracle_true_anomalies']['confusion_matrix'])
    
    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Reds', 
                xticklabels=classes, yticklabels=classes, ax=ax5)
    ax5.set_title('Normalized Confusion Matrix', fontweight='bold')
    ax5.set_xlabel('Predicted Class')
    ax5.set_ylabel('True Class')
    
    # 6. Class Imbalance Severity
    ax6 = axes[1, 2]
    
    # Calculate imbalance ratios
    max_support = max(support)
    imbalance_ratios = [max_support / s for s in support]
    
    bars = ax6.bar(classes, imbalance_ratios, color=['#e74c3c', '#f39c12', '#e74c3c', '#2ecc71', '#f39c12'])
    ax6.set_ylabel('Imbalance Ratio (max/class)')
    ax6.set_title('Class Imbalance Severity', fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.set_yscale('log')
    
    # Add value labels
    for bar, ratio in zip(bars, imbalance_ratios):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.1f}x', ha='center', va='bottom', fontweight='bold')
    
    # Color code bars by severity
    for bar, ratio in zip(bars, imbalance_ratios):
        if ratio > 1000:
            bar.set_color('#e74c3c')  # Red for severe
        elif ratio > 100:
            bar.set_color('#f39c12')  # Orange for moderate
        else:
            bar.set_color('#2ecc71')  # Green for mild
    
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(__file__).parent.parent / "model_artifacts" / "class_imbalance_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Class imbalance visualization saved to: {output_path}")
    
    plt.show()
    
    return {
        'classes': classes,
        'support': support,
        'precision': precision,
        'recall': recall,
        'f1_scores': f1_scores,
        'imbalance_ratios': imbalance_ratios
    }

def create_imbalance_summary_table(data):
    """Create a summary table of class imbalance metrics"""
    
    oracle_report = data['stage2_attack_category_classification']['oracle_true_anomalies']['classification_report']
    
    classes = ['Botnet', 'DoS', 'Infiltration', 'Other', 'PortScan']
    
    summary_data = []
    total_support = sum(oracle_report[cls]['support'] for cls in classes)
    
    for cls in classes:
        support = oracle_report[cls]['support']
        percentage = (support / total_support) * 100
        
        summary_data.append({
            'Class': cls,
            'Support': support,
            'Percentage': percentage,
            'Precision': oracle_report[cls]['precision'],
            'Recall': oracle_report[cls]['recall'],
            'F1-Score': oracle_report[cls]['f1-score'],
            'Imbalance_Ratio': max(oracle_report[c]['support'] for c in classes) / support
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    csv_path = Path(__file__).parent.parent / "model_artifacts" / "class_imbalance_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Class imbalance summary saved to: {csv_path}")
    
    return df

def generate_recommendations(data):
    """Generate specific recommendations based on the analysis"""
    
    oracle_report = data['stage2_attack_category_classification']['oracle_true_anomalies']['classification_report']
    
    classes = ['Botnet', 'DoS', 'Infiltration', 'Other', 'PortScan']
    support = [oracle_report[cls]['support'] for cls in classes]
    f1_scores = [oracle_report[cls]['f1-score'] for cls in classes]
    
    recommendations = []
    
    # Identify critical issues
    for i, cls in enumerate(classes):
        if support[i] < 1000:  # Rare classes
            recommendations.append({
                'priority': 'CRITICAL',
                'class': cls,
                'issue': f'Extremely rare class ({support[i]} samples)',
                'impact': f'F1-Score: {f1_scores[i]:.3f}',
                'solution': 'Implement aggressive oversampling and class-weighted loss'
            })
        elif f1_scores[i] < 0.3:  # Poor performance
            recommendations.append({
                'priority': 'HIGH',
                'class': cls,
                'issue': f'Poor classification performance',
                'impact': f'F1-Score: {f1_scores[i]:.3f}',
                'solution': 'Use focal loss and data augmentation'
            })
        elif support[i] > 100000:  # Dominant classes
            recommendations.append({
                'priority': 'MEDIUM',
                'class': cls,
                'issue': f'Dominant class causing bias',
                'impact': f'May overshadow rare classes',
                'solution': 'Consider undersampling or hierarchical classification'
            })
    
    # Save recommendations
    rec_df = pd.DataFrame(recommendations)
    rec_path = Path(__file__).parent.parent / "model_artifacts" / "class_imbalance_recommendations.csv"
    rec_df.to_csv(rec_path, index=False)
    print(f"✅ Recommendations saved to: {rec_path}")
    
    return rec_df

def main():
    """Main analysis function"""
    print("🔍 Starting Class Imbalance Analysis...")
    
    # Load data
    data = load_metrics_data()
    if data is None:
        return
    
    print("✅ Data loaded successfully")
    
    # Create visualizations
    viz_data = create_class_distribution_plot(data)
    
    # Create summary table
    summary_df = create_imbalance_summary_table(data)
    print("\n📊 Class Imbalance Summary:")
    print(summary_df.to_string(index=False))
    
    # Generate recommendations
    rec_df = generate_recommendations(data)
    print("\n💡 Top Recommendations:")
    print(rec_df.head().to_string(index=False))
    
    # Key insights
    print("\n🎯 Key Insights:")
    print(f"   - Most imbalanced class: {viz_data['classes'][np.argmax(viz_data['imbalance_ratios'])]}")
    print(f"   - Worst performing class: {viz_data['classes'][np.argmin(viz_data['f1_scores'])]}")
    print(f"   - Imbalance severity: {max(viz_data['imbalance_ratios']):.1f}x")
    print(f"   - Macro F1-Score: {np.mean(viz_data['f1_scores']):.3f}")
    
    print("\n✅ Class imbalance analysis completed!")

if __name__ == "__main__":
    main()
