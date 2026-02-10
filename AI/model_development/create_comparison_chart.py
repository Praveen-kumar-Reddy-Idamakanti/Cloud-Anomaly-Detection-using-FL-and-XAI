import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from pathlib import Path

# Create comparison chart
def create_comparison_chart():
    # Load the comparison data
    artifacts_dir = Path(__file__).resolve().parents[1] / "model_artifacts"
    comparison_file = artifacts_dir / "centralized_vs_federated_comparison.json"
    
    with open(comparison_file, 'r') as f:
        data = json.load(f)

    # Extract metrics
    centralized = data['centralized']['best_metrics']
    federated = data['federated']['best_metrics']

    # Create comparison visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Centralized vs Federated Baseline Comparison', fontsize=16, fontweight='bold')

    # Metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    centralized_values = [centralized[m] for m in metrics]
    federated_values = [federated[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    # Bar chart comparison
    bars1 = ax1.bar(x - width/2, centralized_values, width, label='Centralized', alpha=0.8, color='#2E86AB')
    bars2 = ax1.bar(x + width/2, federated_values, width, label='Federated', alpha=0.8, color='#A23B72')

    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (c, f) in enumerate(zip(centralized_values, federated_values)):
        ax1.text(i - width/2, c + 0.01, f'{c:.3f}', ha='center', va='bottom', fontsize=10)
        ax1.text(i + width/2, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=10)

    # ROC-AUC comparison
    roc_data = [data['centralized']['best_metrics']['roc_auc'], data['federated']['roc_auc']]
    bars = ax2.bar(['Centralized', 'Federated'], roc_data, color=['#2E86AB', '#A23B72'])
    ax2.set_ylabel('ROC-AUC Score')
    ax2.set_title('ROC-AUC Comparison')
    ax2.grid(True, alpha=0.3)

    # Add value labels
    for i, v in enumerate(roc_data):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

    # Privacy comparison
    privacy_data = [506, 0.4]  # MB data exposure
    bars = ax3.bar(['Centralized', 'Federated'], privacy_data, color=['#FF6B6B', '#4ECDC4'])
    ax3.set_ylabel('Data Exposure (MB)')
    ax3.set_title('Privacy Preservation')
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for i, v in enumerate(privacy_data):
        ax3.text(i, v + 10, f'{v:.1f} MB', ha='center', va='bottom', fontsize=10)

    # Communication efficiency
    comm_data = [506, 2]  # Total communication
    bars = ax4.bar(['Centralized', 'Federated'], comm_data, color=['#FF6B6B', '#4ECDC4'])
    ax4.set_ylabel('Total Communication (MB)')
    ax4.set_title('Communication Efficiency')
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for i, v in enumerate(comm_data):
        ax4.text(i, v + 20, f'{v:.0f} MB', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    
    # Save the chart
    output_file = artifacts_dir / "centralized_vs_federated_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison chart saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    create_comparison_chart()
