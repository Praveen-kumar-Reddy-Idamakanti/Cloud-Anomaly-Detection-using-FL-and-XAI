import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json

def generate_stage2_confusion_matrix():
    """Generate confusion matrix for Stage-2 attack classification"""
    
    # Load actual confusion matrix data from your metrics
    confusion_matrix_data = [
        [0, 37, 0, 218, 20],      # Botnet row
        [0, 113149, 0, 11479, 587],  # DoS row  
        [0, 2, 0, 12, 11],          # Infiltration row
        [0, 2774, 0, 253062, 667], # Other row
        [0, 1842, 0, 15715, 48422]  # PortScan row
    ]
    
    class_names = ['Botnet', 'DoS', 'Infiltration', 'Other', 'PortScan']
    
    # Create the figure with more space
    plt.figure(figsize=(14, 10))
    
    # Create heatmap
    sns.heatmap(confusion_matrix_data, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Stage-2 Attack Classification Confusion Matrix\n(Oracle Mode - True Anomalies)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    # Add grid for better visual separation
    plt.grid(False)
    
    # Calculate and add performance metrics
    total_samples = np.sum(confusion_matrix_data)
    correct_predictions = np.trace(confusion_matrix_data)
    overall_accuracy = correct_predictions / total_samples
    
    # Add text box with key metrics - positioned to avoid overlap
    metrics_text = f'Overall Accuracy: {overall_accuracy:.3f}\n'
    metrics_text += f'Total Samples: {total_samples:,}\n'
    metrics_text += f'Correct Predictions: {correct_predictions:,}'
    
    plt.text(1.15, 0.95, metrics_text, 
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='black'))
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(__file__).resolve().parents[1] / "model_artifacts" / "stage2_confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Stage-2 confusion matrix saved to: {output_path}")
    
    # Print analysis for paper reference
    print("\n📊 Confusion Matrix Analysis:")
    print("=" * 50)
    
    # Per-class analysis
    for i, class_name in enumerate(class_names):
        true_positives = confusion_matrix_data[i][i]
        row_total = sum(confusion_matrix_data[i])
        col_total = sum(confusion_matrix_data[j][i] for j in range(len(class_names)))
        
        precision = true_positives / col_total if col_total > 0 else 0
        recall = true_positives / row_total if row_total > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{class_name}:")
        print(f"  True Positives: {true_positives:,}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        # Show main confusions
        for j, other_class in enumerate(class_names):
            if i != j and confusion_matrix_data[i][j] > 0:
                print(f"  → Misclassified as {other_class}: {confusion_matrix_data[i][j]:,}")
    
    print(f"\n🎯 Key Insights:")
    print(f"  • DoS: Strong detection (90.4% recall) but confused with Other")
    print(f"  • PortScan: High precision (97.4%) but moderate recall (73.4%)")
    print(f"  • Other: Excellent separation (98.7% recall)")
    print(f"  • Rare classes (Botnet, Infiltration): Need specialized handling")
    
    return output_path

if __name__ == "__main__":
    generate_stage2_confusion_matrix()
