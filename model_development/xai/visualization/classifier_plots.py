"""
Classifier Visualization Plots for XAI Phase 3

Provides specialized visualization capabilities for attack type classification explainability:
- Confusion matrix visualizations
- Attack type feature importance plots
- Decision boundary visualizations
- Confidence and uncertainty plots
- Misclassification analysis visualizations
- LIME explanation plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Optional Plotly imports for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

class ClassifierPlotter:
    """
    Specialized plotter for classifier explanations
    """
    
    def __init__(self):
        self.color_palette = {
            'Normal': '#2E86AB',
            'DoS': '#A23B72',
            'PortScan': '#F18F01',
            'BruteForce': '#C73E1D',
            'WebAttack': '#6A4C93',
            'Infiltration': '#FF6B6B'
        }
        
    def plot_confusion_matrix(self, confusion_matrix_data, class_names=None, save_path=None):
        """
        Plot confusion matrix with annotations
        
        Args:
            confusion_matrix_data: Confusion matrix array
            class_names: List of class names
            save_path: Path to save the plot
        """
        if class_names is None:
            class_names = ['Normal', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(confusion_matrix_data, 
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Number of Samples'})
        
        plt.title('Confusion Matrix - Attack Type Classification')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Add accuracy text
        accuracy = np.trace(confusion_matrix_data) / np.sum(confusion_matrix_data)
        plt.text(0.5, 1.05, f'Overall Accuracy: {accuracy:.3f}', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attack_type_feature_importance(self, importance_data, top_k=15, save_path=None):
        """
        Plot feature importance for each attack type
        
        Args:
            importance_data: Dictionary of attack type importance scores
            top_k: Number of top features to display
            save_path: Path to save the plot
        """
        # Create subplots for each attack type
        n_attack_types = len(importance_data)
        cols = 2
        rows = (n_attack_types + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if n_attack_types == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (attack_type, importance_scores) in enumerate(importance_data.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Sort features by importance
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            features, scores = zip(*sorted_features)
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(features)), scores, color=self.color_palette.get(attack_type, '#888888'))
            
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels([f'F_{f.split("_")[1]}' for f in features])
            ax.set_xlabel('Importance Score')
            ax.set_title(f'{attack_type} - Top {top_k} Features')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            
            # Add value labels
            for j, v in enumerate(scores):
                ax.text(v + 0.001, j, f'{v:.3f}', va='center', fontsize=9)
        
        # Remove empty subplots
        for i in range(n_attack_types, rows * cols):
            row, col = i // cols, i % cols
            if rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confidence_distribution(self, confidence_data, save_path=None):
        """
        Plot confidence distribution analysis
        
        Args:
            confidence_data: Dictionary from confidence analysis
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall confidence histogram
        overall_stats = confidence_data['overall_stats']
        ax1.hist([overall_stats['mean_confidence']], bins=20, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Overall Confidence Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 2. Per-class confidence box plot
        per_class_stats = confidence_data['per_class_stats']
        class_names = list(per_class_stats.keys())
        confidences = [per_class_stats[cls]['mean_confidence'] for cls in class_names]
        colors = [self.color_palette.get(cls, '#888888') for cls in class_names]
        
        bp = ax2.boxplot(confidences, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_xticks(range(1, len(class_names) + 1))
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.set_ylabel('Mean Confidence')
        ax2.set_title('Confidence by Attack Type')
        ax2.grid(True, alpha=0.3)
        
        # 3. Low confidence samples
        low_conf = confidence_data['low_confidence_samples']
        ax3.bar(['Low Confidence'], [low_conf['count']], color='orange', alpha=0.7)
        ax3.set_ylabel('Number of Samples')
        ax3.set_title(f'Low Confidence Samples (< 0.5): {low_conf["count"]} ({low_conf["percentage"]:.1f}%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. High uncertainty samples
        high_unc = confidence_data['high_uncertainty_samples']
        ax4.bar(['High Uncertainty'], [high_unc['count']], color='red', alpha=0.7)
        ax4.set_ylabel('Number of Samples')
        ax4.set_title(f'High Uncertainty Samples: {high_unc["count"]} ({high_unc["percentage"]:.1f}%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_misclassification_analysis(self, misclassification_data, save_path=None):
        """
        Plot misclassification analysis
        
        Args:
            misclassification_data: Dictionary from misclassification analysis
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion matrix
        cm = misclassification_data['confusion_matrix']
        class_names = ['Normal', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax1, cbar=False)
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # 2. Most confused pairs
        confused_pairs = misclassification_data['most_confused_pairs'][:10]
        if confused_pairs:
            pairs = [f"{pair['true_class']}→{pair['predicted_class']}" for pair in confused_pairs]
            counts = [pair['count'] for pair in confused_pairs]
            
            ax2.barh(range(len(pairs)), counts, color='coral', alpha=0.7)
            ax2.set_yticks(range(len(pairs)))
            ax2.set_yticklabels(pairs)
            ax2.set_xlabel('Number of Misclassifications')
            ax2.set_title('Top 10 Confused Class Pairs')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3)
        
        # 3. Misclassification rate
        misclass_rate = misclassification_data['misclassification_rate']
        correct_rate = 1 - misclass_rate
        
        ax3.pie([correct_rate, misclass_rate], 
               labels=['Correct', 'Misclassified'], 
               colors=['lightgreen', 'lightcoral'],
               autopct='%1.1f%%')
        ax3.set_title(f'Classification Accuracy: {correct_rate:.1%}')
        
        # 4. Misclassification patterns summary
        patterns = misclassification_data['misclassification_patterns']
        pattern_types = list(patterns.keys())
        pattern_counts = [patterns[pattern]['count'] for pattern in pattern_types]
        
        if pattern_types:
            ax4.barh(range(len(pattern_types)), pattern_counts, color='orange', alpha=0.7)
            ax4.set_yticks(range(len(pattern_types)))
            ax4.set_yticklabels([p.replace('_', ' ') for p in pattern_types])
            ax4.set_xlabel('Number of Samples')
            ax4.set_title('Misclassification Patterns')
            ax4.invert_yaxis()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_decision_boundaries(self, data, labels, classifier_model, feature_indices=(0, 1), resolution=100, save_path=None):
        """
        Plot decision boundaries for 2D feature space
        
        Args:
            data: Input data (numpy array)
            labels: True labels
            classifier_model: Trained classifier model
            feature_indices: Which two features to plot
            resolution: Grid resolution for decision boundary
            save_path: Path to save the plot
        """
        # Select only two features
        X_2d = data[:, feature_indices]
        
        # Create mesh grid
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                           np.linspace(y_min, y_max, resolution))
        
        # Predict on mesh grid
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Pad mesh points to full feature dimension
        full_mesh = np.zeros((mesh_points.shape[0], data.shape[1]))
        full_mesh[:, feature_indices[0]] = mesh_points[:, 0]
        full_mesh[:, feature_indices[1]] = mesh_points[:, 1]
        
        # Use mean values for other features
        for i in range(data.shape[1]):
            if i not in feature_indices:
                full_mesh[:, i] = np.mean(data[:, i])
        
        # Get predictions
        import torch
        with torch.no_grad():
            mesh_tensor = torch.FloatTensor(full_mesh)
            predictions = classifier_model(mesh_tensor)
            pred_classes = torch.argmax(predictions, dim=1).numpy().reshape(xx.shape)
        
        # Plot decision boundary
        plt.figure(figsize=(12, 8))
        
        # Plot decision boundary
        plt.contourf(xx, yy, pred_classes, alpha=0.3, cmap='viridis')
        
        # Plot actual data points
        class_names = ['Normal', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A4C93', '#FF6B6B']
        
        for i, class_name in enumerate(class_names):
            mask = labels == i
            if np.any(mask):
                plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                          c=colors[i], label=class_name, alpha=0.7, edgecolors='black')
        
        plt.xlabel(f'Feature {feature_indices[0]}')
        plt.ylabel(f'Feature {feature_indices[1]}')
        plt.title(f'Decision Boundaries (Features {feature_indices[0]} & {feature_indices[1]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_lime_explanation(self, lime_explanation, save_path=None):
        """
        Plot LIME explanation for attack classification
        
        Args:
            lime_explanation: LIME explanation object
            save_path: Path to save the plot
        """
        if not lime_explanation:
            print("No LIME explanation provided")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Get explanation for the predicted class
        if hasattr(lime_explanation, 'as_list'):
            # Get explanation for the class with highest probability
            predicted_class = max(lime_explanation.class_probs, key=lime_explanation.class_probs.get)
            explanation_list = lime_explanation.as_list(label=predicted_class)
            
            features = [item[0] for item in explanation_list]
            weights = [item[1] for item in explanation_list]
            colors = ['red' if w > 0 else 'blue' for w in weights]
            
            # Create horizontal bar chart
            bars = plt.barh(range(len(features)), weights, color=colors, alpha=0.7)
            
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Weight')
            plt.title(f'LIME Explanation for {predicted_class} Classification')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(weights):
                plt.text(v + 0.001 * np.sign(v), i, f'{v:.3f}', va='center', 
                        ha='left' if v > 0 else 'right', fontsize=10)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.7, label='Supports'),
                              Patch(facecolor='blue', alpha=0.7, label='Opposes')]
            plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_confidence_analysis(self, confidence_data, save_path=None):
        """
        Create interactive confidence analysis plot (if Plotly available)
        
        Args:
            confidence_data: Dictionary from confidence analysis
            save_path: Path to save the HTML plot
            
        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Use static plots instead.")
            return None
        
        per_class_stats = confidence_data['per_class_stats']
        class_names = list(per_class_stats.keys())
        confidences = [per_class_stats[cls]['mean_confidence'] for cls in class_names]
        std_confs = [per_class_stats[cls]['std_confidence'] for cls in class_names]
        colors = [self.color_palette.get(cls, '#888888') for cls in class_names]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=class_names,
            y=confidences,
            error_y=dict(type='data', array=std_confs, visible=True),
            marker_color=colors,
            name='Mean Confidence'
        ))
        
        fig.update_layout(
            title='Interactive Confidence Analysis by Attack Type',
            xaxis_title='Attack Type',
            yaxis_title='Mean Confidence Score',
            yaxis=dict(range=[0, 1])
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_comprehensive_classifier_summary(self, confidence_data, misclassification_data, 
                                             feature_importance_data, save_path=None):
        """
        Create comprehensive summary plot for classifier analysis
        
        Args:
            confidence_data: Dictionary from confidence analysis
            misclassification_data: Dictionary from misclassification analysis
            feature_importance_data: Dictionary from feature importance analysis
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Overall accuracy
        cm = misclassification_data['confusion_matrix']
        accuracy = np.trace(cm) / np.sum(cm)
        axes[0, 0].pie([accuracy, 1-accuracy], labels=['Correct', 'Incorrect'], 
                      colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        axes[0, 0].set_title(f'Overall Accuracy: {accuracy:.1%}')
        
        # 2. Confidence distribution
        overall_stats = confidence_data['overall_stats']
        axes[0, 1].hist([overall_stats['mean_confidence']], bins=20, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Top misclassifications
        confused_pairs = misclassification_data['most_confused_pairs'][:5]
        if confused_pairs:
            pairs = [f"{pair['true_class']}→{pair['predicted_class']}" for pair in confused_pairs]
            counts = [pair['count'] for pair in confused_pairs]
            
            axes[0, 2].barh(range(len(pairs)), counts, color='coral', alpha=0.7)
            axes[0, 2].set_yticks(range(len(pairs)))
            axes[0, 2].set_yticklabels(pairs)
            axes[0, 2].set_title('Top 5 Misclassifications')
            axes[0, 2].invert_yaxis()
        
        # 4. Feature importance heatmap
        if feature_importance_data:
            # Create importance matrix
            all_features = set()
            for attack_type, importance in feature_importance_data.items():
                all_features.update(importance.keys())
            
            all_features = sorted(list(all_features))[:10]  # Top 10 features
            importance_matrix = []
            
            for attack_type in feature_importance_data.keys():
                row = []
                for feature in all_features:
                    row.append(feature_importance_data[attack_type].get(feature, 0))
                importance_matrix.append(row)
            
            importance_df = pd.DataFrame(importance_matrix, 
                                        index=list(feature_importance_data.keys()),
                                        columns=[f'F_{f.split("_")[1]}' for f in all_features])
            
            sns.heatmap(importance_df, annot=True, fmt='.3f', cmap='viridis', ax=axes[1, 0])
            axes[1, 0].set_title('Feature Importance Heatmap')
        
        # 5. Low confidence analysis
        low_conf = confidence_data['low_confidence_samples']
        axes[1, 1].bar(['Low Confidence'], [low_conf['count']], color='orange', alpha=0.7)
        axes[1, 1].set_title(f'Low Confidence: {low_conf["percentage"]:.1f}%')
        axes[1, 1].set_ylabel('Number of Samples')
        
        # 6. Summary statistics
        summary_text = f"""
Classifier Performance Summary:

• Overall Accuracy: {accuracy:.1%}
• Misclassification Rate: {misclassification_data["misclassification_rate"]:.1%}
• Mean Confidence: {overall_stats["mean_confidence"]:.3f}
• Low Confidence Samples: {low_conf["count"]} ({low_conf["percentage"]:.1f}%)

Most Confused Pairs:
"""
        
        for i, pair in enumerate(confused_pairs[:3], 1):
            summary_text += f"{i}. {pair['true_class']} → {pair['predicted_class']}: {pair['count']} samples\n"
        
        axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
