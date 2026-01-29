"""
Autoencoder Visualization Plots for XAI Phase 2

Provides specialized visualization capabilities for autoencoder explainability:
- Reconstruction error visualizations
- Per-feature error analysis plots
- Latent space visualizations
- Feature attribution plots
- Anomaly explanation visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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

class AutoencoderPlotter:
    """
    Specialized plotter for autoencoder explanations
    """
    
    def __init__(self):
        self.color_palette = {
            'normal': '#2E86AB',
            'anomaly': '#A23B72',
            'reconstruction': '#F18F01',
            'latent': '#6A4C93'
        }
        
    def plot_reconstruction_error_distribution(self, reconstruction_errors, save_path=None):
        """
        Plot distribution of reconstruction errors
        
        Args:
            reconstruction_errors: Dictionary from AutoencoderExplainer.compute_reconstruction_errors
            save_path: Path to save the plot
        """
        errors = reconstruction_errors['total_errors']
        predictions = reconstruction_errors['predictions']
        threshold = reconstruction_errors['threshold']
        
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of errors
        normal_errors = errors[predictions == 0]
        anomaly_errors = errors[predictions == 1]
        
        ax1.hist(normal_errors, bins=50, alpha=0.7, label='Normal', color=self.color_palette['normal'])
        ax1.hist(anomaly_errors, bins=50, alpha=0.7, label='Anomaly', color=self.color_palette['anomaly'])
        ax1.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Reconstruction Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        data_to_plot = [normal_errors, anomaly_errors]
        bp = ax2.boxplot(data_to_plot, labels=['Normal', 'Anomaly'], patch_artist=True)
        bp['boxes'][0].set_facecolor(self.color_palette['normal'])
        bp['boxes'][1].set_facecolor(self.color_palette['anomaly'])
        ax2.axhline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
        ax2.set_ylabel('Reconstruction Error')
        ax2.set_title('Reconstruction Error Box Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_per_feature_reconstruction_errors(self, feature_analysis, top_k=20, save_path=None):
        """
        Plot per-feature reconstruction error analysis
        
        Args:
            feature_analysis: Dictionary from AutoencoderExplainer.analyze_per_feature_reconstruction
            top_k: Number of top features to display
            save_path: Path to save the plot
        """
        ranked_features = feature_analysis['ranked_features'][:top_k]
        features = [f[0] for f in ranked_features]
        error_diffs = [f[1]['error_difference'] for f in ranked_features]
        
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        bars = plt.barh(range(len(features)), error_diffs)
        
        # Color bars based on error difference magnitude
        for i, bar in enumerate(bars):
            if error_diffs[i] >= np.percentile(error_diffs, 80):
                bar.set_color(self.color_palette['anomaly'])
            elif error_diffs[i] >= np.percentile(error_diffs, 60):
                bar.set_color(self.color_palette['reconstruction'])
            else:
                bar.set_color(self.color_palette['normal'])
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Error Difference (Anomaly - Normal)')
        plt.title(f'Top {top_k} Features Contributing to Anomaly Detection')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(error_diffs):
            plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_comparison(self, original, reconstructed, feature_names=None, top_k=10, save_path=None):
        """
        Plot original vs reconstructed values for features
        
        Args:
            original: Original sample values
            reconstructed: Reconstructed sample values
            feature_names: List of feature names
            top_k: Number of top features to show
            save_path: Path to save the plot
        """
        if feature_names is None:
            feature_names = [f'feature_{i:02d}' for i in range(len(original))]
        
        # Calculate reconstruction errors per feature
        errors = np.abs(original - reconstructed)
        
        # Get top k features with highest errors
        top_indices = np.argsort(errors)[-top_k:]
        top_features = [feature_names[i] for i in top_indices]
        top_original = original[top_indices]
        top_reconstructed = reconstructed[top_indices]
        top_errors = errors[top_indices]
        
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(top_features))
        width = 0.35
        
        plt.bar(x - width/2, top_original, width, label='Original', color=self.color_palette['normal'])
        plt.bar(x + width/2, top_reconstructed, width, label='Reconstructed', color=self.color_palette['reconstruction'])
        
        plt.xlabel('Features')
        plt.ylabel('Feature Values')
        plt.title(f'Original vs Reconstructed Values (Top {top_k} Error Features)')
        plt.xticks(x, top_features, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add error annotations
        for i, error in enumerate(top_errors):
            plt.text(i, max(top_original[i], top_reconstructed[i]) + 0.01, 
                    f'Error: {error:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_latent_space_clusters(self, latent_representations, method='tsne', save_path=None):
        """
        Plot latent space with cluster visualization
        
        Args:
            latent_representations: Dictionary from AutoencoderExplainer.extract_latent_representations
            method: Dimensionality reduction method ('tsne', 'pca')
            save_path: Path to save the plot
        """
        latent_vectors = latent_representations['latent_vectors']
        labels = latent_representations['labels']
        errors = latent_representations['reconstruction_errors']
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
        elif method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'tsne' or 'pca'")
        
        reduced_vectors = reducer.fit_transform(latent_vectors)
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot colored by reconstruction error
        scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], 
                            c=errors, cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Reconstruction Error')
        
        # Overlay normal vs anomaly markers
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        
        if np.any(normal_mask):
            plt.scatter(reduced_vectors[normal_mask, 0], reduced_vectors[normal_mask, 1], 
                       c='blue', alpha=0.3, s=20, marker='o', label='Normal')
        if np.any(anomaly_mask):
            plt.scatter(reduced_vectors[anomaly_mask, 0], reduced_vectors[anomaly_mask, 1], 
                       c='red', alpha=0.3, s=20, marker='x', label='Anomaly')
        
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Latent Space Clustering ({method.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_attributions(self, attribution_data, feature_names=None, top_k=15, save_path=None):
        """
        Plot feature attributions from SHAP or Integrated Gradients
        
        Args:
            attribution_data: Dictionary containing attribution results
            feature_names: List of feature names
            top_k: Number of top features to show
            save_path: Path to save the plot
        """
        if 'shap_values' in attribution_data:
            values = attribution_data['shap_values'].flatten()
            method = 'SHAP'
        elif 'attributions' in attribution_data:
            values = attribution_data['attributions'].flatten()
            method = 'Integrated Gradients'
        else:
            raise ValueError("Attribution data must contain 'shap_values' or 'attributions'")
        
        if feature_names is None:
            feature_names = [f'feature_{i:02d}' for i in range(len(values))]
        
        # Get top k features by absolute attribution
        top_indices = np.argsort(np.abs(values))[-top_k:]
        top_features = [feature_names[i] for i in top_indices]
        top_values = values[top_indices]
        
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        colors = ['red' if v > 0 else 'blue' for v in top_values]
        bars = plt.barh(range(len(top_features)), top_values, color=colors, alpha=0.7)
        
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel(f'Feature Attribution ({method})')
        plt.title(f'Top {top_k} Feature Attributions ({method})')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(top_values):
            plt.text(v + 0.001 * np.sign(v), i, f'{v:.4f}', va='center', ha='left' if v > 0 else 'right', fontsize=10)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Positive Attribution'),
                          Patch(facecolor='blue', alpha=0.7, label='Negative Attribution')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_anomaly_explanation_summary(self, explanation_data, save_path=None):
        """
        Plot comprehensive anomaly explanation summary
        
        Args:
            explanation_data: Dictionary from AutoencoderExplainer.explain_anomaly_sample
            save_path: Path to save the plot
        """
        top_features = explanation_data['top_contributing_features'][:10]
        features = [f[0] for f in top_features]
        errors = [f[1] for f in top_features]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Top contributing features
        bars = ax1.barh(range(len(features)), errors)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features)
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_title('Top 10 Contributing Features')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # 2. Original vs Reconstructed comparison
        original = explanation_data['original_sample'][:10]
        reconstructed = explanation_data['reconstructed_sample'][:10]
        
        x = np.arange(len(original))
        ax2.plot(x, original, 'o-', label='Original', color=self.color_palette['normal'])
        ax2.plot(x, reconstructed, 's-', label='Reconstructed', color=self.color_palette['reconstruction'])
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Feature Value')
        ax2.set_title('Original vs Reconstructed (First 10 Features)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Error distribution across features
        per_feature_errors = explanation_data['per_feature_errors'][:20]
        ax3.hist(per_feature_errors, bins=20, alpha=0.7, color=self.color_palette['anomaly'])
        ax3.set_xlabel('Reconstruction Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Per-Feature Error Distribution (First 20 Features)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics
        threshold_str = f"{explanation_data['threshold']:.6f}" if explanation_data['threshold'] else 'N/A'
        stats_text = f"""
Anomaly Analysis Summary:
• Reconstruction Error: {explanation_data['reconstruction_error']:.6f}
• Is Anomaly: {explanation_data['is_anomaly']}
• Threshold: {threshold_str}

Error Statistics:
• Mean Feature Error: {np.mean(explanation_data['per_feature_errors']):.6f}
• Max Feature Error: {np.max(explanation_data['per_feature_errors']):.6f}
• Min Feature Error: {np.min(explanation_data['per_feature_errors']):.6f}
• Std Feature Error: {np.std(explanation_data['per_feature_errors']):.6f}
"""
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Explanation Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_interactive_reconstruction_analysis(self, reconstruction_errors, save_path=None):
        """
        Create interactive reconstruction analysis plot (if Plotly available)
        
        Args:
            reconstruction_errors: Dictionary from AutoencoderExplainer.compute_reconstruction_errors
            save_path: Path to save the HTML plot
            
        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Use static plots instead.")
            return None
        
        errors = reconstruction_errors['total_errors']
        predictions = reconstruction_errors['predictions']
        threshold = reconstruction_errors['threshold']
        
        # Create interactive histogram
        fig = go.Figure()
        
        # Add normal samples
        normal_errors = errors[predictions == 0]
        fig.add_trace(go.Histogram(
            x=normal_errors,
            name='Normal',
            opacity=0.7,
            marker_color=self.color_palette['normal']
        ))
        
        # Add anomaly samples
        anomaly_errors = errors[predictions == 1]
        fig.add_trace(go.Histogram(
            x=anomaly_errors,
            name='Anomaly',
            opacity=0.7,
            marker_color=self.color_palette['anomaly']
        ))
        
        # Add threshold line
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Threshold: {threshold:.4f}")
        
        fig.update_layout(
            title='Interactive Reconstruction Error Distribution',
            xaxis_title='Reconstruction Error',
            yaxis_title='Frequency',
            barmode='overlay',
            hovermode='x unified'
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
