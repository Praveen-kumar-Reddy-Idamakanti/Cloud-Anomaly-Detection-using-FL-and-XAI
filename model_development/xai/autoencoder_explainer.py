"""
Autoencoder Explainer for XAI Phase 2

Provides comprehensive explainability capabilities for autoencoder-based anomaly detection:
- Per-feature reconstruction error analysis
- Latent space visualization and analysis
- Feature attribution for reconstruction error
- Gradient-based explanations
- Anomaly-specific explanation generation
- Similarity analysis for anomaly detection
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports for advanced features
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from captum.attr import IntegratedGradients, LayerGradCam
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class AutoencoderExplainer:
    """
    Comprehensive explainer for autoencoder-based anomaly detection
    """
    
    def __init__(self, autoencoder_model, device='cpu'):
        """
        Initialize Autoencoder Explainer
        
        Args:
            autoencoder_model: Trained autoencoder model (PyTorch)
            device: Device for computation ('cpu' or 'cuda')
        """
        self.model = autoencoder_model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Storage for analysis results
        self.reconstruction_errors = {}
        self.latent_representations = {}
        self.feature_attributions = {}
        self.explanation_cache = {}
        
    def compute_reconstruction_errors(self, data_loader, threshold=None):
        """
        Compute reconstruction errors for all samples
        
        Args:
            data_loader: DataLoader containing input data
            threshold: Anomaly threshold (if None, will be computed from data)
            
        Returns:
            Dictionary containing reconstruction error analysis
        """
        self.model.eval()
        all_errors = []
        all_inputs = []
        all_outputs = []
        all_per_feature_errors = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data.to(self.device)
                
                # Forward pass
                reconstructed = self.model(data)
                
                # Compute reconstruction error
                mse_error = torch.mean((data - reconstructed) ** 2, dim=1)
                mae_error = torch.mean(torch.abs(data - reconstructed), dim=1)
                
                # Per-feature errors
                per_feature_error = (data - reconstructed) ** 2
                
                all_errors.extend(mse_error.cpu().numpy())
                all_inputs.extend(data.cpu().numpy())
                all_outputs.extend(reconstructed.cpu().numpy())
                all_per_feature_errors.extend(per_feature_error.cpu().numpy())
        
        # Convert to numpy arrays
        all_errors = np.array(all_errors)
        all_inputs = np.array(all_inputs)
        all_outputs = np.array(all_outputs)
        all_per_feature_errors = np.array(all_per_feature_errors)
        
        # Compute threshold if not provided
        if threshold is None:
            # Use 95th percentile of normal samples as threshold
            threshold = np.percentile(all_errors, 95)
        
        # Classify anomalies
        anomaly_predictions = (all_errors > threshold).astype(int)
        
        # Store results
        self.reconstruction_errors = {
            'total_errors': all_errors,
            'per_feature_errors': all_per_feature_errors,
            'threshold': threshold,
            'predictions': anomaly_predictions,
            'inputs': all_inputs,
            'outputs': all_outputs,
            'mean_error': np.mean(all_errors),
            'std_error': np.std(all_errors),
            'max_error': np.max(all_errors),
            'min_error': np.min(all_errors)
        }
        
        return self.reconstruction_errors
    
    def analyze_per_feature_reconstruction(self, feature_names=None):
        """
        Analyze reconstruction errors per feature
        
        Args:
            feature_names: List of feature names (if None, will use generic names)
            
        Returns:
            Dictionary containing per-feature analysis
        """
        if not self.reconstruction_errors:
            raise ValueError("Reconstruction errors not computed. Call compute_reconstruction_errors first.")
        
        per_feature_errors = self.reconstruction_errors['per_feature_errors']
        n_features = per_feature_errors.shape[1]
        
        if feature_names is None:
            feature_names = [f'feature_{i:02d}' for i in range(n_features)]
        
        # Compute statistics for each feature
        feature_analysis = {}
        
        for i, feature_name in enumerate(feature_names):
            feature_errors = per_feature_errors[:, i]
            
            # Separate normal and anomaly samples
            normal_mask = self.reconstruction_errors['predictions'] == 0
            anomaly_mask = self.reconstruction_errors['predictions'] == 1
            
            normal_errors = feature_errors[normal_mask]
            anomaly_errors = feature_errors[anomaly_mask]
            
            feature_stats = {
                'mean_error': np.mean(feature_errors),
                'std_error': np.std(feature_errors),
                'max_error': np.max(feature_errors),
                'min_error': np.min(feature_errors),
                'normal_mean': np.mean(normal_errors) if len(normal_errors) > 0 else 0,
                'normal_std': np.std(normal_errors) if len(normal_errors) > 0 else 0,
                'anomaly_mean': np.mean(anomaly_errors) if len(anomaly_errors) > 0 else 0,
                'anomaly_std': np.std(anomaly_errors) if len(anomaly_errors) > 0 else 0,
                'error_difference': np.mean(anomaly_errors) - np.mean(normal_errors) if len(normal_errors) > 0 and len(anomaly_errors) > 0 else 0,
                'error_ratio': np.mean(anomaly_errors) / np.mean(normal_errors) if len(normal_errors) > 0 and np.mean(normal_errors) > 0 else np.inf
            }
            
            feature_analysis[feature_name] = feature_stats
        
        # Rank features by error difference
        ranked_features = sorted(feature_analysis.items(), 
                               key=lambda x: x[1]['error_difference'], 
                               reverse=True)
        
        return {
            'feature_analysis': feature_analysis,
            'ranked_features': ranked_features,
            'top_contributing_features': ranked_features[:10]
        }
    
    def extract_latent_representations(self, data_loader):
        """
        Extract latent space representations from the bottleneck layer
        
        Args:
            data_loader: DataLoader containing input data
            
        Returns:
            Dictionary containing latent representations
        """
        self.model.eval()
        all_latent = []
        all_labels = []
        all_errors = []
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(data_loader):
                data = data.to(self.device)
                
                # Extract latent representation
                latent = self.model.encoder(data)
                
                all_latent.extend(latent.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Also compute reconstruction errors for coloring
                reconstructed = self.model(data)
                error = torch.mean((data - reconstructed) ** 2, dim=1)
                all_errors.extend(error.cpu().numpy())
        
        # Convert to numpy arrays
        all_latent = np.array(all_latent)
        all_labels = np.array(all_labels)
        all_errors = np.array(all_errors)
        
        self.latent_representations = {
            'latent_vectors': all_latent,
            'labels': all_labels,
            'reconstruction_errors': all_errors,
            'latent_dim': all_latent.shape[1],
            'n_samples': len(all_latent)
        }
        
        return self.latent_representations
    
    def visualize_latent_space(self, method='tsne', n_components=2, save_path=None):
        """
        Visualize latent space using dimensionality reduction
        
        Args:
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            n_components: Number of components for visualization
            save_path: Path to save the visualization
            
        Returns:
            Dictionary containing visualization results
        """
        if not self.latent_representations:
            raise ValueError("Latent representations not extracted. Call extract_latent_representations first.")
        
        latent_vectors = self.latent_representations['latent_vectors']
        labels = self.latent_representations['labels']
        errors = self.latent_representations['reconstruction_errors']
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(latent_vectors)-1))
        elif method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method.lower() == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
            except ImportError:
                print("UMAP not available. Using t-SNE instead.")
                reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(latent_vectors)-1))
        else:
            raise ValueError("Method must be 'tsne', 'pca', or 'umap'")
        
        # Reduce dimensions
        reduced_vectors = reducer.fit_transform(latent_vectors)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Color by reconstruction error
        scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], 
                            c=errors, cmap='viridis', alpha=0.6, s=30)
        plt.colorbar(scatter, label='Reconstruction Error')
        
        # Add labels for normal vs anomaly
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
        plt.title(f'Latent Space Visualization ({method.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return {
            'reduced_vectors': reduced_vectors,
            'method': method,
            'explained_variance': reducer.explained_variance_ratio_ if hasattr(reducer, 'explained_variance_ratio_') else None
        }
    
    def compute_shap_attributions(self, sample_data, background_data=None, nsamples=100):
        """
        Compute SHAP values for autoencoder reconstruction
        
        Args:
            sample_data: Sample data to explain (tensor or numpy array)
            background_data: Background data for SHAP (if None, uses random samples)
            nsamples: Number of samples for SHAP computation
            
        Returns:
            Dictionary containing SHAP attributions
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install shap package for this functionality.")
            return {}
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Convert to tensor if needed
        if isinstance(sample_data, np.ndarray):
            sample_data = torch.FloatTensor(sample_data)
        
        sample_data = sample_data.to(self.device)
        
        # Create background data if not provided
        if background_data is None and hasattr(self, 'reconstruction_errors'):
            # Use normal samples as background
            normal_mask = self.reconstruction_errors['predictions'] == 0
            if np.any(normal_mask):
                background_data = self.reconstruction_errors['inputs'][normal_mask][:nsamples]
            else:
                background_data = np.random.normal(0, 1, (nsamples, sample_data.shape[-1]))
        
        if isinstance(background_data, np.ndarray):
            background_data = torch.FloatTensor(background_data)
        
        background_data = background_data.to(self.device)
        
        # Define model function for SHAP
        def model_function(x):
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                reconstruction = self.model(x_tensor)
                # Return reconstruction error as explanation target
                error = torch.mean((x_tensor - reconstruction) ** 2, dim=1)
            return error.cpu().numpy()
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(model_function, background_data.cpu().numpy())
        
        # Compute SHAP values
        shap_values = explainer.shap_values(sample_data.cpu().numpy(), nsamples=nsamples)
        
        # Store results
        sample_key = f"sample_{hash(str(sample_data.cpu().numpy()))}"
        self.feature_attributions[sample_key] = {
            'shap_values': shap_values,
            'sample_data': sample_data.cpu().numpy(),
            'method': 'shap_kernel',
            'nsamples': nsamples
        }
        
        return self.feature_attributions[sample_key]
    
    def compute_integrated_gradients(self, sample_data, baseline=None, n_steps=50):
        """
        Compute Integrated Gradients for feature attribution
        
        Args:
            sample_data: Sample data to explain (tensor or numpy array)
            baseline: Baseline for integration (if None, uses zeros)
            n_steps: Number of steps for integration
            
        Returns:
            Dictionary containing Integrated Gradients attributions
        """
        if not CAPTUM_AVAILABLE:
            print("Captum not available. Install captum package for this functionality.")
            return {}
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Convert to tensor if needed
        if isinstance(sample_data, np.ndarray):
            sample_data = torch.FloatTensor(sample_data)
        
        sample_data = sample_data.to(self.device)
        sample_data.requires_grad_(True)
        
        # Create baseline if not provided
        if baseline is None:
            baseline = torch.zeros_like(sample_data)
        elif isinstance(baseline, np.ndarray):
            baseline = torch.FloatTensor(baseline)
        
        baseline = baseline.to(self.device)
        
        # Create Integrated Gradients explainer
        ig = IntegratedGradients(self.model)
        
        # Compute attributions
        with torch.enable_grad():
            attributions = ig.attribute(sample_data, baseline, n_steps=n_steps)
        
        # Store results
        sample_key = f"sample_{hash(str(sample_data.cpu().numpy()))}"
        self.feature_attributions[sample_key] = {
            'attributions': attributions.cpu().numpy(),
            'sample_data': sample_data.cpu().numpy(),
            'baseline': baseline.cpu().numpy(),
            'method': 'integrated_gradients',
            'n_steps': n_steps
        }
        
        return self.feature_attributions[sample_key]
    
    def explain_anomaly_sample(self, sample_data, feature_names=None, threshold=None):
        """
        Generate comprehensive explanation for an anomalous sample
        
        Args:
            sample_data: Sample data to explain (tensor or numpy array)
            feature_names: List of feature names
            threshold: Anomaly threshold
            
        Returns:
            Dictionary containing comprehensive explanation
        """
        # Convert to tensor if needed
        if isinstance(sample_data, np.ndarray):
            sample_data = torch.FloatTensor(sample_data)
        
        sample_data = sample_data.to(self.device)
        
        # Compute reconstruction
        with torch.no_grad():
            reconstructed = self.model(sample_data)
            reconstruction_error = torch.mean((sample_data - reconstructed) ** 2)
        
        # Determine if it's an anomaly
        is_anomaly = False
        if threshold is not None:
            is_anomaly = reconstruction_error.item() > threshold
        
        # Get per-feature errors
        per_feature_error = (sample_data - reconstructed) ** 2
        per_feature_error = per_feature_error.cpu().numpy().flatten()
        
        # Rank features by contribution to error
        if feature_names is None:
            feature_names = [f'feature_{i:02d}' for i in range(len(per_feature_error))]
        
        feature_contributions = list(zip(feature_names, per_feature_error))
        feature_contributions.sort(key=lambda x: x[1], reverse=True)
        
        # Generate explanation text
        explanation = {
            'reconstruction_error': reconstruction_error.item(),
            'is_anomaly': is_anomaly,
            'threshold': threshold,
            'top_contributing_features': feature_contributions[:10],
            'feature_contributions': feature_contributions,
            'original_sample': sample_data.cpu().numpy().flatten(),
            'reconstructed_sample': reconstructed.cpu().numpy().flatten(),
            'per_feature_errors': per_feature_error
        }
        
        # Add attribution if available
        sample_key = f"sample_{hash(str(sample_data.cpu().numpy()))}"
        if sample_key in self.feature_attributions:
            explanation['attributions'] = self.feature_attributions[sample_key]
        
        return explanation
    
    def find_similar_samples(self, sample_data, data_loader, top_k=5, metric='euclidean'):
        """
        Find most similar samples to a given sample
        
        Args:
            sample_data: Sample data to find similarities for
            data_loader: DataLoader containing dataset
            top_k: Number of similar samples to return
            metric: Distance metric ('euclidean', 'cosine')
            
        Returns:
            Dictionary containing similar samples analysis
        """
        # Convert to tensor if needed
        if isinstance(sample_data, np.ndarray):
            sample_data = torch.FloatTensor(sample_data)
        
        sample_data = sample_data.to(self.device)
        
        # Get latent representation of sample
        with torch.no_grad():
            sample_latent = self.model.encoder(sample_data).cpu().numpy()
        
        # Collect latent representations from dataset
        all_latent = []
        all_data = []
        all_labels = []
        all_errors = []
        
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(data_loader):
                data = data.to(self.device)
                
                # Get latent representations
                latent = self.model.encoder(data)
                
                all_latent.extend(latent.cpu().numpy())
                all_data.extend(data.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Compute reconstruction errors
                reconstructed = self.model(data)
                error = torch.mean((data - reconstructed) ** 2, dim=1)
                all_errors.extend(error.cpu().numpy())
        
        all_latent = np.array(all_latent)
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        all_errors = np.array(all_errors)
        
        # Compute distances
        if metric == 'euclidean':
            distances = np.linalg.norm(all_latent - sample_latent, axis=1)
        elif metric == 'cosine':
            distances = cosine_distances(sample_latent.reshape(1, -1), all_latent)[0]
        else:
            raise ValueError("Metric must be 'euclidean' or 'cosine'")
        
        # Get top-k most similar (excluding self if in dataset)
        top_indices = np.argsort(distances)[:top_k + 1]  # +1 to exclude self
        
        similar_samples = []
        for idx in top_indices:
            if np.linalg.norm(all_data[idx] - sample_data.cpu().numpy()) > 1e-6:  # Exclude self
                similar_samples.append({
                    'index': idx,
                    'distance': distances[idx],
                    'data': all_data[idx],
                    'label': all_labels[idx],
                    'reconstruction_error': all_errors[idx],
                    'latent_representation': all_latent[idx]
                })
                
                if len(similar_samples) >= top_k:
                    break
        
        return {
            'sample_latent': sample_latent,
            'similar_samples': similar_samples,
            'metric': metric,
            'top_k': top_k
        }
    
    def generate_explanation_report(self, sample_data, feature_names=None, threshold=None):
        """
        Generate comprehensive explanation report for a sample
        
        Args:
            sample_data: Sample data to explain
            feature_names: List of feature names
            threshold: Anomaly threshold
            
        Returns:
            Formatted explanation report
        """
        explanation = self.explain_anomaly_sample(sample_data, feature_names, threshold)
        
        report = f"""
=== AUTOENCODER EXPLANATION REPORT ===

Sample Analysis:
- Reconstruction Error: {explanation['reconstruction_error']:.6f}
- Is Anomaly: {explanation['is_anomaly']}
- Threshold: {explanation['threshold'] if explanation['threshold'] else 'Not specified'}

Top 10 Contributing Features to Reconstruction Error:
"""
        
        for i, (feature, error) in enumerate(explanation['top_contributing_features'], 1):
            report += f"{i:2d}. {feature:<20} {error:.6f}\n"
        
        report += f"""
Feature Statistics:
- Total Features: {len(explanation['feature_contributions'])}
- Mean Feature Error: {np.mean([e for _, e in explanation['feature_contributions']]):.6f}
- Max Feature Error: {np.max([e for _, e in explanation['feature_contributions']]):.6f}
- Min Feature Error: {np.min([e for _, e in explanation['feature_contributions']]):.6f}

Sample vs Reconstruction Comparison:
"""
        
        # Add sample vs reconstruction comparison for top features
        original = explanation['original_sample']
        reconstructed = explanation['reconstructed_sample']
        
        for i, (feature, error) in enumerate(explanation['top_contributing_features'][:5], 1):
            feature_idx = feature_names.index(feature) if feature_names and feature in feature_names else i-1
            if feature_idx < len(original):
                report += f"- {feature}: Original={original[feature_idx]:.4f}, Reconstructed={reconstructed[feature_idx]:.4f}, Error={error:.6f}\n"
        
        return report
