"""
XAI Visualization Suite

Provides comprehensive visualization capabilities for:
- Feature distribution plots
- Correlation heatmaps
- Feature importance visualizations
- Attack type pattern comparisons
- Interactive plots using Plotly
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Optional Plotly imports for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class XAIPlotter:
    """
    Comprehensive visualization suite for XAI analysis
    """
    
    def __init__(self):
        self.color_palette = {
            'normal': '#2E86AB',
            'anomaly': '#A23B72',
            'attack_1': '#F18F01',
            'attack_2': '#C73E1D',
            'attack_3': '#6A4C93',
            'attack_4': '#FF6B6B',
            'attack_5': '#4ECDC4'
        }
        
    def plot_feature_distributions(self, df: pd.DataFrame, features: list = None, 
                                 label_col: str = 'label', plot_type: str = 'histogram',
                                 save_path: str = None) -> None:
        """
        Plot feature distributions comparing normal vs anomalous traffic
        
        Args:
            df: Input DataFrame
            features: List of features to plot (if None, plot top 10)
            label_col: Name of the label column
            plot_type: Type of plot ('histogram', 'box', 'violin')
            save_path: Path to save the plot
        """
        if features is None:
            # Select top 10 numeric features
            numeric_features = df.select_dtypes(include=[np.number]).columns.drop(label_col, errors='ignore')
            features = list(numeric_features[:10])
        
        # Separate normal and anomaly samples
        normal_samples = df[df[label_col] == 0]
        anomaly_samples = df[df[label_col] != 0]
        
        # Create subplots
        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(features):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            if plot_type == 'histogram':
                ax.hist(normal_samples[feature].dropna(), alpha=0.7, bins=30, 
                       label='Normal', color=self.color_palette['normal'])
                ax.hist(anomaly_samples[feature].dropna(), alpha=0.7, bins=30, 
                       label='Anomaly', color=self.color_palette['anomaly'])
            elif plot_type == 'box':
                data_to_plot = [normal_samples[feature].dropna(), anomaly_samples[feature].dropna()]
                ax.boxplot(data_to_plot, labels=['Normal', 'Anomaly'])
            elif plot_type == 'violin':
                combined_data = pd.concat([
                    normal_samples[feature].dropna().to_frame().assign(Type='Normal'),
                    anomaly_samples[feature].dropna().to_frame().assign(Type='Anomaly')
                ])
                sns.violinplot(data=combined_data, x='Type', y=feature, ax=ax)
            
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency' if plot_type == 'histogram' else 'Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame, 
                               save_path: str = None, figsize: tuple = (12, 10)) -> None:
        """
        Plot correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            save_path: Path to save the plot
            figsize: Figure size tuple
        """
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_scores: dict, top_k: int = 20,
                              save_path: str = None) -> None:
        """
        Plot feature importance scores
        
        Args:
            importance_scores: Dictionary of feature importance scores
            top_k: Number of top features to display
            save_path: Path to save the plot
        """
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features, scores = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), scores)
        
        # Color bars based on importance level
        for i, bar in enumerate(bars):
            if scores[i] >= np.percentile(scores, 80):
                bar.set_color('#A23B72')
            elif scores[i] >= np.percentile(scores, 60):
                bar.set_color('#F18F01')
            else:
                bar.set_color('#2E86AB')
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importance Scores')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(scores):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attack_type_patterns(self, df: pd.DataFrame, features: list = None,
                                label_col: str = 'label', save_path: str = None) -> None:
        """
        Plot feature patterns across different attack types
        
        Args:
            df: Input DataFrame
            features: List of features to plot
            label_col: Name of the label column
            save_path: Path to save the plot
        """
        if features is None:
            numeric_features = df.select_dtypes(include=[np.number]).columns.drop(label_col, errors='ignore')
            features = list(numeric_features[:6])  # Plot 6 features
        
        # Get unique labels
        labels = sorted(df[label_col].unique())
        
        # Create subplots
        n_features = len(features)
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(features):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Plot box plots for each attack type
            data_to_plot = []
            labels_to_plot = []
            
            for label in labels:
                label_data = df[df[label_col] == label][feature].dropna()
                if len(label_data) > 0:
                    data_to_plot.append(label_data)
                    if label == 0:
                        labels_to_plot.append('Normal')
                    else:
                        labels_to_plot.append(f'Attack_{label}')
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
                
                # Color the boxes
                for j, box in enumerate(bp['boxes']):
                    if labels_to_plot[j] == 'Normal':
                        box.set_facecolor(self.color_palette['normal'])
                    else:
                        color_key = f'attack_{j % 5 + 1}'
                        box.set_facecolor(self.color_palette.get(color_key, '#FF6B6B'))
            
            ax.set_title(f'{feature} by Attack Type')
            ax.set_xlabel('Attack Type')
            ax.set_ylabel('Feature Value')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_analysis(self, df: pd.DataFrame, label_col: str = 'label',
                         n_components: int = 2, save_path: str = None) -> None:
        """
        Plot PCA analysis for dimensionality reduction visualization
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
            n_components: Number of PCA components
            save_path: Path to save the plot
        """
        # Prepare data
        numeric_features = df.select_dtypes(include=[np.number]).columns.drop(label_col, errors='ignore')
        X = df[numeric_features].fillna(df[numeric_features].median())
        y = df[label_col]
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        pca_df['label'] = y
        pca_df['label_str'] = y.apply(lambda x: 'Normal' if x == 0 else f'Attack_{x}')
        
        # Plot
        if n_components == 2:
            plt.figure(figsize=(10, 8))
            
            for label in pca_df['label'].unique():
                subset = pca_df[pca_df['label'] == label]
                color = self.color_palette['normal'] if label == 0 else self.color_palette.get(f'attack_{label}', '#FF6B6B')
                plt.scatter(subset['PC1'], subset['PC2'], alpha=0.6, label=f'Class {label}', color=color)
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.title('PCA Analysis of Network Traffic Features')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        elif n_components == 3:
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            for label in pca_df['label'].unique():
                subset = pca_df[pca_df['label'] == label]
                color = self.color_palette['normal'] if label == 0 else self.color_palette.get(f'attack_{label}', '#FF6B6B')
                ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], alpha=0.6, label=f'Class {label}', color=color)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
            ax.set_title('3D PCA Analysis of Network Traffic Features')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_feature_importance(self, importance_scores: dict, 
                                         save_path: str = None):
        """
        Create interactive feature importance plot using Plotly
        
        Args:
            importance_scores: Dictionary of feature importance scores
            save_path: Path to save the HTML plot
            
        Returns:
            Plotly figure object or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Use plot_feature_importance() for static plots.")
            return None
            
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        # Create interactive plot
        fig = go.Figure(data=[
            go.Bar(
                x=list(scores),
                y=list(features),
                orientation='h',
                marker=dict(
                    color=list(scores),
                    colorscale='Viridis',
                    showscale=True
                ),
                text=[f'{score:.4f}' for score in scores],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Interactive Feature Importance Scores',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=800,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_interactive_correlation_matrix(self, correlation_matrix: pd.DataFrame,
                                          save_path: str = None):
        """
        Create interactive correlation heatmap using Plotly
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            save_path: Path to save the HTML plot
            
        Returns:
            Plotly figure object or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Use plot_correlation_heatmap() for static plots.")
            return None
            
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Interactive Feature Correlation Matrix',
            xaxis_title='Features',
            yaxis_title='Features',
            width=800,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
