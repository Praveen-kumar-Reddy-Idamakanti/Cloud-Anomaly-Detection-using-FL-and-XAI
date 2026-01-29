"""
XAI Dashboard Component

Provides interactive dashboard capabilities for:
- Real-time feature exploration
- Dynamic explanation generation
- Interactive visualizations
- Comprehensive XAI insights
"""

import pandas as pd
import numpy as np

# Optional Plotly imports for interactive dashboard
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Optional IPython import for Jupyter notebook support
try:
    from IPython.display import display, HTML
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class XAIDashboard:
    """
    Interactive dashboard for XAI analysis and visualization
    """
    
    def __init__(self):
        self.data = None
        self.feature_importance = None
        self.correlation_matrix = None
        self.baseline_patterns = None
        
    def load_data(self, df: pd.DataFrame, label_col: str = 'label') -> None:
        """
        Load data into the dashboard
        
        Args:
            df: Input DataFrame
            label_col: Name of the label column
        """
        self.data = df.copy()
        self.label_col = label_col
        
    def load_analysis_results(self, feature_importance: dict = None, 
                           correlation_matrix: pd.DataFrame = None,
                           baseline_patterns: dict = None) -> None:
        """
        Load pre-computed analysis results
        
        Args:
            feature_importance: Feature importance scores
            correlation_matrix: Correlation matrix
            baseline_patterns: Baseline pattern analysis
        """
        self.feature_importance = feature_importance
        self.correlation_matrix = correlation_matrix
        self.baseline_patterns = baseline_patterns
    
    def create_overview_dashboard(self):
        """
        Create overview dashboard with key insights
        
        Returns:
            Plotly figure with overview dashboard or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Cannot create interactive dashboard.")
            return None
            
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data first.")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Class Distribution', 'Feature Importance Top 10', 
                          'Data Quality Overview', 'Sample Statistics'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "table"}, {"type": "indicator"}]]
        )
        
        # Class distribution pie chart
        class_counts = self.data[self.label_col].value_counts()
        class_labels = ['Normal' if x == 0 else f'Attack_{x}' for x in class_counts.index]
        
        fig.add_trace(
            go.Pie(labels=class_labels, values=class_counts.values, name="Class Distribution"),
            row=1, col=1
        )
        
        # Feature importance bar chart
        if self.feature_importance:
            top_features = list(self.feature_importance.keys())[:10]
            top_scores = [self.feature_importance[f] for f in top_features]
            
            fig.add_trace(
                go.Bar(x=top_scores, y=top_features, orientation='h', name="Feature Importance"),
                row=1, col=2
            )
        
        # Data quality table
        quality_data = [
            ['Total Samples', len(self.data)],
            ['Total Features', len(self.data.columns) - 1],
            ['Missing Values', self.data.isnull().sum().sum()],
            ['Duplicate Rows', self.data.duplicated().sum()],
            ['Normal Samples', len(self.data[self.data[self.label_col] == 0])],
            ['Anomalous Samples', len(self.data[self.data[self.label_col] != 0])]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[list(row) for row in zip(*quality_data)])
            ),
            row=2, col=1
        )
        
        # Sample statistics indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=len(self.data),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Total Samples"},
                delta={'reference': 10000}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="XAI Overview Dashboard",
            showlegend=False
        )
        
        return fig
    
    def create_feature_explorer(self):
        """
        Create interactive feature explorer dashboard
        
        Returns:
            Plotly figure for feature exploration or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Cannot create interactive dashboard.")
            return None
            
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data first.")
        
        # Get numeric features
        numeric_features = self.data.select_dtypes(include=[np.number]).columns.drop(self.label_col, errors='ignore')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Distribution', 'Feature vs Label', 
                          'Correlation Heatmap', 'Feature Statistics'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "table"}]]
        )
        
        # Feature distribution (for first feature)
        if len(numeric_features) > 0:
            feature = numeric_features[0]
            
            # Normal vs anomaly distribution
            normal_data = self.data[self.data[self.label_col] == 0][feature]
            anomaly_data = self.data[self.data[self.label_col] != 0][feature]
            
            fig.add_trace(
                go.Histogram(x=normal_data, name="Normal", opacity=0.7),
                row=1, col=1
            )
            fig.add_trace(
                go.Histogram(x=anomaly_data, name="Anomaly", opacity=0.7),
                row=1, col=1
            )
            
            # Feature vs label scatter
            fig.add_trace(
                go.Scatter(
                    x=self.data[feature], 
                    y=self.data[self.label_col],
                    mode='markers',
                    name=f'{feature} vs Label',
                    opacity=0.6
                ),
                row=1, col=2
            )
        
        # Correlation heatmap (subset)
        if self.correlation_matrix is not None:
            # Take first 10 features for visualization
            corr_subset = self.correlation_matrix.iloc[:10, :10]
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_subset.values,
                    x=corr_subset.columns,
                    y=corr_subset.columns,
                    colorscale='RdBu',
                    zmid=0
                ),
                row=2, col=1
            )
        
        # Feature statistics table
        if len(numeric_features) > 0:
            feature_stats = []
            for feature in numeric_features[:5]:  # Show top 5 features
                stats = [
                    feature,
                    f"{self.data[feature].mean():.3f}",
                    f"{self.data[feature].std():.3f}",
                    f"{self.data[feature].min():.3f}",
                    f"{self.data[feature].max():.3f}",
                    f"{self.data[feature].isnull().sum()}"
                ]
                feature_stats.append(stats)
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['Feature', 'Mean', 'Std', 'Min', 'Max', 'Missing']),
                    cells=dict(values=[list(row) for row in zip(*feature_stats)])
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Feature Explorer Dashboard"
        )
        
        return fig
    
    def create_model_explanation_dashboard(self):
        """
        Create model explanation dashboard
        
        Returns:
            Plotly figure for model explanations or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Cannot create interactive dashboard.")
            return None
            
        if self.feature_importance is None:
            raise ValueError("Feature importance not loaded. Call load_analysis_results first.")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Importance Ranking', 'Importance Distribution',
                          'Top Features Analysis', 'Feature Categories'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # Feature importance ranking
        features = list(self.feature_importance.keys())[:15]
        scores = [self.feature_importance[f] for f in features]
        
        fig.add_trace(
            go.Bar(x=scores, y=features, orientation='h', name="Importance Score"),
            row=1, col=1
        )
        
        # Importance distribution
        all_scores = list(self.feature_importance.values())
        fig.add_trace(
            go.Histogram(x=all_scores, name="Score Distribution", nbinsx=20),
            row=1, col=2
        )
        
        # Top features analysis (importance vs feature index)
        top_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        feature_indices = list(range(len(top_features)))
        feature_scores = [score for _, score in top_features]
        
        fig.add_trace(
            go.Scatter(
                x=feature_indices, 
                y=feature_scores,
                mode='markers+lines',
                name="Top Features Trend"
            ),
            row=2, col=1
        )
        
        # Feature categories (based on importance quartiles)
        all_scores = list(self.feature_importance.values())
        q1, q2, q3 = np.percentile(all_scores, [25, 50, 75])
        
        high_importance = sum(1 for s in all_scores if s >= q3)
        medium_high = sum(1 for s in all_scores if q2 <= s < q3)
        medium_low = sum(1 for s in all_scores if q1 <= s < q2)
        low_importance = sum(1 for s in all_scores if s < q1)
        
        fig.add_trace(
            go.Pie(
                labels=['High', 'Medium-High', 'Medium-Low', 'Low'],
                values=[high_importance, medium_high, medium_low, low_importance],
                name="Importance Categories"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Model Explanation Dashboard",
            showlegend=False
        )
        
        return fig
    
    def generate_insights_summary(self) -> str:
        """
        Generate textual insights summary
        
        Returns:
            Formatted insights summary
        """
        if self.data is None:
            return "No data loaded for analysis."
        
        insights = []
        
        # Dataset insights
        total_samples = len(self.data)
        normal_samples = len(self.data[self.data[self.label_col] == 0])
        anomaly_samples = len(self.data[self.data[self.label_col] != 0])
        anomaly_rate = (anomaly_samples / total_samples) * 100
        
        insights.append(f"📊 **Dataset Overview**")
        insights.append(f"- Total samples: {total_samples:,}")
        insights.append(f"- Normal samples: {normal_samples:,} ({(normal_samples/total_samples)*100:.1f}%)")
        insights.append(f"- Anomalous samples: {anomaly_samples:,} ({anomaly_rate:.1f}%)")
        insights.append(f"- Features analyzed: {len(self.data.columns) - 1}")
        
        # Feature importance insights
        if self.feature_importance:
            top_feature = max(self.feature_importance.items(), key=lambda x: x[1])
            insights.append(f"\n🎯 **Feature Importance Insights**")
            insights.append(f"- Most important feature: {top_feature[0]} (score: {top_feature[1]:.4f})")
            insights.append(f"- Features with high importance: {sum(1 for v in self.feature_importance.values() if v > np.percentile(list(self.feature_importance.values()), 75))}")
        
        # Data quality insights
        missing_values = self.data.isnull().sum().sum()
        duplicate_rows = self.data.duplicated().sum()
        
        insights.append(f"\n🔍 **Data Quality Insights**")
        insights.append(f"- Missing values: {missing_values}")
        insights.append(f"- Duplicate rows: {duplicate_rows}")
        insights.append(f"- Data completeness: {((total_samples * len(self.data.columns) - missing_values) / (total_samples * len(self.data.columns))) * 100:.1f}%")
        
        # Correlation insights
        if self.correlation_matrix is not None:
            high_corr_pairs = 0
            for i in range(len(self.correlation_matrix.columns)):
                for j in range(i+1, len(self.correlation_matrix.columns)):
                    if abs(self.correlation_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs += 1
            
            insights.append(f"\n🔗 **Correlation Insights**")
            insights.append(f"- Highly correlated feature pairs: {high_corr_pairs}")
        
        return "\n".join(insights)
    
    def export_dashboard_html(self, filename: str = "xai_dashboard.html") -> None:
        """
        Export complete dashboard to HTML file
        
        Args:
            filename: Output HTML filename
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Cannot export interactive dashboard.")
            print("Use static plots from XAIPlotter instead.")
            return
            
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data first.")
        
        # Create all dashboard components
        overview_fig = self.create_overview_dashboard()
        explorer_fig = self.create_feature_explorer()
        explanation_fig = self.create_model_explanation_dashboard()
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>XAI Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard-section {{ margin: 30px 0; }}
                .insights {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                h1, h2 {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1>XAI Analysis Dashboard</h1>
            
            <div class="dashboard-section">
                <h2>Overview</h2>
                {overview_fig.to_html(include_plotlyjs=False, div_id="overview")}
            </div>
            
            <div class="dashboard-section">
                <h2>Feature Explorer</h2>
                {explorer_fig.to_html(include_plotlyjs=False, div_id="explorer")}
            </div>
            
            <div class="dashboard-section">
                <h2>Model Explanations</h2>
                {explanation_fig.to_html(include_plotlyjs=False, div_id="explanation")}
            </div>
            
            <div class="dashboard-section insights">
                <h2>Key Insights</h2>
                <pre>{self.generate_insights_summary()}</pre>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Dashboard exported to {filename}")
    
    def display_dashboard(self) -> None:
        """
        Display dashboard in Jupyter notebook
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data first.")
        
        if not IPYTHON_AVAILABLE:
            print("IPython not available. Use export_dashboard_html() to save dashboard to file.")
            print("\nKey Insights:")
            print(self.generate_insights_summary())
            return
        
        # Display insights
        display(HTML(f"<h3>XAI Dashboard Insights</h3><pre>{self.generate_insights_summary()}</pre>"))
        
        # Display overview
        overview_fig = self.create_overview_dashboard()
        display(overview_fig)
        
        # Display feature explorer
        explorer_fig = self.create_feature_explorer()
        display(explorer_fig)
        
        # Display model explanations
        explanation_fig = self.create_model_explanation_dashboard()
        display(explanation_fig)
