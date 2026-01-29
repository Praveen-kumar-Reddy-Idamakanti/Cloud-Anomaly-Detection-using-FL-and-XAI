"""
Integrated Visualization Plots for XAI Phase 4

Provides specialized visualization capabilities for two-stage integrated explanations:
- End-to-end explanation visualizations
- Feature evolution plots
- Attack progression visualizations
- Comparative analysis charts
- Integrated dashboard components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

class IntegratedPlotter:
    """
    Specialized plotter for integrated two-stage explanations
    """
    
    def __init__(self):
        self.color_palette = {
            'normal': '#2E86AB',
            'anomaly': '#A23B72',
            'attack': '#F18F01',
            'dos': '#C73E1D',
            'portscan': '#6A4C93',
            'bruteforce': '#FF6B6B',
            'webattack': '#4ECDC4',
            'infiltration': '#95E77E'
        }
        
    def plot_two_stage_summary(self, integrated_explanation, save_path=None):
        """
        Plot comprehensive two-stage explanation summary
        
        Args:
            integrated_explanation: Dictionary from IntegratedExplainer.explain_two_stage_prediction
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall status
        status = integrated_explanation['unified_analysis']['overall_status']
        consistency = integrated_explanation['unified_analysis']['stage_correlation']['correlation_level']
        confidence = integrated_explanation['unified_analysis']['confidence_analysis']['overall_confidence']
        risk_level = integrated_explanation['unified_analysis']['risk_assessment']['risk_level']
        
        # Create status visualization
        ax1.text(0.5, 0.7, f"Status: {status['status']}", ha='center', va='center', 
                fontsize=16, fontweight='bold', transform=ax1.transAxes)
        ax1.text(0.5, 0.5, f"Consistency: {consistency}", ha='center', va='center', 
                fontsize=12, transform=ax1.transAxes)
        ax1.text(0.5, 0.3, f"Confidence: {confidence:.1%}", ha='center', va='center', 
                fontsize=12, transform=ax1.transAxes)
        ax1.text(0.5, 0.1, f"Risk Level: {risk_level}", ha='center', va='center', 
                fontsize=12, fontweight='bold', color='red' if risk_level in ['HIGH', 'CRITICAL'] else 'green',
                transform=ax1.transAxes)
        ax1.set_title('Overall Analysis')
        ax1.axis('off')
        
        # 2. Stage comparison
        stage1_result = "ANOMALY" if integrated_explanation['stage1_anomaly']['is_anomaly'] else "NORMAL"
        stage2_result = integrated_explanation['stage2_attack']['prediction_result']['predicted_classes'][0]
        
        stages = ['Stage 1\n(Autoencoder)', 'Stage 2\n(Classifier)']
        results = [stage1_result, stage2_result]
        colors = [self.color_palette['anomaly'] if stage1_result == "ANOMALY" else self.color_palette['normal'],
                  self.color_palette.get(stage2_result.lower(), self.color_palette['attack'])]
        
        bars = ax2.bar(stages, [1, 1], color=colors, alpha=0.7)
        ax2.set_ylim(0, 1.2)
        ax2.set_title('Stage Results')
        ax2.set_ylabel('Detection Status')
        
        # Add result labels
        for bar, result in zip(bars, results):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    result, ha='center', va='bottom', fontweight='bold')
        
        # 3. Confidence analysis
        anomaly_conf = integrated_explanation['unified_analysis']['confidence_analysis']['anomaly_confidence']
        attack_conf = integrated_explanation['unified_analysis']['confidence_analysis']['attack_confidence']
        
        confidences = [anomaly_conf, attack_conf]
        labels = ['Anomaly\nConfidence', 'Attack\nConfidence']
        
        bars = ax3.bar(labels, confidences, color=[self.color_palette['anomaly'], self.color_palette['attack']], alpha=0.7)
        ax3.set_ylim(0, 1)
        ax3.set_title('Confidence Analysis')
        ax3.set_ylabel('Confidence Score')
        
        # Add value labels
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{conf:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Top features
        top_features = integrated_explanation['unified_analysis']['feature_importance']['top_features'][:8]
        features = [f[0] for f in top_features]
        importance = [f[1] for f in top_features]
        
        bars = ax4.barh(range(len(features)), importance, color=self.color_palette['attack'], alpha=0.7)
        ax4.set_yticks(range(len(features)))
        ax4.set_yticklabels([f'F_{f.split("_")[1]}' for f in features])
        ax4.set_xlabel('Combined Importance')
        ax4.set_title('Top Features (Both Stages)')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_evolution(self, comparative_analysis, save_path=None):
        """
        Plot feature evolution from normal to anomaly to attack
        
        Args:
            comparative_analysis: Dictionary from IntegratedExplainer.create_comparative_analysis
            save_path: Path to save the plot
        """
        feature_evolution = comparative_analysis['feature_evolution']
        top_features = feature_evolution['top_evolving_features'][:10]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature evolution heatmap
        features = [f[0] for f in top_features]
        evolution_matrix = []
        
        for feature, data in top_features:
            evolution_matrix.append([
                data['normal_mean'],
                data['anomaly_mean'],
                data['attack_mean']
            ])
        
        evolution_matrix = np.array(evolution_matrix)
        
        sns.heatmap(evolution_matrix, 
                   xticklabels=['Normal', 'Anomaly', 'Attack'],
                   yticklabels=[f'F_{f.split("_")[1]}' for f in features],
                   annot=True, fmt='.3f', cmap='viridis', ax=ax1)
        ax1.set_title('Feature Evolution Heatmap')
        
        # 2. Evolution patterns distribution
        evolution_summary = feature_evolution['evolution_summary']
        patterns = list(evolution_summary['pattern_counts'].keys())
        counts = list(evolution_summary['pattern_counts'].values())
        
        bars = ax2.bar(patterns, counts, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Evolution Pattern')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Evolution Pattern Distribution')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(count), ha='center', va='bottom')
        
        # 3. Total evolution magnitude
        total_evolution = [data['total_evolution'] for _, data in top_features]
        
        bars = ax3.barh(range(len(features)), total_evolution, color='coral', alpha=0.7)
        ax3.set_yticks(range(len(features)))
        ax3.set_yticklabels([f'F_{f.split("_")[1]}' for f in features])
        ax3.set_xlabel('Total Evolution')
        ax3.set_title('Feature Evolution Magnitude')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)
        
        # 4. Evolution pattern summary
        pattern_percentages = evolution_summary['pattern_percentages']
        dominant_pattern = evolution_summary['dominant_pattern']
        
        summary_text = f"""
Evolution Pattern Summary:

Dominant Pattern: {dominant_pattern}

Pattern Distribution:
"""
        
        for pattern, percentage in pattern_percentages.items():
            summary_text += f"• {pattern}: {percentage:.1%}\n"
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Evolution Pattern Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_attack_progression(self, attack_progression, save_path=None):
        """
        Plot attack progression analysis
        
        Args:
            attack_progression: Dictionary from IntegratedExplainer.analyze_attack_progression
            save_path: Path to save the plot
        """
        attack_patterns = attack_progression['attack_patterns']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Sample count per attack type
        attack_types = list(attack_patterns.keys())
        sample_counts = [attack_patterns[attack]['sample_count'] for attack in attack_types]
        colors = [self.color_palette.get(attack.lower(), self.color_palette['attack']) for attack in attack_types]
        
        bars = ax1.bar(attack_types, sample_counts, color=colors, alpha=0.7)
        ax1.set_xlabel('Attack Type')
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Sample Distribution by Attack Type')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, count in zip(bars, sample_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    str(count), ha='center', va='bottom')
        
        # 2. Average anomaly error per attack type
        avg_errors = [attack_patterns[attack]['avg_anomaly_error'] for attack in attack_types]
        
        bars = ax2.bar(attack_types, avg_errors, color=colors, alpha=0.7)
        ax2.set_xlabel('Attack Type')
        ax2.set_ylabel('Average Anomaly Error')
        ax2.set_title('Anomaly Error by Attack Type')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Consistency rate per attack type
        consistency_rates = [attack_patterns[attack]['consistency_rate'] for attack in attack_types]
        
        bars = ax3.bar(attack_types, consistency_rates, color=colors, alpha=0.7)
        ax3.set_xlabel('Attack Type')
        ax3.set_ylabel('Consistency Rate')
        ax3.set_title('Stage Consistency by Attack Type')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bar, rate in zip(bars, consistency_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # 4. Common features analysis
        # Get top 5 most common features across all attacks
        all_common_features = {}
        for attack, patterns in attack_patterns.items():
            for feature, count in patterns['common_features'][:5]:
                if feature not in all_common_features:
                    all_common_features[feature] = 0
                all_common_features[feature] += count
        
        # Sort and get top features
        sorted_features = sorted(all_common_features.items(), key=lambda x: x[1], reverse=True)[:10]
        features = [f'F_{f.split("_")[1]}' for f, _ in sorted_features]
        counts = [count for _, count in sorted_features]
        
        bars = ax4.barh(range(len(features)), counts, color='lightblue', alpha=0.7)
        ax4.set_yticks(range(len(features)))
        ax4.set_yticklabels(features)
        ax4.set_xlabel('Frequency')
        ax4.set_title('Most Common Important Features')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_comparative_analysis(self, comparative_analysis, save_path=None):
        """
        Plot comprehensive comparative analysis
        
        Args:
            comparative_analysis: Dictionary from IntegratedExplainer.create_comparative_analysis
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confidence patterns
        conf_patterns = comparative_analysis['confidence_patterns']
        anomaly_conf = conf_patterns['anomaly_confidence_distribution']
        attack_conf = conf_patterns['attack_confidence_distribution']
        
        groups = ['Normal', 'Anomaly', 'Attack']
        anomaly_means = [anomaly_conf['normal_samples']['mean'], anomaly_conf['anomaly_samples']['mean'], anomaly_conf['attack_samples']['mean']]
        attack_means = [attack_conf['normal_samples']['mean'], attack_conf['anomaly_samples']['mean'], attack_conf['attack_samples']['mean']]
        
        x = np.arange(len(groups))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, anomaly_means, width, label='Anomaly Confidence', color=self.color_palette['anomaly'], alpha=0.7)
        bars2 = ax1.bar(x + width/2, attack_means, width, label='Attack Confidence', color=self.color_palette['attack'], alpha=0.7)
        
        ax1.set_xlabel('Sample Type')
        ax1.set_ylabel('Mean Confidence')
        ax1.set_title('Confidence Patterns by Sample Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(groups)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk progression
        risk_prog = comparative_analysis['risk_progression']
        risk_levels = risk_prog['risk_levels']
        
        # Create stacked bar chart for risk distribution
        bottom = np.zeros(len(groups))
        for risk_level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
            values = [risk_levels[sample][risk_level] for sample in ['normal_samples', 'anomaly_samples', 'attack_samples']]
            bars = ax2.bar(groups, values, bottom=bottom, label=risk_level, 
                       color=['green', 'yellow', 'orange', 'red'][['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].index(risk_level)],
                       alpha=0.7)
            bottom += values
        
        ax2.set_xlabel('Sample Type')
        ax2.set_ylabel('Risk Distribution')
        ax2.set_title('Risk Progression by Sample Type')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Stage transitions
        transitions = comparative_analysis['stage_transitions']
        transition_rates = [transitions['normal_to_anomaly_rate'], transitions['anomaly_to_attack_rate'], transitions['normal_to_attack_rate']]
        transition_labels = ['Normal→Anomaly', 'Anomaly→Attack', 'Normal→Attack']
        
        bars = ax3.bar(transition_labels, transition_rates, color=['blue', 'orange', 'red'], alpha=0.7)
        ax3.set_ylabel('Transition Rate')
        ax3.set_title('Stage Transition Rates')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars, transition_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        # 4. Summary statistics
        summary_text = f"""
Comparative Analysis Summary:

Overall Consistency: {attack_progression['overall_consistency']:.2f}

Stage Transition Patterns:
• Normal → Anomaly: {transitions['normal_to_anomaly_rate']:.1%}
• Anomaly → Attack: {transitions['anomaly_to_attack_rate']:.1%}
• Normal → Attack: {transitions['normal_to_attack_rate']:.1%}

Risk Progression: {risk_prog['risk_progression_trend']}
"""
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title('Comparative Analysis Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_integrated_dashboard(self, integrated_explanation, save_path=None):
        """
        Create comprehensive integrated dashboard
        
        Args:
            integrated_explanation: Dictionary from IntegratedExplainer.explain_two_stage_prediction
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall status (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        status = integrated_explanation['unified_analysis']['overall_status']
        risk_level = integrated_explanation['unified_analysis']['risk_assessment']['risk_level']
        
        ax1.text(0.5, 0.7, f"STATUS", ha='center', va='center', fontsize=14, fontweight='bold', transform=ax1.transAxes)
        ax1.text(0.5, 0.5, f"{status['status']}", ha='center', va='center', fontsize=16, fontweight='bold',
                color='red' if risk_level in ['HIGH', 'CRITICAL'] else 'green', transform=ax1.transAxes)
        ax1.text(0.5, 0.3, f"Risk: {risk_level}", ha='center', va='center', fontsize=12,
                color='red' if risk_level in ['HIGH', 'CRITICAL'] else 'orange' if risk_level == 'MEDIUM' else 'green',
                transform=ax1.transAxes)
        ax1.set_title('Overall Status')
        ax1.axis('off')
        
        # 2. Stage results (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        stage1_result = "ANOMALY" if integrated_explanation['stage1_anomaly']['is_anomaly'] else "NORMAL"
        stage2_result = integrated_explanation['stage2_attack']['prediction_result']['predicted_classes'][0]
        
        stages = ['Stage 1', 'Stage 2']
        results = [stage1_result, stage2_result]
        colors = [self.color_palette['anomaly'] if stage1_result == "ANOMALY" else self.color_palette['normal'],
                  self.color_palette.get(stage2_result.lower(), self.color_palette['attack'])]
        
        bars = ax2.bar(stages, [1, 1], color=colors, alpha=0.7)
        ax2.set_title('Stage Results')
        ax2.set_ylim(0, 1.2)
        
        for bar, result in zip(bars, results):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    result, ha='center', va='bottom', fontweight='bold')
        
        # 3. Confidence analysis (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        confidence = integrated_explanation['unified_analysis']['confidence_analysis']
        
        metrics = ['Anomaly\nConf', 'Attack\nConf', 'Overall\nConf']
        values = [confidence['anomaly_confidence'], confidence['attack_confidence'], confidence['overall_confidence']]
        
        bars = ax3.bar(metrics, values, color=[self.color_palette['anomaly'], self.color_palette['attack'], 'purple'], alpha=0.7)
        ax3.set_ylim(0, 1)
        ax3.set_title('Confidence Analysis')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Top features (top right, spanning 2 columns)
        ax4 = fig.add_subplot(gs[0, 3:])
        top_features = integrated_explanation['unified_analysis']['feature_importance']['top_features'][:8]
        features = [f'F_{f.split("_")[1]}' for f, _ in top_features]
        importance = [imp for _, imp in top_features]
        
        bars = ax4.barh(range(len(features)), importance, color='teal', alpha=0.7)
        ax4.set_yticks(range(len(features)))
        ax4.set_yticklabels(features)
        ax4.set_xlabel('Importance')
        ax4.set_title('Top Features (Combined)')
        ax4.invert_yaxis()
        ax4.grid(True, alpha=0.3)
        
        # 5. Anomaly details (middle left)
        ax5 = fig.add_subplot(gs[1, 0])
        anomaly_exp = integrated_explanation['stage1_anomaly']
        
        ax5.text(0.5, 0.8, "Anomaly Detection", ha='center', va='center', fontsize=12, fontweight='bold', transform=ax5.transAxes)
        ax5.text(0.5, 0.6, f"Error: {anomaly_exp['reconstruction_error']:.6f}", ha='center', va='center', transform=ax5.transAxes)
        ax5.text(0.5, 0.4, f"Threshold: {anomaly_exp['threshold']:.6f}" if anomaly_exp['threshold'] else "Threshold: N/A", 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.text(0.5, 0.2, f"Top Feature: {anomaly_exp['top_contributing_features'][0][0]}", 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Stage 1 Details')
        ax5.axis('off')
        
        # 6. Attack details (middle middle)
        ax6 = fig.add_subplot(gs[1, 1])
        attack_exp = integrated_explanation['stage2_attack']
        
        ax6.text(0.5, 0.8, "Attack Classification", ha='center', va='center', fontsize=12, fontweight='bold', transform=ax6.transAxes)
        ax6.text(0.5, 0.6, f"Attack: {attack_exp['prediction_result']['predicted_classes'][0]}", ha='center', va='center', transform=ax6.transAxes)
        ax6.text(0.5, 0.4, f"Confidence: {attack_exp['prediction_result']['confidence_scores'][0]:.1%}", ha='center', va='center', transform=ax6.transAxes)
        ax6.text(0.5, 0.2, f"2nd Best: {self._get_second_best(attack_exp)}", ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Stage 2 Details')
        ax6.axis('off')
        
        # 7. Risk assessment (middle right)
        ax7 = fig.add_subplot(gs[1, 2])
        risk_assess = integrated_explanation['unified_analysis']['risk_assessment']
        
        ax7.text(0.5, 0.7, "Risk Assessment", ha='center', va='center', fontsize=12, fontweight='bold', transform=ax7.transAxes)
        ax7.text(0.5, 0.5, f"Level: {risk_assess['risk_level']}", ha='center', va='center', fontsize=14, fontweight='bold',
                color='red' if risk_assess['risk_level'] in ['HIGH', 'CRITICAL'] else 'orange' if risk_assess['risk_level'] == 'MEDIUM' else 'green',
                transform=ax7.transAxes)
        ax7.text(0.5, 0.3, f"Anomaly: {risk_assess['anomaly_contribution']}", ha='center', va='center', transform=ax7.transAxes)
        ax7.text(0.5, 0.1, f"Multiplier: {risk_assess['risk_multiplier']:.1f}x", ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Risk Assessment')
        ax7.axis('off')
        
        # 8. Recommendations (middle right)
        ax8 = fig.add_subplot(gs[1, 3])
        recommendations = self._generate_recommendations(integrated_explanation)
        
        ax8.text(0.5, 0.5, recommendations, ha='center', va='center', fontsize=10,
                transform=ax8.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax8.set_title('Recommendations')
        ax8.axis('off')
        
        # 9. Feature comparison (bottom, spanning all columns)
        ax9 = fig.add_subplot(gs[2, :])
        
        # Get top 10 features from both stages
        anomaly_features = dict(integrated_explanation['stage1_anomaly']['top_contributing_features'][:10])
        
        # Create comparison table
        feature_names = list(anomaly_features.keys())[:10]
        anomaly_errors = list(anomaly_features.values())[:10]
        
        # Create table data
        table_data = []
        for i, feature in enumerate(feature_names):
            table_data.append([
                f'F_{feature.split("_")[1]}',
                f'{anomaly_errors[i]:.6f}',
                f'{integrated_explanation["unified_analysis"]["feature_importance"]["combined_importance"].get(feature, 0):.4f}',
                self._get_feature_impact(feature, integrated_explanation)
            ])
        
        # Create table
        table = ax9.table(cellText=table_data,
                         colLabels=['Feature', 'Anomaly Error', 'Combined Importance', 'Impact'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.15, 0.25, 0.25, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(4):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4CAF50')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if j % 2 == 0:  # Alternating columns
                        cell.set_facecolor('#E3F2FD')
                    else:
                        cell.set_facecolor('#F5F5F5')
        
        ax9.set_title('Feature Analysis Comparison')
        ax9.axis('off')
        
        plt.suptitle('Two-Stage Integrated Explanation Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _get_second_best(self, attack_explanation):
        """Get second best attack type"""
        probabilities = attack_explanation['prediction_result']['probabilities'][0]
        attack_names = self.classifier_explainer.attack_type_names
        sorted_indices = np.argsort(probabilities)[-2:]
        return f"{attack_names[sorted_indices[0]]} ({probabilities[sorted_indices[0]]:.1%})"
    
    def _get_feature_impact(self, feature, integrated_explanation):
        """Get feature impact description"""
        combined_importance = integrated_explanation['unified_analysis']['feature_importance']['combined_importance']
        if feature in combined_importance:
            importance = combined_importance[feature]
            if importance > 0.05:
                return "HIGH"
            elif importance > 0.01:
                return "MEDIUM"
            else:
                return "LOW"
        return "UNKNOWN"
    
    def _generate_recommendations(self, integrated_explanation):
        """Generate recommendations based on integrated analysis"""
        risk_level = integrated_explanation['unified_analysis']['risk_assessment']['risk_level']
        predicted_attack = integrated_explanation['stage2_attack']['prediction_result']['predicted_classes'][0]
        
        if risk_level == 'CRITICAL':
            return " IMMEDIATE ACTION REQUIRED\n• Isolate affected systems\n• Conduct full security audit\n• Enable enhanced monitoring"
        elif risk_level == 'HIGH':
            return " HIGH PRIORITY ACTIONS\n• Block suspicious IPs\n• Scan for compromises\n• Update security rules"
        elif risk_level == 'MEDIUM':
            return " STANDARD RESPONSE\n• Investigate suspicious activity\n• Review system logs\n• Update monitoring"
        else:
            return " ROUTINE MONITORING\n• Continue normal monitoring\n• Log event for analysis"
    
    def _get_second_best(self, attack_explanation):
        """Get second best attack type"""
        probabilities = attack_explanation['prediction_result']['probabilities'][0]
        attack_names = self.classifier_explainer.attack_type_names
        sorted_indices = np.argsort(probabilities)[-2:]
        return f"{attack_names[sorted_indices[0]]} ({probabilities[sorted_indices[0]]:.1%})"
    
    def plot_interactive_integrated_dashboard(self, integrated_explanation, save_path=None):
        """
        Create interactive integrated dashboard (if Plotly available)
        
        Args:
            integrated_explanation: Dictionary from IntegratedExplainer.explain_two_stage_prediction
            save_path: Path to save the HTML plot
            
        Returns:
            Plotly figure or None if Plotly not available
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Use static plots instead.")
            return None
        
        # Create subplot layout
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                'Overall Status',
                'Stage Results',
                'Confidence Analysis',
                'Risk Assessment',
                'Top Features',
                'Recommendations'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "table"}]
            ]
        )
        
        # 1. Overall status indicator
        status = integrated_explanation['unified_analysis']['overall_status']
        risk_level = integrated_explanation['unified_analysis']['risk_assessment']['risk_level']
        
        fig.add_trace(go.Indicator(
            mode="number+gauge+number",
            value=integrated_explanation['unified_analysis']['confidence_analysis']['overall_confidence'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Confidence"},
            gauge={'axis': {'range': [None, 100]}},
            number={'suffix': "%"}
        ), row=1, col=1)
        
        # 2. Stage results
        stage1_result = "ANOMALY" if integrated_explanation['stage1_anomaly']['is_anomaly'] else "NORMAL"
        stage2_result = integrated_explanation['stage2_attack']['prediction_result']['predicted_classes'][0]
        
        fig.add_trace(go.Bar(
            x=['Stage 1', 'Stage 2'],
            y=[1, 1],
            marker_color=['red' if stage1_result == "ANOMALY" else 'green',
                         'orange' if stage2_result != "Normal" else 'blue'],
            name=stage1_result
        ), row=1, col=2)
        
        # 3. Confidence analysis
        confidence = integrated_explanation['unified_analysis']['confidence_analysis']
        
        fig.add_trace(go.Bar(
            x=['Anomaly', 'Attack', 'Overall'],
            y=[confidence['anomaly_confidence'], confidence['attack_confidence'], confidence['overall_confidence']],
            marker_color=['red', 'orange', 'purple'],
            name='Confidence'
        ), row=1, col=3)
        
        # 4. Risk assessment pie chart
        fig.add_trace(go.Pie(
            labels=['Low', 'Medium', 'High', 'Critical'],
            values=[0.1, 0.2, 0.5, 0.2],  # Placeholder values
            marker_colors=['green', 'yellow', 'orange', 'red']
        ), row=2, col=1)
        
        # 5. Top features
        top_features = integrated_explanation['unified_analysis']['feature_importance']['top_features'][:8]
        features = [f'F_{f.split("_")[1]}' for f, _ in top_features]
        importance = [imp for _, imp in top_features]
        
        fig.add_trace(go.Bar(
            x=features,
            y=importance,
            marker_color='teal',
            name='Importance'
        ), row=2, col=2)
        
        # 6. Recommendations table
        recommendations = self._generate_recommendations(integrated_explanation)
        
        fig.add_trace(go.Table(
            header=dict(values=['Recommendations']),
            cells=dict(values=[[recommendations]])
        ), row=2, col=3)
        
        fig.update_layout(
            title_text="Two-Stage Integrated Explanation Dashboard",
            height=800,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
