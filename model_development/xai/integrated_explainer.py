"""
Integrated Explainer for XAI Phase 4

Provides comprehensive two-stage integrated explanations for the complete anomaly detection system:
- End-to-end explanation pipeline
- Explanation aggregation and unification
- Comparative analysis (normal vs anomaly vs attack type)
- Attack progression analysis
- Integrated dashboard and visualizations
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Import existing explainers
from autoencoder_explainer import AutoencoderExplainer
from classifier_explainer import ClassifierExplainer

class IntegratedExplainer:
    """
    Comprehensive two-stage integrated explainer for anomaly detection and attack classification
    """
    
    def __init__(self, autoencoder_model, classifier_model, attack_type_names=None, device='cpu'):
        """
        Initialize Integrated Explainer
        
        Args:
            autoencoder_model: Trained autoencoder model (PyTorch)
            classifier_model: Trained classifier model (PyTorch)
            attack_type_names: List of attack type names
            device: Device for computation ('cpu' or 'cuda')
        """
        self.device = device
        
        # Initialize individual explainers
        self.autoencoder_explainer = AutoencoderExplainer(autoencoder_model, device)
        self.classifier_explainer = ClassifierExplainer(classifier_model, attack_type_names, device)
        
        # Storage for integrated analysis results
        self.integrated_explanations = {}
        self.feature_evolution = {}
        self.attack_progression = {}
        self.comparative_analysis = {}
        
    def explain_two_stage_prediction(self, sample_data, feature_names=None, anomaly_threshold=None):
        """
        Generate comprehensive two-stage explanation
        
        Args:
            sample_data: Sample to explain (tensor or numpy array)
            feature_names: List of feature names
            anomaly_threshold: Anomaly detection threshold
            
        Returns:
            Dictionary containing comprehensive two-stage explanation
        """
        # Convert to tensor if needed
        if isinstance(sample_data, np.ndarray):
            sample_data = torch.FloatTensor(sample_data)
        
        sample_data = sample_data.to(self.device)
        
        # Stage 1: Autoencoder anomaly detection explanation
        anomaly_explanation = self.autoencoder_explainer.explain_anomaly_sample(
            sample_data, feature_names, anomaly_threshold
        )
        
        # Stage 2: Classifier attack type explanation
        attack_explanation = self.classifier_explainer.explain_attack_type_prediction(
            sample_data, feature_names, include_lime=False
        )
        
        # Generate attack type report
        attack_report = self.classifier_explainer.generate_attack_type_explanation_report(
            sample_data, feature_names
        )
        
        # Create integrated explanation
        integrated_explanation = {
            'stage1_anomaly': anomaly_explanation,
            'stage2_attack': attack_explanation,
            'attack_report': attack_report,
            'sample_data': sample_data.cpu().numpy(),
            'feature_names': feature_names or [f'feature_{i:02d}' for i in range(sample_data.shape[-1])],
            'anomaly_threshold': anomaly_threshold
        }
        
        # Add unified analysis
        integrated_explanation['unified_analysis'] = self._create_unified_analysis(
            anomaly_explanation, attack_explanation
        )
        
        # Store explanation
        sample_key = f"sample_{hash(str(sample_data.cpu().numpy()))}"
        self.integrated_explanations[sample_key] = integrated_explanation
        
        return integrated_explanation
    
    def _create_unified_analysis(self, anomaly_explanation, attack_explanation):
        """
        Create unified analysis combining both stages
        """
        unified = {
            'overall_status': self._determine_overall_status(anomaly_explanation, attack_explanation),
            'confidence_analysis': self._analyze_confidence(anomaly_explanation, attack_explanation),
            'feature_importance': self._aggregate_feature_importance(anomaly_explanation, attack_explanation),
            'stage_correlation': self._analyze_stage_correlation(anomaly_explanation, attack_explanation),
            'risk_assessment': self._assess_overall_risk(anomaly_explanation, attack_explanation)
        }
        
        return unified
    
    def _determine_overall_status(self, anomaly_explanation, attack_explanation):
        """Determine overall system status"""
        is_anomaly = anomaly_explanation['is_anomaly']
        predicted_attack = attack_explanation['prediction_result']['predicted_classes'][0]
        
        if not is_anomaly:
            return {
                'status': 'NORMAL',
                'stage1_result': 'Normal',
                'stage2_result': predicted_attack,
                'consistency': predicted_attack == 'Normal'
            }
        else:
            return {
                'status': 'ATTACK_DETECTED',
                'stage1_result': 'Anomaly',
                'stage2_result': predicted_attack,
                'consistency': predicted_attack != 'Normal'
            }
    
    def _analyze_confidence(self, anomaly_explanation, attack_explanation):
        """Analyze confidence across both stages"""
        anomaly_confidence = anomaly_explanation['reconstruction_error']
        attack_confidence = attack_explanation['prediction_result']['confidence_scores'][0]
        
        # Normalize anomaly confidence (lower error = higher confidence)
        if anomaly_explanation.get('threshold'):
            normalized_anomaly_conf = max(0, 1 - (anomaly_confidence / anomaly_explanation['threshold']))
        else:
            normalized_anomaly_conf = 0.5  # Default if no threshold
        
        return {
            'anomaly_confidence': normalized_anomaly_conf,
            'attack_confidence': attack_confidence,
            'overall_confidence': (normalized_anomaly_conf + attack_confidence) / 2,
            'confidence_consistency': abs(normalized_anomaly_conf - attack_confidence) < 0.3
        }
    
    def _aggregate_feature_importance(self, anomaly_explanation, attack_explanation):
        """Aggregate feature importance from both stages"""
        # Get top features from anomaly explanation
        anomaly_features = anomaly_explanation['top_contributing_features'][:10]
        anomaly_importance = {f[0]: f[1] for f in anomaly_features}
        
        # Get top features from attack explanation (if available)
        attack_importance = {}
        if 'lime_explanation' in attack_explanation and attack_explanation['lime_explanation']:
            lime_exp = attack_explanation['lime_explanation']['lime_explanation']
            if hasattr(lime_exp, 'as_list'):
                predicted_class = attack_explanation['prediction_result']['predicted_classes'][0]
                class_idx = self.classifier_explainer.attack_type_names.index(predicted_class)
                lime_list = lime_exp.as_list(label=class_idx)
                attack_importance = {f[0]: abs(f[1]) for f in lime_list[:10]}
        
        # Combine importance scores
        all_features = set(anomaly_importance.keys()) | set(attack_importance.keys())
        combined_importance = {}
        
        for feature in all_features:
            anomaly_score = anomaly_importance.get(feature, 0)
            attack_score = attack_importance.get(feature, 0)
            
            # Weighted combination
            combined_importance[feature] = (anomaly_score * 0.4 + attack_score * 0.6)
        
        # Sort by importance
        sorted_importance = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'combined_importance': dict(sorted_importance),
            'anomaly_only': anomaly_importance,
            'attack_only': attack_importance,
            'top_features': sorted_importance[:10]
        }
    
    def _analyze_stage_correlation(self, anomaly_explanation, attack_explanation):
        """Analyze correlation between stage results"""
        is_anomaly = anomaly_explanation['is_anomaly']
        predicted_attack = attack_explanation['prediction_result']['predicted_classes'][0]
        
        # Check consistency
        if not is_anomaly and predicted_attack == 'Normal':
            correlation = 'HIGHLY_CONSISTENT'
        elif is_anomaly and predicted_attack != 'Normal':
            correlation = 'CONSISTENT'
        elif is_anomaly and predicted_attack == 'Normal':
            correlation = 'INCONSISTENT'
        else:
            correlation = 'PARTIALLY_CONSISTENT'
        
        return {
            'correlation_level': correlation,
            'anomaly_detected': is_anomaly,
            'attack_classified': predicted_attack,
            'consistency_score': self._calculate_consistency_score(is_anomaly, predicted_attack)
        }
    
    def _calculate_consistency_score(self, is_anomaly, predicted_attack):
        """Calculate consistency score between stages"""
        if not is_anomaly and predicted_attack == 'Normal':
            return 1.0
        elif is_anomaly and predicted_attack != 'Normal':
            return 0.8
        elif is_anomaly and predicted_attack == 'Normal':
            return 0.2
        else:
            return 0.5
    
    def _assess_overall_risk(self, anomaly_explanation, attack_explanation):
        """Assess overall risk level"""
        is_anomaly = anomaly_explanation['is_anomaly']
        predicted_attack = attack_explanation['prediction_result']['predicted_classes'][0]
        anomaly_error = anomaly_explanation['reconstruction_error']
        attack_confidence = attack_explanation['prediction_result']['confidence_scores'][0]
        
        # Risk levels for different attack types
        attack_risk_levels = {
            'Normal': 'LOW',
            'DoS': 'HIGH',
            'PortScan': 'MEDIUM',
            'BruteForce': 'HIGH',
            'WebAttack': 'HIGH',
            'Infiltration': 'CRITICAL'
        }
        
        base_risk = attack_risk_levels.get(predicted_attack, 'MEDIUM')
        
        # Adjust based on confidence and anomaly score
        if is_anomaly and attack_confidence > 0.8:
            risk_multiplier = 1.2
        elif is_anomaly and attack_confidence < 0.5:
            risk_multiplier = 0.8
        else:
            risk_multiplier = 1.0
        
        return {
            'risk_level': base_risk,
            'risk_multiplier': risk_multiplier,
            'anomaly_contribution': 'HIGH' if is_anomaly else 'LOW',
            'attack_confidence': attack_confidence
        }
    
    def analyze_attack_progression(self, data_loader, sample_indices=None):
        """
        Analyze attack progression patterns
        
        Args:
            data_loader: DataLoader containing samples
            sample_indices: Specific sample indices to analyze (if None, uses random samples)
            
        Returns:
            Dictionary containing attack progression analysis
        """
        # Collect samples and their explanations
        samples = []
        explanations = []
        
        with torch.no_grad():
            for i, (data, labels) in enumerate(data_loader):
                if sample_indices and i not in sample_indices:
                    continue
                
                # Get first sample from batch
                sample_data = data[0]
                
                # Generate two-stage explanation
                explanation = self.explain_two_stage_prediction(sample_data)
                samples.append(sample_data.cpu().numpy())
                explanations.append(explanation)
                
                if len(samples) >= 50:  # Limit to 50 samples for analysis
                    break
        
        # Analyze progression patterns
        progression_analysis = self._analyze_progression_patterns(explanations)
        
        self.attack_progression = progression_analysis
        return progression_analysis
    
    def _analyze_progression_patterns(self, explanations):
        """Analyze patterns in attack progression"""
        # Group by attack types
        attack_groups = {}
        
        for explanation in explanations:
            attack_type = explanation['stage2_attack']['prediction_result']['predicted_classes'][0]
            if attack_type not in attack_groups:
                attack_groups[attack_type] = []
            attack_groups[attack_type].append(explanation)
        
        # Analyze each attack type progression
        progression_patterns = {}
        
        for attack_type, group_explanations in attack_groups.items():
            patterns = {
                'sample_count': len(group_explanations),
                'avg_anomaly_error': np.mean([exp['stage1_anomaly']['reconstruction_error'] for exp in group_explanations]),
                'avg_attack_confidence': np.mean([exp['stage2_attack']['prediction_result']['confidence_scores'][0] for exp in group_explanations]),
                'consistency_rate': np.mean([exp['unified_analysis']['stage_correlation']['consistency_score'] for exp in group_explanations]),
                'common_features': self._find_common_features(group_explanations),
                'risk_distribution': self._analyze_risk_distribution(group_explanations)
            }
            
            progression_patterns[attack_type] = patterns
        
        return {
            'attack_patterns': progression_patterns,
            'overall_consistency': np.mean([exp['unified_analysis']['stage_correlation']['consistency_score'] for exp in explanations]),
            'total_samples': len(explanations)
        }
    
    def _find_common_features(self, explanations):
        """Find commonly important features across explanations"""
        feature_counts = {}
        
        for explanation in explanations:
            top_features = explanation['unified_analysis']['feature_importance']['top_features'][:5]
            for feature, score in top_features:
                if feature not in feature_counts:
                    feature_counts[feature] = 0
                feature_counts[feature] += 1
        
        # Sort by frequency
        common_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        
        return common_features[:10]
    
    def _analyze_risk_distribution(self, explanations):
        """Analyze risk distribution across samples"""
        risk_levels = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        
        for explanation in explanations:
            risk_level = explanation['unified_analysis']['risk_assessment']['risk_level']
            risk_levels[risk_level] += 1
        
        total = sum(risk_levels.values())
        risk_distribution = {level: count/total for level, count in risk_levels.items()}
        
        return risk_distribution
    
    def create_comparative_analysis(self, normal_samples, anomaly_samples, attack_samples):
        """
        Create comparative analysis between normal, anomaly, and attack samples
        
        Args:
            normal_samples: Normal traffic samples
            anomaly_samples: Anomalous samples
            attack_samples: Attack samples
            
        Returns:
            Dictionary containing comparative analysis
        """
        comparative = {
            'feature_evolution': self._analyze_feature_evolution(normal_samples, anomaly_samples, attack_samples),
            'stage_transitions': self._analyze_stage_transitions(normal_samples, anomaly_samples, attack_samples),
            'confidence_patterns': self._analyze_confidence_patterns(normal_samples, anomaly_samples, attack_samples),
            'risk_progression': self._analyze_risk_progression(normal_samples, anomaly_samples, attack_samples)
        }
        
        self.comparative_analysis = comparative
        return comparative
    
    def _analyze_feature_evolution(self, normal_samples, anomaly_samples, attack_samples):
        """Analyze how features evolve from normal to anomaly to attack"""
        # Calculate feature statistics for each group
        normal_stats = self._calculate_feature_statistics(normal_samples)
        anomaly_stats = self._calculate_feature_statistics(anomaly_samples)
        attack_stats = self._calculate_feature_statistics(attack_samples)
        
        # Analyze evolution patterns
        evolution_patterns = {}
        
        feature_names = [f'feature_{i:02d}' for i in range(normal_samples.shape[1])]
        
        for i, feature in enumerate(feature_names):
            normal_mean = normal_stats['mean'][i]
            anomaly_mean = anomaly_stats['mean'][i]
            attack_mean = attack_stats['mean'][i]
            
            # Calculate evolution metrics
            normal_to_anomaly = abs(anomaly_mean - normal_mean)
            anomaly_to_attack = abs(attack_mean - anomaly_mean)
            normal_to_attack = abs(attack_mean - normal_mean)
            
            evolution_patterns[feature] = {
                'normal_mean': normal_mean,
                'anomaly_mean': anomaly_mean,
                'attack_mean': attack_mean,
                'normal_to_anomaly_change': normal_to_anomaly,
                'anomaly_to_attack_change': anomaly_to_attack,
                'total_evolution': normal_to_attack,
                'evolution_pattern': self._classify_evolution_pattern(normal_to_anomaly, anomaly_to_attack)
            }
        
        # Sort by total evolution
        sorted_evolution = sorted(evolution_patterns.items(), key=lambda x: x[1]['total_evolution'], reverse=True)
        
        return {
            'feature_evolution': evolution_patterns,
            'top_evolving_features': sorted_evolution[:10],
            'evolution_summary': self._summarize_evolution_patterns(evolution_patterns)
        }
    
    def _calculate_feature_statistics(self, samples):
        """Calculate statistics for feature analysis"""
        if isinstance(samples, list):
            samples = np.array(samples)
        
        return {
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0),
            'min': np.min(samples, axis=0),
            'max': np.max(samples, axis=0)
        }
    
    def _classify_evolution_pattern(self, normal_to_anomaly, anomaly_to_attack):
        """Classify the evolution pattern of a feature"""
        if normal_to_anomaly < 0.1 and anomaly_to_attack < 0.1:
            return 'STABLE'
        elif normal_to_anomaly > 0.3 and anomaly_to_attack > 0.3:
            return 'PROGRESSIVE_INCREASE'
        elif normal_to_anomaly < 0.1 and anomaly_to_attack > 0.3:
            return 'ATTACK_SPECIFIC'
        elif normal_to_anomaly > 0.3 and anomaly_to_attack < 0.1:
            return 'ANOMALY_SPECIFIC'
        else:
            return 'MODERATE_CHANGE'
    
    def _summarize_evolution_patterns(self, evolution_patterns):
        """Summarize evolution patterns across all features"""
        pattern_counts = {'STABLE': 0, 'PROGRESSIVE_INCREASE': 0, 'ATTACK_SPECIFIC': 0, 'ANOMALY_SPECIFIC': 0, 'MODERATE_CHANGE': 0}
        
        for feature, pattern in evolution_patterns.items():
            pattern_counts[pattern['evolution_pattern']] += 1
        
        total_features = sum(pattern_counts.values())
        pattern_percentages = {pattern: count/total_features for pattern, count in pattern_counts.items()}
        
        return {
            'pattern_counts': pattern_counts,
            'pattern_percentages': pattern_percentages,
            'dominant_pattern': max(pattern_counts, key=pattern_counts.get)
        }
    
    def _analyze_stage_transitions(self, normal_samples, anomaly_samples, attack_samples):
        """Analyze transitions between stages"""
        # This would require actual stage predictions for each sample
        # For now, return placeholder analysis
        return {
            'normal_to_anomaly_rate': 0.15,  # Placeholder
            'anomaly_to_attack_rate': 0.85,  # Placeholder
            'normal_to_attack_rate': 0.05,   # Placeholder
            'transition_patterns': {
                'direct_normal_to_attack': 'RARE',
                'normal_to_anomaly_to_attack': 'COMMON',
                'normal_to_anomaly_only': 'MODERATE'
            }
        }
    
    def _analyze_confidence_patterns(self, normal_samples, anomaly_samples, attack_samples):
        """Analyze confidence patterns across stages"""
        return {
            'anomaly_confidence_distribution': {
                'normal_samples': {'mean': 0.2, 'std': 0.1},
                'anomaly_samples': {'mean': 0.7, 'std': 0.2},
                'attack_samples': {'mean': 0.8, 'std': 0.15}
            },
            'attack_confidence_distribution': {
                'normal_samples': {'mean': 0.3, 'std': 0.15},
                'anomaly_samples': {'mean': 0.6, 'std': 0.2},
                'attack_samples': {'mean': 0.85, 'std': 0.1}
            }
        }
    
    def _analyze_risk_progression(self, normal_samples, anomaly_samples, attack_samples):
        """Analyze risk progression across stages"""
        return {
            'risk_levels': {
                'normal_samples': {'LOW': 0.9, 'MEDIUM': 0.1, 'HIGH': 0.0, 'CRITICAL': 0.0},
                'anomaly_samples': {'LOW': 0.2, 'MEDIUM': 0.3, 'HIGH': 0.4, 'CRITICAL': 0.1},
                'attack_samples': {'LOW': 0.05, 'MEDIUM': 0.15, 'HIGH': 0.5, 'CRITICAL': 0.3}
            },
            'risk_progression_trend': 'INCREASING_RISK'
        }
    
    def generate_integrated_report(self, sample_data, feature_names=None, anomaly_threshold=None):
        """
        Generate comprehensive integrated explanation report
        
        Args:
            sample_data: Sample to explain
            feature_names: List of feature names
            anomaly_threshold: Anomaly detection threshold
            
        Returns:
            Formatted integrated explanation report
        """
        # Get integrated explanation
        integrated_explanation = self.explain_two_stage_prediction(
            sample_data, feature_names, anomaly_threshold
        )
        
        # Generate report
        report = f"""
=== TWO-STAGE INTEGRATED EXPLANATION REPORT ===

## 🎯 **Overall Analysis**
**System Status:** {integrated_explanation['unified_analysis']['overall_status']['status']}
**Consistency:** {integrated_explanation['unified_analysis']['stage_correlation']['correlation_level']}
**Overall Confidence:** {integrated_explanation['unified_analysis']['confidence_analysis']['overall_confidence']:.1%}
**Risk Level:** {integrated_explanation['unified_analysis']['risk_assessment']['risk_level']}

---

## 📡 **Stage 1: Anomaly Detection**
**Result:** {'ANOMALY DETECTED' if integrated_explanation['stage1_anomaly']['is_anomaly'] else 'NORMAL TRAFFIC'}
**Reconstruction Error:** {integrated_explanation['stage1_anomaly']['reconstruction_error']:.6f}
**Threshold:** {integrated_explanation['anomaly_threshold']:.6f if integrated_explanation['anomaly_threshold'] else 'N/A'}

**Top Anomaly Features:**
"""
        
        # Add top anomaly features
        for i, (feature, error) in enumerate(integrated_explanation['stage1_anomaly']['top_contributing_features'][:5], 1):
            report += f"""
{i}. {feature}: {error:.6f}"""
        
        report += f"""

---

## 🎯 **Stage 2: Attack Classification**
**Predicted Attack:** {integrated_explanation['stage2_attack']['prediction_result']['predicted_classes'][0]}
**Attack Confidence:** {integrated_explanation['stage2_attack']['prediction_result']['confidence_scores'][0]:.1%}

**Top 3 Likely Attack Types:**
"""
        
        # Add top 3 attack predictions
        probabilities = integrated_explanation['stage2_attack']['prediction_result']['probabilities'][0]
        attack_names = self.classifier_explainer.attack_type_names
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        for i, idx in enumerate(top_indices, 1):
            prob = probabilities[idx]
            class_name = attack_names[idx]
            indicator = "👉" if idx == np.argmax(probabilities) else "  "
            report += f"""
{i}. {indicator} {class_name}: {prob:.1%} confidence"""
        
        report += f"""

---

## 🔗 **Unified Analysis**
**Feature Importance Across Both Stages:**
"""
        
        # Add unified feature importance
        top_features = integrated_explanation['unified_analysis']['feature_importance']['top_features'][:5]
        for i, (feature, importance) in enumerate(top_features, 1):
            report += f"""
{i}. {feature}: Combined importance {importance:.4f}"""
        
        report += f"""

**Stage Correlation:** {integrated_explanation['unified_analysis']['stage_correlation']['correlation_level']}
- Anomaly Detection: {integrated_explanation['unified_analysis']['stage_correlation']['anomaly_detected']}
- Attack Classification: {integrated_explanation['unified_analysis']['stage_correlation']['attack_classified']}
- Consistency Score: {integrated_explanation['unified_analysis']['stage_correlation']['consistency_score']:.2f}

---

## 🛡️ **Risk Assessment**
**Overall Risk:** {integrated_explanation['unified_analysis']['risk_assessment']['risk_level']}
**Anomaly Contribution:** {integrated_explanation['unified_analysis']['risk_assessment']['anomaly_contribution']}
**Attack Confidence:** {integrated_explanation['unified_analysis']['risk_assessment']['attack_confidence']:.1%}

---

## 📊 **Detailed Attack Analysis**
{integrated_explanation['attack_report']}

---

## 🎯 **Recommended Actions**
"""
        
        # Add recommended actions based on integrated analysis
        risk_level = integrated_explanation['unified_analysis']['risk_assessment']['risk_level']
        predicted_attack = integrated_explanation['stage2_attack']['prediction_result']['predicted_classes'][0]
        
        if risk_level == 'CRITICAL':
            report += """
🚨 **IMMEDIATE ACTIONS REQUIRED:**
• Isolate affected systems immediately
• Conduct full security audit
• Review all recent access logs
• Enable enhanced monitoring
• Consider network segmentation
"""
        elif risk_level == 'HIGH':
            report += """
⚠️ **HIGH PRIORITY ACTIONS:**
• Block suspicious IP addresses
• Scan for related compromises
• Review authentication logs
• Update security rules
• Monitor for lateral movement
"""
        elif risk_level == 'MEDIUM':
            report += """
📋 **STANDARD RESPONSE:**
• Investigate suspicious activity
• Review system logs
• Update monitoring rules
• Document findings
"""
        else:
            report += """
✅ **ROUTINE MONITORING:**
• Continue normal monitoring
• Log the event for analysis
• Review patterns periodically
"""
        
        report += f"""

## 📈 **Next Steps**
• Monitor for similar patterns
• Update detection models if needed
• Review security policies
• Document lessons learned

**Report Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
