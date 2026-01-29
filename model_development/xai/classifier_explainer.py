"""
Classifier Explainer for XAI Phase 3

Provides comprehensive explainability capabilities for attack type classification:
- Multi-class classification explanations
- LIME-based local explanations
- Attack type-specific feature importance
- Confidence and uncertainty analysis
- Decision boundary visualization
- Misclassification analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Optional imports for advanced explanations
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class ClassifierExplainer:
    """
    Comprehensive explainer for attack type classification
    """
    
    def __init__(self, classifier_model, attack_type_names=None, device='cpu'):
        """
        Initialize Classifier Explainer
        
        Args:
            classifier_model: Trained classification model (PyTorch)
            attack_type_names: List of attack type names
            device: Device for computation ('cpu' or 'cuda')
        """
        self.model = classifier_model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Attack type mapping
        self.attack_type_names = attack_type_names or [
            'Normal', 'DoS', 'PortScan', 'BruteForce', 'WebAttack', 'Infiltration'
        ]
        
        # Storage for analysis results
        self.explanations = {}
        self.feature_importance = {}
        self.confidence_analysis = {}
        self.misclassification_analysis = {}
        
    def predict_with_confidence(self, data):
        """
        Make predictions with confidence scores
        
        Args:
            data: Input data (tensor or numpy array)
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        self.model.eval()
        
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
        
        data = data.to(self.device)
        
        # Ensure data is 2D
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        with torch.no_grad():
            # Get model outputs (logits)
            outputs = self.model(data)
            
            # Convert to probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predictions and confidence
            predictions = torch.argmax(probabilities, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
            
            return {
                'predictions': predictions.cpu().numpy(),
                'probabilities': probabilities.cpu().numpy(),
                'confidence_scores': confidence_scores.cpu().numpy(),
                'predicted_classes': [self.attack_type_names[pred] for pred in predictions.cpu().numpy()]
            }
    
    def explain_attack_type_lime(self, sample_data, training_data=None, feature_names=None, num_features=10):
        """
        Generate LIME explanation for attack type classification
        
        Args:
            sample_data: Sample to explain (numpy array)
            training_data: Training data for LIME (if None, uses sample data)
            feature_names: List of feature names
            num_features: Number of features to include in explanation
            
        Returns:
            Dictionary containing LIME explanation
        """
        if not LIME_AVAILABLE:
            print("LIME not available. Install lime package for this functionality.")
            return {}
        
        # Convert to numpy if tensor
        if hasattr(sample_data, 'cpu'):
            sample_data = sample_data.cpu().numpy()
        
        if sample_data.ndim == 1:
            sample_data = sample_data.reshape(1, -1)
        
        # Use sample data as training data if not provided
        if training_data is None:
            training_data = sample_data
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i:02d}' for i in range(sample_data.shape[1])]
        
        # Create LIME explainer
        def predict_fn(x):
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                outputs = self.model(x_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
        
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=self.attack_type_names,
            mode='classification',
            discretize_continuous=True
        )
        
        # Generate explanation
        explanation = explainer.explain_instance(
            sample_data[0],
            predict_fn,
            num_features=num_features,
            top_labels=len(self.attack_type_names)
        )
        
        # Store explanation
        sample_key = f"sample_{hash(str(sample_data.tobytes()))}"
        self.explanations[sample_key] = {
            'lime_explanation': explanation,
            'sample_data': sample_data,
            'method': 'lime',
            'feature_names': feature_names
        }
        
        return self.explanations[sample_key]
    
    def compute_attack_type_feature_importance(self, data_loader, method='permutation'):
        """
        Compute feature importance for each attack type
        
        Args:
            data_loader: DataLoader containing labeled data
            method: Importance computation method ('permutation', 'shap', 'coefficients')
            
        Returns:
            Dictionary containing attack type-specific feature importance
        """
        self.model.eval()
        
        # Collect all data and predictions
        all_data = []
        all_labels = []
        all_predictions = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                predictions = torch.argmax(outputs, dim=1)
                
                all_data.extend(data.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
        
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        
        # Compute importance per attack type
        attack_importance = {}
        
        for class_idx, class_name in enumerate(self.attack_type_names):
            # Get samples of this class
            class_mask = (all_labels == class_idx) | (all_predictions == class_idx)
            class_data = all_data[class_mask]
            class_labels = all_labels[class_mask]
            
            if len(class_data) == 0:
                continue
            
            if method == 'permutation':
                importance = self._compute_permutation_importance(class_data, class_labels)
            elif method == 'shap' and SHAP_AVAILABLE:
                importance = self._compute_shap_importance(class_data, class_labels)
            else:
                importance = self._compute_coefficient_importance(class_data, class_labels)
            
            attack_importance[class_name] = importance
        
        self.feature_importance = attack_importance
        return attack_importance
    
    def _compute_permutation_importance(self, data, labels):
        """Compute permutation feature importance"""
        # Get baseline accuracy
        baseline_pred = self.predict_with_confidence(data)['predictions']
        baseline_accuracy = np.mean(baseline_pred == labels)
        
        importance_scores = {}
        
        for feature_idx in range(data.shape[1]):
            # Permute feature
            permuted_data = data.copy()
            permuted_data[:, feature_idx] = np.random.permutation(permuted_data[:, feature_idx])
            
            # Compute accuracy with permuted feature
            permuted_pred = self.predict_with_confidence(permuted_data)['predictions']
            permuted_accuracy = np.mean(permuted_pred == labels)
            
            # Importance is the drop in accuracy
            importance = baseline_accuracy - permuted_accuracy
            importance_scores[f'feature_{feature_idx:02d}'] = importance
        
        return importance_scores
    
    def _compute_shap_importance(self, data, labels):
        """Compute SHAP feature importance"""
        if not SHAP_AVAILABLE:
            return {}
        
        # Create SHAP explainer
        def predict_fn(x):
            x_tensor = torch.FloatTensor(x).to(self.device)
            with torch.no_grad():
                outputs = self.model(x_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
        
        explainer = shap.KernelExplainer(predict_fn, data[:100])  # Use subset for efficiency
        shap_values = explainer.shap_values(data[:50])  # Explain subset
        
        # Compute mean absolute SHAP values per feature
        if isinstance(shap_values, list):
            # Multi-class case
            mean_shap = np.mean([np.abs(class_shap) for class_shap in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values)
        
        importance_scores = {}
        for i in range(mean_shap.shape[1]):
            importance_scores[f'feature_{i:02d}'] = np.mean(mean_shap[:, i])
        
        return importance_scores
    
    def _compute_coefficient_importance(self, data, labels):
        """Compute importance using simple statistical correlation"""
        importance_scores = {}
        
        for feature_idx in range(data.shape[1]):
            feature_values = data[:, feature_idx]
            
            # Compute correlation with each class
            correlations = []
            for class_idx in range(len(self.attack_type_names)):
                class_mask = labels == class_idx
                if np.sum(class_mask) > 1:
                    correlation = np.corrcoef(feature_values, class_mask.astype(float))[0, 1]
                    correlations.append(abs(correlation))
                else:
                    correlations.append(0)
            
            # Importance is maximum correlation across classes
            importance_scores[f'feature_{feature_idx:02d}'] = max(correlations)
        
        return importance_scores
    
    def analyze_prediction_confidence(self, data_loader):
        """
        Analyze prediction confidence and uncertainty
        
        Args:
            data_loader: DataLoader containing test data
            
        Returns:
            Dictionary containing confidence analysis
        """
        self.model.eval()
        
        all_confidences = []
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                confidence_scores = torch.max(probabilities, dim=1)[0]
                
                all_confidences.extend(confidence_scores.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_confidences = np.array(all_confidences)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Compute confidence statistics per class
        confidence_stats = {}
        for class_idx, class_name in enumerate(self.attack_type_names):
            class_mask = all_predictions == class_idx
            if np.sum(class_mask) > 0:
                class_confidences = all_confidences[class_mask]
                confidence_stats[class_name] = {
                    'mean_confidence': np.mean(class_confidences),
                    'std_confidence': np.std(class_confidences),
                    'min_confidence': np.min(class_confidences),
                    'max_confidence': np.max(class_confidences),
                    'num_samples': len(class_confidences)
                }
        
        # Compute uncertainty metrics
        entropy = -np.sum(all_probabilities * np.log(all_probabilities + 1e-8), axis=1)
        
        confidence_analysis = {
            'overall_stats': {
                'mean_confidence': np.mean(all_confidences),
                'std_confidence': np.std(all_confidences),
                'mean_entropy': np.mean(entropy),
                'std_entropy': np.std(entropy)
            },
            'per_class_stats': confidence_stats,
            'low_confidence_samples': {
                'indices': np.where(all_confidences < 0.5)[0].tolist(),
                'count': np.sum(all_confidences < 0.5),
                'percentage': (np.sum(all_confidences < 0.5) / len(all_confidences)) * 100
            },
            'high_uncertainty_samples': {
                'indices': np.where(entropy > np.percentile(entropy, 90))[0].tolist(),
                'count': np.sum(entropy > np.percentile(entropy, 90)),
                'percentage': (np.sum(entropy > np.percentile(entropy, 90)) / len(entropy)) * 100
            }
        }
        
        self.confidence_analysis = confidence_analysis
        return confidence_analysis
    
    def analyze_misclassifications(self, data_loader):
        """
        Analyze misclassification patterns
        
        Args:
            data_loader: DataLoader containing test data
            
        Returns:
            Dictionary containing misclassification analysis
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_data = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_data.extend(data.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        all_data = np.array(all_data)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Find misclassified samples
        misclassified_mask = all_predictions != all_labels
        misclassified_indices = np.where(misclassified_mask)[0]
        
        # Analyze misclassification patterns
        misclassification_patterns = {}
        
        for true_class in range(len(self.attack_type_names)):
            for pred_class in range(len(self.attack_type_names)):
                if true_class != pred_class:
                    # Find samples misclassified as pred_class when actually true_class
                    pattern_mask = (all_labels == true_class) & (all_predictions == pred_class)
                    pattern_indices = np.where(pattern_mask)[0]
                    
                    if len(pattern_indices) > 0:
                        true_class_name = self.attack_type_names[true_class]
                        pred_class_name = self.attack_type_names[pred_class]
                        
                        misclassification_patterns[f"{true_class_name}_as_{pred_class_name}"] = {
                            'count': len(pattern_indices),
                            'indices': pattern_indices.tolist(),
                            'avg_confidence': np.mean(all_probabilities[pattern_indices, pred_class]),
                            'sample_data': all_data[pattern_indices]
                        }
        
        # Find most confused class pairs
        confused_pairs = []
        for i in range(len(self.attack_type_names)):
            for j in range(len(self.attack_type_names)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': self.attack_type_names[i],
                        'predicted_class': self.attack_type_names[j],
                        'count': cm[i, j],
                        'rate': cm[i, j] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
                    })
        
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        
        misclassification_analysis = {
            'confusion_matrix': cm,
            'total_misclassifications': len(misclassified_indices),
            'misclassification_rate': len(misclassified_indices) / len(all_labels),
            'misclassification_patterns': misclassification_patterns,
            'most_confused_pairs': confused_pairs[:10],
            'misclassified_samples': {
                'indices': misclassified_indices.tolist(),
                'true_labels': all_labels[misclassified_indices].tolist(),
                'predicted_labels': all_predictions[misclassified_indices].tolist(),
                'probabilities': all_probabilities[misclassified_indices].tolist()
            }
        }
        
        self.misclassification_analysis = misclassification_analysis
        return misclassification_analysis
    
    def explain_attack_type_prediction(self, sample_data, feature_names=None, include_lime=True):
        """
        Generate comprehensive explanation for attack type prediction
        
        Args:
            sample_data: Sample to explain
            feature_names: List of feature names
            include_lime: Whether to include LIME explanation
            
        Returns:
            Dictionary containing comprehensive explanation
        """
        # Get prediction with confidence
        prediction_result = self.predict_with_confidence(sample_data)
        
        # Get LIME explanation if requested and available
        lime_explanation = None
        if include_lime and LIME_AVAILABLE:
            try:
                lime_explanation = self.explain_attack_type_lime(sample_data, feature_names=feature_names)
            except Exception as e:
                print(f"LIME explanation failed: {str(e)}")
        
        # Create comprehensive explanation
        explanation = {
            'prediction_result': prediction_result,
            'lime_explanation': lime_explanation,
            'sample_data': sample_data.cpu().numpy() if hasattr(sample_data, 'cpu') else sample_data,
            'feature_names': feature_names or [f'feature_{i:02d}' for i in range(sample_data.shape[-1])],
            'attack_type_names': self.attack_type_names
        }
        
        return explanation
    
    def generate_attack_type_explanation_report(self, sample_data, feature_names=None):
        """
        Generate user-friendly explanation report for attack type prediction
        
        Args:
            sample_data: Sample to explain
            feature_names: List of feature names
            
        Returns:
            Formatted explanation report string
        """
        explanation = self.explain_attack_type_prediction(sample_data, feature_names)
        
        pred_result = explanation['prediction_result']
        predicted_class = pred_result['predicted_classes'][0]
        confidence = pred_result['confidence_scores'][0]
        probabilities = pred_result['probabilities'][0]
        
        report = f"""
=== ATTACK TYPE CLASSIFICATION EXPLANATION ===

Prediction Summary:
- Predicted Attack Type: {predicted_class}
- Confidence: {min(confidence * 100, 99.9):.1f}%
- Risk Level: {self._get_risk_level(predicted_class)}

---

## 🎯 **Top 3 Most Likely Attack Types:"""
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        for i, idx in enumerate(top_indices, 1):
            prob = probabilities[idx]
            class_name = self.attack_type_names[idx]
            indicator = "👉" if idx == np.argmax(probabilities) else "  "
            report += f"""
{i}. {indicator} {class_name}: {prob:.1%} confidence"""
        
        report += f"""

---

## 🔍 **Why This Classification?**

**Primary Factors:**
"""
        
        # Add LIME explanation if available
        if explanation['lime_explanation'] and 'lime_explanation' in explanation['lime_explanation']:
            lime_exp = explanation['lime_explanation']['lime_explanation']
            
            # Get explanation for the predicted class
            predicted_idx = self.attack_type_names.index(predicted_class)
            if hasattr(lime_exp, 'as_list'):
                lime_list = lime_exp.as_list(label=predicted_idx)
                for i, (feature, weight) in enumerate(lime_list[:5], 1):
                    report += f"""
{i}. {feature}: {'Supports' if weight > 0 else 'Opposes'} classification (weight: {weight:.3f})"""
        
        report += f"""

---

## 📊 **Attack Type Characteristics:**

**{predicted_class} Profile:**
{self._get_attack_type_description(predicted_class)}

---

## 🛡️ **Recommended Actions:**

{self._get_attack_type_recommendations(predicted_class)}

---

## 📈 **Prediction Confidence Analysis:**

- **Confidence Level**: {min(confidence * 100, 99.9):.1f}%
- **Uncertainty**: {'Low' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'High'}
- **Reliability**: {'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low'}

**Note:** If confidence is low, consider additional analysis before taking action.
"""
        
        return report
    
    def _get_risk_level(self, attack_type):
        """Get risk level for attack type"""
        risk_levels = {
            'Normal': '🟢 LOW',
            'DoS': '🟠 HIGH',
            'PortScan': '🟡 MEDIUM',
            'BruteForce': '🟠 HIGH',
            'WebAttack': '🟠 HIGH',
            'Infiltration': '🔴 CRITICAL'
        }
        return risk_levels.get(attack_type, '🟡 MEDIUM')
    
    def _get_attack_type_description(self, attack_type):
        """Get description of attack type"""
        descriptions = {
            'Normal': 'Normal network traffic with no malicious activity detected.',
            'DoS': 'Denial of Service attack attempting to overwhelm system resources.',
            'PortScan': 'Network reconnaissance activity scanning for open ports and services.',
            'BruteForce': 'Repeated login attempts trying to guess credentials.',
            'WebAttack': 'Web-based attacks including SQL injection, XSS, or other exploits.',
            'Infiltration': 'Advanced persistent threat with unauthorized system access.'
        }
        return descriptions.get(attack_type, 'Unknown attack type.')
    
    def _get_attack_type_recommendations(self, attack_type):
        """Get recommendations for attack type"""
        recommendations = {
            'Normal': 'Continue normal monitoring. No action required.',
            'DoS': '• Implement rate limiting\n• Check DDoS protection\n• Monitor resource usage\n• Consider traffic filtering',
            'PortScan': '• Block scanning IP addresses\n• Review firewall rules\n• Monitor for follow-up attacks\n• Update intrusion detection',
            'BruteForce': '• Lock affected accounts\n• Review authentication logs\n• Strengthen password policies\n• Enable multi-factor authentication',
            'WebAttack': '• Scan web applications for vulnerabilities\n• Review web server logs\n• Update web application firewall\n• Patch affected systems',
            'Infiltration': '• Immediate incident response\n• Isolate affected systems\n• Conduct forensic analysis\n• Review access logs and permissions'
        }
        return recommendations.get(attack_type, 'Review security logs and monitor for suspicious activity.')
