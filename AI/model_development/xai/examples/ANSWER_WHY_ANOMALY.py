"""
ANSWER: "Why did this anomaly happen?"

This is the complete solution for answering users' questions about anomalies.
It provides the exact code and workflow to explain anomalies in human-readable terms.
"""

import numpy as np
import pandas as pd
import torch
from autoencoder_explainer import AutoencoderExplainer
from visualization.autoencoder_plots import AutoencoderPlotter

class AnomalyExplanationSystem:
    """
    Complete system for answering "Why did this anomaly happen?"
    """
    
    def __init__(self, trained_autoencoder, feature_names=None):
        """
        Initialize the explanation system
        
        Args:
            trained_autoencoder: Your trained autoencoder model
            feature_names: List of human-readable feature names
        """
        self.explainer = AutoencoderExplainer(trained_autoencoder)
        self.plotter = AutoencoderPlotter()
        self.feature_names = feature_names or [f'feature_{i:02d}' for i in range(78)]
        
        # Map features to security-relevant descriptions
        self.feature_descriptions = {
            'feature_01': 'Source IP address patterns',
            'feature_02': 'Destination IP address patterns',
            'feature_03': 'Port usage patterns',
            'feature_04': 'Protocol distribution',
            'feature_05': 'Packet size distribution',
            'feature_06': 'Traffic volume',
            'feature_07': 'Connection duration',
            'feature_08': 'Data transfer rate',
            'feature_09': 'Request frequency',
            'feature_10': 'Response time patterns',
            'feature_11': 'TCP flag combinations',
            'feature_12': 'TCP window size',
            'feature_13': 'Packet inter-arrival time',
            'feature_14': 'Byte distribution',
            'feature_15': 'Connection state',
            'feature_16': 'Geographic location patterns',
            'feature_17': 'Time of day patterns',
            'feature_18': 'Day of week patterns',
            'feature_19': 'Service port usage',
            'feature_20': 'Application layer patterns'
        }
    
    def answer_why_anomaly_happened(self, sample_data, threshold=None, include_visuals=True):
        """
        Main method to answer "Why did this anomaly happen?"
        
        Args:
            sample_data: The anomalous sample to explain
            threshold: Anomaly detection threshold
            include_visuals: Whether to generate visual explanations
            
        Returns:
            Complete explanation dictionary and user-friendly text
        """
        
        # Get technical explanation
        explanation = self.explainer.explain_anomaly_sample(sample_data, self.feature_names, threshold)
        
        # Convert to user-friendly explanation
        user_explanation = self._create_user_friendly_explanation(explanation)
        
        # Generate visual explanations if requested
        if include_visuals:
            self._generate_visual_explanations(explanation)
        
        return {
            'technical_explanation': explanation,
            'user_explanation': user_explanation,
            'is_anomaly': explanation['is_anomaly'],
            'confidence': self._calculate_confidence(explanation),
            'recommended_actions': self._get_recommended_actions(explanation)
        }
    
    def _create_user_friendly_explanation(self, explanation):
        """
        Convert technical explanation to user-friendly language
        """
        
        if not explanation['is_anomaly']:
            return """
🟢 **NORMAL TRAFFIC DETECTED**

This network traffic appears to be normal and within expected patterns.
The system did not detect any unusual behavior that would require attention.

**Key Points:**
• All features are within normal ranges
• Reconstruction error is below threshold
• No security action required
"""
        
        # Get top contributing features
        top_features = explanation['top_contributing_features'][:5]
        
        user_text = f"""
🚨 **ANOMALY DETECTED - WHY THIS HAPPENED**

**Risk Level:** {self._get_risk_level(explanation['reconstruction_error'])}
**Detection Confidence:** {self._calculate_confidence(explanation):.1f}%
**Reconstruction Error:** {explanation['reconstruction_error']:.6f}

---

## 🎯 **MAIN REASONS FOR ANOMALY:**

"""
        
        # Add each top feature with human-readable explanation
        for i, (feature, error) in enumerate(top_features, 1):
            feature_desc = self.feature_descriptions.get(feature, feature)
            severity = self._get_severity_level(error, explanation['per_feature_errors'])
            
            user_text += f"""
### {i}. {feature_desc} - {severity}
**What happened:** This feature showed unusual behavior that deviates from normal patterns
**Impact level:** {self._get_impact_description(error)}
**Security relevance:** {self._get_security_relevance(feature)}
"""
        
        # Add comparison section
        user_text += f"""

---

## 📊 **EXPECTED vs ACTUAL BEHAVIOR:**

**What the system expected:** Normal network traffic patterns based on historical data
**What actually happened:** Unusual traffic with significant deviations
**Difference magnitude:** {explanation['reconstruction_error']:.6f} (Threshold: {explanation.get('threshold', 'N/A')})

**Top 3 Anomalous Features:**
"""
        
        for i, (feature, error) in enumerate(top_features[:3], 1):
            feature_idx = self.feature_names.index(feature) if feature in self.feature_names else i-1
            if feature_idx < len(explanation['original_sample']):
                original = explanation['original_sample'][feature_idx]
                reconstructed = explanation['reconstructed_sample'][feature_idx]
                user_text += f"""
{i}. {feature}: Expected {reconstructed:.4f}, Got {original:.4f} (Error: {error:.6f})
"""
        
        # Add actionable recommendations
        user_text += f"""

---

## 🛡️ **IMMEDIATE ACTIONS RECOMMENDED:**

{self._get_actionable_recommendations(explanation)}

---

## 📞 **NEXT STEPS:**

• Review the detailed technical analysis below
• Check system logs for related events
• Monitor for similar patterns
• Update security rules if this is a confirmed threat

**Need more details?** Contact security team with anomaly ID and timestamp.
"""
        
        return user_text
    
    def _get_risk_level(self, error):
        """Convert error to risk level"""
        if error > 0.05:
            return "🔴 CRITICAL"
        elif error > 0.03:
            return "🟠 HIGH"
        elif error > 0.02:
            return "🟡 MEDIUM"
        else:
            return "🔵 LOW"
    
    def _calculate_confidence(self, explanation):
        """Calculate confidence in anomaly detection"""
        if explanation.get('threshold'):
            confidence = min(95, (explanation['reconstruction_error'] / explanation['threshold']) * 100)
            return max(50, confidence)  # Minimum 50% confidence
        return 75.0  # Default confidence
    
    def _get_severity_level(self, error, all_errors):
        """Get severity level for a feature error"""
        percentile = (error < all_errors).mean() * 100
        if percentile >= 95:
            return "🔴 CRITICAL"
        elif percentile >= 90:
            return "🟠 HIGH"
        elif percentile >= 80:
            return "🟡 MEDIUM"
        else:
            return "🔵 LOW"
    
    def _get_impact_description(self, error):
        """Get human-readable impact description"""
        if error > 0.01:
            return "Severe deviation from normal patterns"
        elif error > 0.005:
            return "Significant deviation from normal patterns"
        elif error > 0.001:
            return "Moderate deviation from normal patterns"
        else:
            return "Minor deviation from normal patterns"
    
    def _get_security_relevance(self, feature):
        """Get security relevance of a feature"""
        security_relevance = {
            'feature_01': 'Unusual source IP may indicate spoofing or unauthorized access',
            'feature_02': 'Abnormal destination patterns could suggest data exfiltration',
            'feature_03': 'Port usage anomalies often indicate scanning or exploitation',
            'feature_04': 'Protocol changes may signal tunneling or covert channels',
            'feature_05': 'Packet size anomalies can indicate fragmentation attacks',
            'feature_06': 'Traffic volume changes may suggest DDoS or breaches',
            'feature_07': 'Unusual connection durations could indicate persistence',
            'feature_08': 'Transfer rate anomalies may signal data exfiltration',
            'feature_09': 'Request frequency changes often indicate brute force',
            'feature_10': 'Response time anomalies could suggest service manipulation'
        }
        return security_relevance.get(feature, 'This feature showed anomalous behavior requiring investigation')
    
    def _get_actionable_recommendations(self, explanation):
        """Get specific actionable recommendations"""
        top_features = explanation['top_contributing_features'][:3]
        
        recommendations = []
        
        for feature, error in top_features:
            if 'port' in self.feature_descriptions.get(feature, '').lower():
                recommendations.append("🔍 **PORT SECURITY**: Review firewall rules and block suspicious ports")
            elif 'frequency' in self.feature_descriptions.get(feature, '').lower():
                recommendations.append("⚡ **RATE LIMITING**: Implement rate limiting for affected services")
            elif 'ip' in self.feature_descriptions.get(feature, '').lower():
                recommendations.append("🚫 **ACCESS CONTROL**: Review and restrict suspicious IP addresses")
            elif 'size' in self.feature_descriptions.get(feature, '').lower():
                recommendations.append("📦 **PACKET ANALYSIS**: Investigate unusual packet patterns for covert channels")
            else:
                recommendations.append("🔍 **GENERAL**: Monitor systems for related suspicious activity")
        
        return "\n".join(recommendations) if recommendations else "🔍 **MONITOR**: Increase monitoring and review system logs"
    
    def _generate_visual_explanations(self, explanation):
        """Generate visual explanations"""
        
        # Feature comparison plot
        self.plotter.plot_feature_comparison(
            explanation['original_sample'],
            explanation['reconstructed_sample'],
            self.feature_names,
            save_path='anomaly_explanation_comparison.png'
        )
        
        # Comprehensive explanation summary
        self.plotter.plot_anomaly_explanation_summary(
            explanation,
            save_path='anomaly_explanation_summary.png'
        )
    
    def _get_recommended_actions(self, explanation):
        """Get structured recommended actions"""
        actions = []
        
        if explanation['is_anomaly']:
            actions.extend([
                {
                    'priority': 'IMMEDIATE',
                    'action': 'Investigate the source of anomalous traffic',
                    'details': 'Review logs and identify the origin of unusual patterns'
                },
                {
                    'priority': 'IMMEDIATE',
                    'action': 'Check for security breaches',
                    'details': 'Verify if systems have been compromised'
                },
                {
                    'priority': 'URGENT',
                    'action': 'Monitor for related activity',
                    'details': 'Watch for similar patterns from other sources'
                }
            ])
        else:
            actions.append({
                'priority': 'INFO',
                'action': 'Continue normal monitoring',
                'details': 'No immediate action required'
            })
        
        return actions

# USAGE EXAMPLE
def example_usage():
    """
    Example of how to use the system to answer "Why did this anomaly happen?"
    """
    
    print("🎯 EXAMPLE: Answering 'Why did this anomaly happen?'")
    print("=" * 60)
    
    # This would be your actual trained autoencoder and anomalous sample
    # For demonstration, here's the workflow:
    
    print("\n1️⃣ **Initialize the explanation system:**")
    print("""
from model_development.xai import AnomalyExplanationSystem

# Initialize with your trained autoencoder
explanation_system = AnomalyExplanationSystem(
    trained_autoencoder=your_autoencoder_model,
    feature_names=your_feature_names
)
""")
    
    print("\n2️⃣ **When an anomaly is detected, ask for explanation:**")
    print("""
# Get the anomalous sample (from your detection system)
anomalous_sample = get_anomalous_sample()

# Ask the system to explain why it happened
result = explanation_system.answer_why_anomaly_happened(
    sample_data=anomalous_sample,
    threshold=your_detection_threshold,
    include_visuals=True
)
""")
    
    print("\n3️⃣ **Get the user-friendly explanation:**")
    print("""
# Print the explanation for the user
print(result['user_explanation'])

# Get recommended actions
for action in result['recommended_actions']:
    print(f"{action['priority']}: {action['action']}")
    print(f"  Details: {action['details']}")
""")
    
    print("\n4️⃣ **Example output:**")
    print("""
🚨 **ANOMALY DETECTED - WHY THIS HAPPENED**

**Risk Level:** 🟠 HIGH
**Detection Confidence:** 87.3%
**Reconstruction Error:** 0.034567

---

## 🎯 **MAIN REASONS FOR ANOMALY:**

### 1. Port usage patterns - 🟠 HIGH
**What happened:** This feature showed unusual behavior that deviates from normal patterns
**Impact level:** Significant deviation from normal patterns
**Security relevance:** Port usage anomalies often indicate scanning or exploitation

### 2. Request frequency - 🔴 CRITICAL
**What happened:** This feature showed unusual behavior that deviates from normal patterns
**Impact level:** Severe deviation from normal patterns
**Security relevance:** Request frequency changes often indicate brute force

---

## 🛡️ **IMMEDIATE ACTIONS RECOMMENDED:**

🔍 **PORT SECURITY**: Review firewall rules and block suspicious ports
⚡ **RATE LIMITING**: Implement rate limiting for affected services
🔍 **GENERAL**: Monitor systems for related suspicious activity
""")

if __name__ == "__main__":
    example_usage()
