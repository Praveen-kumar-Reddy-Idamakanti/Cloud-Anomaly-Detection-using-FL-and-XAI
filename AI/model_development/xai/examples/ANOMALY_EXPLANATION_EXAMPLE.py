"""
Anomaly Explanation Example for Users

This script demonstrates how to explain "WHY" an anomaly happened
in a way that users can understand and act upon.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def explain_anomaly_to_user(explainer, sample_data, feature_names=None, threshold=None):
    """
    Generate user-friendly explanation for why an anomaly occurred
    
    Args:
        explainer: Trained AutoencoderExplainer
        sample_data: The anomalous sample to explain
        feature_names: Human-readable feature names
        threshold: Anomaly detection threshold
    
    Returns:
        User-friendly explanation string
    """
    
    # Get technical explanation
    explanation = explainer.explain_anomaly_sample(sample_data, feature_names, threshold)
    
    # Convert to user-friendly explanation
    user_explanation = generate_user_friendly_explanation(explanation, feature_names)
    
    return user_explanation

def generate_user_friendly_explanation(explanation, feature_names=None):
    """
    Convert technical explanation to user-friendly language
    """
    
    is_anomaly = explanation['is_anomaly']
    reconstruction_error = explanation['reconstruction_error']
    top_features = explanation['top_contributing_features'][:5]
    
    if not is_anomaly:
        return """
🟢 **NORMAL TRAFFIC DETECTED**

This network traffic appears to be normal and within expected patterns.
The system did not detect any unusual behavior that would require attention.
"""
    
    # Create human-readable explanation
    user_explanation = f"""
🚨 **ANOMALY DETECTED - WHY THIS HAPPENED**

**Overall Risk Level:** HIGH
**Detection Confidence:** {min(95, (reconstruction_error / explanation['threshold']) * 100):.1f}%

---

## 🎯 **Main Reasons for Anomaly Detection:**

"""
    
    # Add top contributing features with human-readable descriptions
    for i, (feature, error) in enumerate(top_features, 1):
        feature_desc = get_feature_description(feature, feature_names)
        impact_level = get_impact_level(error, explanation['per_feature_errors'])
        
        user_explanation += f"""
### {i}. {feature_desc}
**Impact:** {impact_level}
**What happened:** This feature showed unusual behavior that deviates significantly from normal patterns.
**Why it matters:** {get_feature_importance_description(feature)}
"""
    
    # Add reconstruction comparison
    user_explanation += f"""
---

## 📊 **What the System Expected vs What Actually Happened:**

The system tried to reconstruct this traffic pattern based on learned normal behavior:
- **Expected Pattern:** Normal network traffic with typical feature values
- **Actual Pattern:** Unusual traffic with significant deviations
- **Difference Score:** {reconstruction_error:.6f} (Threshold: {explanation['threshold']:.6f})

---

## 🔍 **Technical Details (for security team):**

**Top 5 Anomalous Features:**
"""
    
    for i, (feature, error) in enumerate(top_features, 1):
        original_val = explanation['original_sample'][feature_names.index(feature)] if feature_names and feature in feature_names else "N/A"
        reconstructed_val = explanation['reconstructed_sample'][feature_names.index(feature)] if feature_names and feature in feature_names else "N/A"
        
        user_explanation += f"""
• {feature}: Expected {reconstructed_val:.4f}, Got {original_val:.4f} (Error: {error:.6f})
"""
    
    # Add actionable recommendations
    user_explanation += f"""

---

## 🛡️ **Recommended Actions:**

1. **Immediate Investigation**
   - Review the top anomalous features listed above
   - Check recent network logs for related events
   - Verify if this corresponds to known attack patterns

2. **Security Monitoring**
   - Increase monitoring on affected systems
   - Look for similar patterns in recent traffic
   - Check for lateral movement attempts

3. **Prevention Measures**
   - Update firewall rules if this is a known attack vector
   - Enhance intrusion detection for similar patterns
   - Review access logs for unauthorized attempts

---

## 📞 **Next Steps:**

If this is a false positive:
- Document the pattern for future reference
- Consider adjusting detection thresholds
- Update the normal behavior model

If this is a true anomaly:
- Follow your incident response procedures
- Document the attack pattern
- Share findings with security team

**Need more details?** Contact the security team with the anomaly ID and timestamp.
"""
    
    return user_explanation

def get_feature_description(feature, feature_names=None):
    """
    Convert feature names to human-readable descriptions
    """
    # Map common network traffic features to descriptions
    feature_descriptions = {
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
        'feature_11': 'Flag combinations',
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
    
    return feature_descriptions.get(feature, f'Network traffic feature {feature}')

def get_impact_level(error, all_errors):
    """
    Convert error value to impact level
    """
    error_percentile = (error < all_errors).mean() * 100
    
    if error_percentile >= 95:
        return "🔴 CRITICAL"
    elif error_percentile >= 90:
        return "🟠 HIGH"
    elif error_percentile >= 80:
        return "🟡 MEDIUM"
    else:
        return "🔵 LOW"

def get_feature_importance_description(feature):
    """
    Explain why a feature is important for security
    """
    importance_descriptions = {
        'feature_01': 'Unusual source IP patterns may indicate spoofing or unauthorized access',
        'feature_02': 'Abnormal destination patterns could suggest data exfiltration or scanning',
        'feature_03': 'Port usage anomalies often indicate port scanning or service exploitation',
        'feature_04': 'Protocol distribution changes may signal tunneling or covert channels',
        'feature_05': 'Packet size anomalies can indicate fragmentation attacks or data hiding',
        'feature_06': 'Traffic volume changes may suggest DDoS attacks or data breaches',
        'feature_07': 'Unusual connection durations could indicate persistent threats or beaconing',
        'feature_08': 'Transfer rate anomalies may signal data exfiltration or command & control',
        'feature_09': 'Request frequency changes often indicate brute force or scanning attacks',
        'feature_10': 'Response time anomalies could suggest service degradation or manipulation'
    }
    
    return importance_descriptions.get(feature, 'This feature showed behavior that deviates from normal network patterns')

def create_interactive_explanation_dashboard(explainer, sample_data, feature_names=None):
    """
    Create an interactive dashboard for explaining anomalies
    """
    from visualization.autoencoder_plots import AutoencoderPlotter
    
    # Get explanation
    explanation = explainer.explain_anomaly_sample(sample_data, feature_names)
    
    # Create visualizations
    plotter = AutoencoderPlotter()
    
    print("🎯 Generating Anomaly Explanation Dashboard...")
    print("=" * 60)
    
    # Print user-friendly explanation
    user_explanation = generate_user_friendly_explanation(explanation, feature_names)
    print(user_explanation)
    
    # Create visual explanations
    print("\n📊 Generating Visual Explanations...")
    
    # Feature comparison plot
    plotter.plot_feature_comparison(
        explanation['original_sample'],
        explanation['reconstructed_sample'],
        feature_names,
        save_path='anomaly_feature_comparison.png'
    )
    
    # Comprehensive explanation summary
    plotter.plot_anomaly_explanation_summary(
        explanation,
        save_path='anomaly_explanation_summary.png'
    )
    
    print("\n✅ Explanation complete!")
    print("📁 Generated files:")
    print("   - anomaly_feature_comparison.png")
    print("   - anomaly_explanation_summary.png")

# Example usage
def example_usage():
    """
    Example of how to use the anomaly explanation system
    """
    print("🎯 Anomaly Explanation Example")
    print("=" * 50)
    
    # This would be your actual trained autoencoder and data
    # For demonstration, we'll show the structure:
    
    example_explanation = """
🚨 **ANOMALY DETECTED - WHY THIS HAPPENED**

**Overall Risk Level:** HIGH
**Detection Confidence:** 87.3%

---

## 🎯 **Main Reasons for Anomaly Detection:**

### 1. Port usage patterns
**Impact:** 🔴 CRITICAL
**What happened:** This feature showed unusual behavior that deviates significantly from normal patterns.
**Why it matters:** Port usage anomalies often indicate port scanning or service exploitation

### 2. Traffic volume
**Impact:** 🟠 HIGH  
**What happened:** This feature showed unusual behavior that deviates significantly from normal patterns.
**Why it matters:** Traffic volume changes may suggest DDoS attacks or data breaches

### 3. Request frequency
**Impact:** 🟡 MEDIUM
**What happened:** This feature showed unusual behavior that deviates significantly from normal patterns.
**Why it matters:** Request frequency changes often indicate brute force or scanning attacks

---

## 📊 **What the System Expected vs What Actually Happened:**

The system tried to reconstruct this traffic pattern based on learned normal behavior:
- **Expected Pattern:** Normal network traffic with typical feature values
- **Actual Pattern:** Unusual traffic with significant deviations
- **Difference Score:** 0.023456 (Threshold: 0.015000)

---

## 🛡️ **Recommended Actions:**

1. **Immediate Investigation**
   - Review the top anomalous features listed above
   - Check recent network logs for related events
   - Verify if this corresponds to known attack patterns

2. **Security Monitoring**
   - Increase monitoring on affected systems
   - Look for similar patterns in recent traffic
   - Check for lateral movement attempts

3. **Prevention Measures**
   - Update firewall rules if this is a known attack vector
   - Enhance intrusion detection for similar patterns
   - Review access logs for unauthorized attempts
"""
    
    print(example_explanation)

if __name__ == "__main__":
    example_usage()
