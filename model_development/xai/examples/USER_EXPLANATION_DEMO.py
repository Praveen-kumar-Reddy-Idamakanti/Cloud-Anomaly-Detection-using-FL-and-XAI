"""
User-Friendly Anomaly Explanation Demo

This script demonstrates how to answer the user's question:
"Why did this anomaly happen?"

It shows the complete workflow from detection to human-readable explanation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def create_demo_scenario():
    """
    Create a realistic demo scenario showing how to explain anomalies
    """
    
    print("🎯 DEMO: Answering 'Why did this anomaly happen?'")
    print("=" * 60)
    
    # Simulate detecting an anomaly
    print("\n📡 **STEP 1: ANOMALY DETECTION**")
    print("-" * 40)
    print("🚨 ANOMALY DETECTED at 2024-01-29 23:45:12")
    print("📍 Source: 192.168.1.105 → Destination: 10.0.0.50")
    print("🔍 Reconstruction Error: 0.034567 (Threshold: 0.015000)")
    print("⚠️  Risk Level: HIGH")
    
    # Now explain WHY it happened
    print("\n❓ **STEP 2: WHY DID THIS HAPPEN?**")
    print("-" * 40)
    
    # Simulate the explanation analysis
    explain_anomaly_causes()
    
    # Show actionable insights
    print("\n🛡️ **STEP 3: WHAT SHOULD WE DO?**")
    print("-" * 40)
    
    provide_actionable_recommendations()
    
    # Show visual explanation
    print("\n📊 **STEP 4: VISUAL EXPLANATION**")
    print("-" * 40)
    
    create_visual_explanation()

def explain_anomaly_causes():
    """
    Explain the specific causes of the anomaly
    """
    
    causes = [
        {
            "feature": "Port Usage Patterns",
            "severity": "🔴 CRITICAL",
            "what": "Unusual port scanning activity detected",
            "why": "Normal traffic uses standard ports (80, 443, 22), but this sample shows activity on 15 uncommon ports",
            "evidence": "Ports: 1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 0123, 1235, 2346, 3457, 4568, 5679"
        },
        {
            "feature": "Request Frequency", 
            "severity": "🟠 HIGH",
            "what": "Extremely high request rate detected",
            "why": "Normal rate: 10 requests/minute. This sample: 1,200 requests/minute (120x increase)",
            "evidence": "Burst pattern: 200 requests in 10 seconds, repeated 6 times"
        },
        {
            "feature": "Packet Size Distribution",
            "severity": "🟡 MEDIUM", 
            "what": "Unusual packet size patterns",
            "why": "Normal packets: 64-1500 bytes. This sample shows many 1-byte packets (possible covert channel)",
            "evidence": "1-byte packets: 45% of total traffic (normal: <1%)"
        },
        {
            "feature": "Connection Duration",
            "severity": "🟡 MEDIUM",
            "what": "Very short connection durations",
            "why": "Normal connections: 30 seconds average. This sample: 0.1 seconds average",
            "evidence": "300 connections, all lasting < 200ms"
        },
        {
            "feature": "Protocol Distribution",
            "severity": "🔵 LOW",
            "what": "Slightly unusual protocol mix",
            "why": "Increase in ICMP traffic (normally <1%, now 5%)",
            "evidence": "ICMP: 5%, TCP: 70%, UDP: 25% (normal: ICMP: 0.5%, TCP: 80%, UDP: 19.5%)"
        }
    ]
    
    for i, cause in enumerate(causes, 1):
        print(f"\n{i}. **{cause['feature']}** - {cause['severity']}")
        print(f"   📋 What happened: {cause['what']}")
        print(f"   🤔 Why it's anomalous: {cause['why']}")
        print(f"   🔍 Evidence: {cause['evidence']}")

def provide_actionable_recommendations():
    """
    Provide specific actions the user should take
    """
    
    recommendations = [
        {
            "priority": "🚨 IMMEDIATE",
            "action": "Block the source IP",
            "details": "Add 192.168.1.105 to firewall blocklist for 24 hours",
            "reason": "Clear port scanning activity"
        },
        {
            "priority": "🚨 IMMEDIATE", 
            "action": "Scan affected systems",
            "details": "Run security scan on 10.0.0.50 and connected systems",
            "reason": "Target of port scanning may be compromised"
        },
        {
            "priority": "⚠️ URGENT",
            "action": "Review authentication logs",
            "details": "Check for brute force attempts on all systems",
            "reason": "High request frequency suggests password attacks"
        },
        {
            "priority": "⚠️ URGENT",
            "action": "Monitor for lateral movement",
            "details": "Watch for similar patterns from other IPs",
            "reason": "Port scanning often precedes broader attacks"
        },
        {
            "priority": "📋 SHORT-TERM",
            "action": "Update detection rules",
            "details": "Add this pattern to anomaly detection signatures",
            "reason": "Prevent future similar attacks"
        }
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. **{rec['action']}** - {rec['priority']}")
        print(f"   💡 Details: {rec['details']}")
        print(f"   🎯 Reason: {rec['reason']}")

def create_visual_explanation():
    """
    Create a simple visual explanation
    """
    
    # Create a simple comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Normal vs Anomaly comparison
    features = ['Port Usage', 'Request Freq', 'Packet Size', 'Conn Duration', 'Protocol']
    normal_values = [0.1, 0.2, 0.15, 0.1, 0.05]
    anomaly_values = [0.9, 0.85, 0.6, 0.7, 0.3]
    
    x = np.arange(len(features))
    width = 0.35
    
    ax1.bar(x - width/2, normal_values, width, label='Normal', color='green', alpha=0.7)
    ax1.bar(x + width/2, anomaly_values, width, label='Anomaly', color='red', alpha=0.7)
    
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title('Normal vs Anomaly Feature Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Timeline of events
    times = ['23:44:00', '23:44:30', '23:45:00', '23:45:30', '23:46:00']
    events = [0, 50, 200, 150, 100]  # Requests per 30 seconds
    
    ax2.plot(times, events, 'ro-', linewidth=2, markersize=8)
    ax2.axhline(y=10, color='green', linestyle='--', label='Normal threshold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Requests per 30 seconds')
    ax2.set_title('Request Frequency Timeline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/demos/anomaly_explanation_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📈 Visual explanation saved as 'images/demos/anomaly_explanation_demo.png'")

def simulate_real_time_explanation():
    """
    Simulate how this would work in real-time
    """
    
    print("\n🔄 **REAL-TIME EXPLANATION WORKFLOW**")
    print("=" * 60)
    
    print("\n1️⃣ **User asks:** 'Why did this anomaly happen?'")
    print("\n2️⃣ **System responds instantly:**")
    
    response = """
🎯 **ANOMALY EXPLANATION**

**Summary:** This was detected as an anomaly because the system observed unusual port scanning activity combined with extremely high request frequency.

**Top 3 Reasons:**
1. **Port Scanning (CRITICAL)**: Activity detected on 15 uncommon ports instead of normal 3 standard ports
2. **Request Frequency (HIGH)**: 1,200 requests/minute vs normal 10 requests/minute  
3. **Packet Patterns (MEDIUM)**: 45% of packets are 1-byte (possible covert channel)

**Immediate Action Required:**
🚨 Block IP 192.168.1.105 and scan target system 10.0.0.50

**Confidence:** 94% that this is a malicious port scanning attack
"""
    
    print(response)
    
    print("\n3️⃣ **User can ask follow-up questions:**")
    print("   • 'Show me the port details'")
    print("   • 'What other systems are affected?'") 
    print("   • 'Has this happened before?'")
    print("   • 'What should I do right now?'")

def main():
    """
    Main demonstration function
    """
    
    # Create demo scenario
    create_demo_scenario()
    
    # Show real-time workflow
    simulate_real_time_explanation()
    
    print("\n" + "=" * 60)
    print("✅ **DEMO COMPLETE**")
    print("=" * 60)
    print("\n🎯 **KEY TAKEAWAYS:**")
    print("1. System detects anomalies automatically")
    print("2. Provides immediate, human-readable explanations")
    print("3. Identifies specific features that caused the anomaly")
    print("4. Gives actionable recommendations")
    print("5. Supports follow-up questions for deeper analysis")
    
    print("\n💡 **FOR USERS:**")
    print("• Ask 'Why did this anomaly happen?' anytime")
    print("• Get immediate, understandable explanations")
    print("• Receive specific actions to take")
    print("• See visual evidence of what happened")
    
    print("\n🔧 **FOR DEVELOPERS:**")
    print("• Use AutoencoderExplainer.explain_anomaly_sample()")
    print("• Convert technical results to user-friendly language")
    print("• Provide actionable security recommendations")
    print("• Create visual explanations for better understanding")

if __name__ == "__main__":
    main()
