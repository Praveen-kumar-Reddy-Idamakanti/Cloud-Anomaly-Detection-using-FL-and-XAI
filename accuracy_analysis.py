"""
Why accuracy stayed the same while other metrics improved
"""

# SIMPLIFIED EXAMPLE:

# BEFORE (17 epochs) - Conservative Model
predictions_before = {
    'True Negatives': 150000,  # Correctly identified normal
    'False Positives': 50000,  # Normal incorrectly flagged
    'True Positives': 250000,  # Correctly identified anomalies  
    'False Negatives': 200000   # Missed anomalies
}

# AFTER (50 epochs) - More Sensitive Model
predictions_after = {
    'True Negatives': 140000,  # Fewer (some normal misclassified)
    'False Positives': 60000,  # More (more false alarms)
    'True Positives': 270000,  # More (found more anomalies)
    'False Negatives': 180000   # Fewer (missed less)
}

# Calculate accuracy
total_samples = 234935 + 447997  # 682932

acc_before = (150000 + 250000) / total_samples  # 0.585
acc_after = (140000 + 270000) / total_samples   # 0.602

print("🎯 KEY INSIGHT:")
print("Model traded some precision for better recall")
print("Accuracy stayed similar but model is more useful!")
print("Better to catch more threats (higher recall)!")

# REAL METRICS:
print("\n📊 YOUR ACTUAL RESULTS:")
print("Accuracy: 63.96% → 63.95% (stable)")
print("Recall: 58.48% → 60.07% (better anomaly detection)")
print("Precision: 81.33% → 79.99% (more false alarms but acceptable)")
print("Bottom line: Better security with same overall accuracy!")
