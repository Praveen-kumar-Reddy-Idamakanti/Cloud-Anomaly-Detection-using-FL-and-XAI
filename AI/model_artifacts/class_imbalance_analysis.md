# Class Imbalance Analysis for Two-Stage Anomaly Detection System

## Executive Summary
The low F1-macro scores (0.54 for Oracle, 0.47 for End-to-End) despite high accuracy (>92%) indicate **severe class imbalance** in the attack category classification stage.

## Class Distribution Analysis

### Attack Category Support (Oracle Evaluation)
```
Botnet:      275 samples   (0.06%)
DoS:        125,215 samples (27.95%)
Infiltration: 25 samples   (0.01%)
Other:      256,503 samples (57.27%)
PortScan:   65,979 samples  (14.73%)
```

**Total Anomaly Samples: 447,997**

## Key Findings

### 1. Extreme Class Imbalance
- **Other class dominates**: 57.27% of all anomalies
- **Rare classes**: Botnet (0.06%), Infiltration (0.01%)
- **Class ratio**: Other:Infiltration = 10,260:1
- **DoS and PortScan**: Moderate representation (27.95%, 14.73%)

### 2. Model Performance by Class

#### Oracle Performance (True Anomalies)
```
Botnet:      Precision: 0.000, Recall: 0.000, F1: 0.000
DoS:         Precision: 0.977, Recall: 0.889, F1: 0.931
Infiltration:Precision: 0.000, Recall: 0.000, F1: 0.000
Other:       Precision: 0.897, Recall: 0.992, F1: 0.942
PortScan:    Precision: 0.967, Recall: 0.737, F1: 0.837
```

#### Critical Issues Identified
1. **Complete failure on rare classes**: Botnet and Infiltration get 0.0 F1 scores
2. **Model bias toward majority class**: Other class gets 99.2% recall
3. **High accuracy misleading**: 92.5% accuracy driven by dominant "Other" class

### 3. Confusion Matrix Analysis

#### Oracle Confusion Matrix
```
Predicted →  Botnet   DoS   Infiltration  Other  PortScan
Actual
Botnet         0      29        0        221      25
DoS            0   111356       0      13082     777
Infiltration   0        1        0         11      13
Other          0     1122       0     254526     855
PortScan       0     1482       0      15852   48645
```

#### Misclassification Patterns
- **Botnet samples**: 80% misclassified as "Other" (221/275)
- **Infiltration samples**: 44% misclassified as "Other" (11/25)
- **DoS samples**: 10.6% misclassified as "Other" (13,082/125,215)
- **PortScan samples**: 24% misclassified as "Other" (15,852/65,979)

## Root Causes

### 1. Training Data Imbalance
- **Insufficient rare class samples**: 25 Infiltration samples inadequate for training
- **Model overfitting to majority class**: "Other" class dominates loss function
- **No class balancing techniques**: No weighted loss or oversampling implemented

### 2. Evaluation Metrics Mismatch
- **Accuracy is misleading**: 92.5% accuracy doesn't reflect poor rare class performance
- **F1-macro reveals true performance**: 0.54 F1-macro shows poor overall balance
- **Weighted vs Macro discrepancy**: 
  - Weighted F1: 0.923 (inflated by majority class)
  - Macro F1: 0.542 (true measure of balance)

## Recommendations

### 1. Immediate Actions (High Priority)

#### Data-Level Solutions
- **Oversampling**: Use SMOTE or random oversampling for rare classes
- **Undersampling**: Reduce "Other" class samples to create balance
- **Data augmentation**: Generate synthetic samples for Botnet/Infiltration

#### Model-Level Solutions
- **Class-weighted loss**: Implement weighted cross-entropy loss
- **Focal loss**: Use focal loss to focus on hard/rare examples
- **Ensemble methods**: Train separate models for rare vs common classes

### 2. Medium-Term Improvements

#### Architecture Changes
- **Hierarchical classification**: First detect common vs rare, then classify
- **One-vs-rest classifiers**: Separate classifiers for each attack type
- **Threshold tuning**: Per-class threshold optimization

#### Evaluation Improvements
- **Per-class metrics**: Track individual class performance
- **Balanced accuracy**: Use balanced accuracy instead of overall accuracy
- **Cost-sensitive evaluation**: Weight errors by attack severity

### 3. Long-Term Solutions

#### Data Collection
- **Targeted data collection**: Gather more Botnet/Infiltration samples
- **Synthetic data generation**: Use GANs or VAEs for rare class synthesis
- **Transfer learning**: Leverage external datasets for rare classes

#### Advanced Techniques
- **Few-shot learning**: Implement meta-learning for rare classes
- **Anomaly detection within anomalies**: Treat rare classes as anomalies
- **Semi-supervised learning**: Use unlabeled data for rare class patterns

## Implementation Priority

### Phase 1 (Week 1-2)
1. Implement class-weighted loss function
2. Add SMOTE oversampling for training
3. Track per-class metrics during training

### Phase 2 (Week 3-4)
1. Experiment with focal loss
2. Implement hierarchical classification
3. Tune per-class decision thresholds

### Phase 3 (Week 5-6)
1. Collect/generate more rare class data
2. Implement ensemble methods
3. Deploy with monitoring for rare class performance

## Success Metrics

### Target Improvements
- **Botnet F1**: 0.0 → 0.3 (minimum viable)
- **Infiltration F1**: 0.0 → 0.2 (minimum viable)
- **Macro F1**: 0.54 → 0.7 (significant improvement)
- **Balanced accuracy**: >0.8 across all classes

### Monitoring Dashboard
- Real-time per-class precision/recall
- Class distribution tracking
- Confusion matrix visualization
- Rare class detection rate alerts

## Conclusion

The current system achieves high accuracy by predominantly classifying samples as "Other", completely failing on rare attack types. This is a critical issue for security applications where detecting rare but severe attacks (like Botnet and Infiltration) is essential.

The class imbalance problem requires immediate attention through both data-level and algorithm-level interventions. The recommended phased approach will systematically address the imbalance while maintaining system performance on majority classes.
