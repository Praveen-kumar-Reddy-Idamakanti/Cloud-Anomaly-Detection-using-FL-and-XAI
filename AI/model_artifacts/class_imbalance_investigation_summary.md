# Class Imbalance Investigation Summary

## Executive Summary

**Investigation Date**: February 15, 2026  
**Issue**: Low F1-macro scores (0.54 Oracle, 0.47 End-to-End) despite high accuracy (>92%)  
**Root Cause**: **Severe class imbalance** in attack category classification  
**Impact**: Complete failure on rare attack types (Botnet, Infiltration)  

## Critical Findings

### 🚨 Extreme Class Imbalance Detected

| Attack Type | Samples | Percentage | F1-Score | Imbalance Ratio |
|-------------|---------|------------|----------|-----------------|
| **Infiltration** | 25 | 0.006% | 0.000 | **10,260:1** |
| **Botnet** | 275 | 0.061% | 0.000 | **933:1** |
| **DoS** | 125,215 | 27.95% | 0.931 | 2.05:1 |
| **PortScan** | 65,979 | 14.73% | 0.837 | 3.89:1 |
| **Other** | 256,503 | 57.26% | 0.942 | 1:1 (Baseline) |

### 📊 Performance Analysis

#### Complete Failure on Rare Classes
- **Botnet**: 0.0 precision, 0.0 recall, 0.0 F1-score
- **Infiltration**: 0.0 precision, 0.0 recall, 0.0 F1-score
- **Impact**: Security system cannot detect critical attack types

#### Misclassification Patterns
- **Botnet → Other**: 80% misclassified as "Other" (221/275)
- **Infiltration → Other**: 44% misclassified as "Other" (11/25)
- **Model bias**: Overwhelmingly predicts majority class

#### Metric Deception
- **High Accuracy (92.5%)**: Misleading due to dominant "Other" class
- **Weighted F1 (0.923)**: Inflated by majority class performance
- **Macro F1 (0.542)**: True measure of poor overall balance

## Root Cause Analysis

### 1. Data Distribution Issues
- **Infiltration**: Only 25 samples in entire dataset
- **Botnet**: Only 275 samples 
- **Training inadequacy**: Insufficient samples for model learning

### 2. Model Training Problems
- **No class balancing**: Standard cross-entropy loss
- **Majority class dominance**: "Other" class overwhelms gradient updates
- **Loss function bias**: Optimizes for overall accuracy, not rare class detection

### 3. Evaluation Misinterpretation
- **Accuracy focus**: Traditional metrics misleading for imbalanced data
- **Business impact**: Missing rare attacks could be catastrophic
- **False confidence**: High accuracy masks critical failures

## Immediate Action Plan

### 🚨 Priority 1: Critical Fixes (Week 1)

#### 1. Implement Class-Weighted Loss
```python
# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
```

#### 2. Aggressive Oversampling
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **Target**: Balance rare classes to at least 1,000 samples each
- **Implementation**: Use imbalanced-learn library

#### 3. Focal Loss Implementation
```python
# Focus on hard/rare examples
criterion = FocalLoss(gamma=2.0, alpha=class_weights)
```

### 🎯 Priority 2: Performance Improvements (Week 2-3)

#### 1. Hierarchical Classification
- **Level 1**: Common vs Rare attack detection
- **Level 2**: Specific attack type classification
- **Benefit**: Isolates rare class learning

#### 2. Ensemble Methods
- **Specialized models**: Separate models for rare vs common classes
- **Voting ensemble**: Combine predictions with confidence weighting
- **Threshold tuning**: Per-class optimal thresholds

#### 3. Data Augmentation
- **GAN-based synthesis**: Generate realistic rare attack samples
- **Feature perturbation**: Create variations of existing rare samples
- **Transfer learning**: Leverage external datasets

### 📈 Priority 3: Advanced Solutions (Week 4-6)

#### 1. Few-Shot Learning
- **Meta-learning**: Learn to learn from few examples
- **Prototypical networks**: Learn class prototypes
- **Siamese networks**: Learn similarity metrics

#### 2. Cost-Sensitive Learning
- **Misclassification costs**: Higher penalty for rare class errors
- **Business impact weighting**: Security-critical attacks weighted higher
- **Custom loss functions**: Domain-specific optimization

#### 3. Anomaly-within-Anomaly Detection
- **Two-stage approach**: 
  1. Detect any anomaly
  2. Classify if anomaly is "rare attack type"
- **Autoencoder ensemble**: Separate models for each attack type

## Success Metrics & Monitoring

### Target Improvements
| Metric | Current | Target (Week 2) | Target (Week 4) |
|--------|---------|-----------------|-----------------|
| Botnet F1 | 0.000 | 0.300 | 0.600 |
| Infiltration F1 | 0.000 | 0.200 | 0.500 |
| Macro F1 | 0.542 | 0.700 | 0.800 |
| Balanced Accuracy | 0.617 | 0.750 | 0.850 |

### Monitoring Dashboard
- **Real-time per-class metrics**
- **Class distribution tracking**
- **Confusion matrix visualization**
- **Rare class detection alerts**
- **Model drift monitoring**

## Implementation Checklist

### Week 1 Tasks
- [ ] Implement class-weighted loss function
- [ ] Add SMOTE oversampling pipeline
- [ ] Create per-class metric tracking
- [ ] Set up monitoring dashboard
- [ ] Test baseline improvements

### Week 2 Tasks
- [ ] Implement focal loss
- [ ] Create hierarchical classification
- [ ] Tune per-class thresholds
- [ ] Validate improvements on holdout set
- [ ] Document performance gains

### Week 3-4 Tasks
- [ ] Implement ensemble methods
- [ ] Add data augmentation pipeline
- [ ] Optimize hyperparameters
- [ ] Conduct stress testing
- [ ] Prepare production deployment

## Risk Assessment

### High Risks
- **Model instability**: Oversampling may cause overfitting
- **Computational cost**: Advanced methods increase training time
- **Complexity**: Ensemble methods harder to maintain

### Mitigation Strategies
- **Cross-validation**: Ensure robustness across data splits
- **A/B testing**: Compare with baseline before deployment
- **Fallback mechanisms**: Keep simple model as backup
- **Continuous monitoring**: Track performance degradation

## Conclusion

The class imbalance problem is **critical and requires immediate attention**. The current system's inability to detect Botnet and Infiltration attacks represents a significant security vulnerability.

The recommended phased approach balances:
- **Urgency**: Immediate fixes for critical failures
- **Effectiveness**: Proven techniques for imbalance
- **Practicality**: Implementation within existing framework
- **Sustainability**: Long-term monitoring and improvement

**Next Step**: Begin implementation of class-weighted loss and SMOTE oversampling within 48 hours.

---

**Files Generated**:
- `class_imbalance_analysis.md` - Detailed technical analysis
- `class_imbalance_analysis.png` - Visual analysis dashboard
- `class_imbalance_summary.csv` - Quantitative summary
- `analyze_class_imbalance.py` - Analysis automation script

**Contact**: Data Science Team for implementation coordination
