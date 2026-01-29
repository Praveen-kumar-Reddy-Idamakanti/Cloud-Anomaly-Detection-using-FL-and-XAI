# Attack Type Classifier Development Plan

## 🎯 Objective
Build Stage 2 of two-stage classification system to identify specific attack types after anomaly detection.

## 📊 Available Infrastructure
✅ **Attack Type Mappings**: 5 specific attack types identified
✅ **Labeled Data**: 447,997 anomaly samples with attack type labels
✅ **Lookup Functions**: Ready-to-use attack type conversion functions
✅ **Two-Stage Framework**: Already designed in phase4

## 🔧 Development Steps

### Step 1: Data Preparation
- Extract only anomaly samples from processed data
- Use Attack_Type_Numeric column as labels (0-4)
- Split into train/validation/test sets
- Handle class imbalance (some attack types are rare)

### Step 2: Model Architecture
- Multi-class neural network classifier
- Input: 78 features (same as autoencoder)
- Output: 5 attack type classes
- Use CrossEntropyLoss for multi-class

### Step 3: Training Pipeline
- Train only on anomaly samples
- Evaluate classification accuracy per attack type
- Generate confusion matrix for attack types
- Save trained classifier

### Step 4: Integration
- Combine with existing autoencoder
- Create two-stage prediction pipeline
- Autoencoder → Anomaly Detection → Attack Type Classification

## 🎯 Expected Performance
- Goal: 70%+ attack type classification accuracy
- Challenge: Some attack types are very rare (Botnet, Infiltration)
- Focus: Good performance on common attacks (DoS, PortScan)

## 📁 Deliverables
1. `attack_type_classifier.py` - Multi-class classifier
2. `attack_type_training.py` - Training pipeline
3. `two_stage_system.py` - Integrated system
4. Performance metrics and visualizations
