# Data Preprocessing Guide for Anomaly Detection

**Project Goal:** Detect anomalies in network traffic data using federated learning  
**Dataset:** CICIDS2017 Network Intrusion Detection Dataset  
**Target:** Clean, processed data ready for autoencoder-based anomaly detection

---

## 📊 Current Data Overview

### Raw Dataset Structure
- **Source:** CICIDS2017 MachineLearningCVE dataset
- **Files:** 8 CSV files containing network traffic data
- **Size:** ~600MB of raw network traffic logs
- **Features:** 78+ network flow features per record
- **Labels:** BENIGN vs various attack types (DDoS, PortScan, WebAttacks, etc.)

### Sample Features
```
Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,
Total Length of Fwd Packets, Total Length of Bwd Packets, Fwd Packet Length Max,
Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std, Flow IAT Max,
Min Packet Length, Max Packet Length, Packet Length Mean, Packet Length Std,
FIN Flag Count, SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count,
Down/Up Ratio, Average Packet Size, Active Mean, Active Std, Active Max,
Idle Mean, Idle Std, Idle Max, Label
```

---

## 🎯 Preprocessing Objectives

1. **Data Cleaning**: Remove inconsistencies, missing values, duplicates
2. **Feature Engineering**: Select/transform relevant features for anomaly detection
3. **Normalization**: Scale features for autoencoder training
4. **Label Encoding**: Convert attack labels to binary anomaly classification
5. **Data Splitting**: Prepare data for federated learning clients
6. **Quality Assurance**: Validate processed data integrity

---

## 📋 Step-by-Step Preprocessing Pipeline

### **Phase 1: Data Quality Assessment** ✅ **COMPLETED**

#### 1.1 Data Profiling ✅
```python
# ✅ Analyzed each CSV file
- ✅ Record count per file: 2,830,743 total records across 8 files
- ✅ Missing value analysis: <1% missing values
- ✅ Data type verification: 79 features per file
- ✅ Feature distribution analysis: Completed
- ✅ Label distribution: BENIGN vs attacks analyzed
```

#### 1.2 Initial Quality Checks ✅
```python
# ✅ Completed quality assessment
- ✅ Duplicate records: Identified and quantified
- ✅ Invalid values: Checked for negative durations, impossible packet counts
- ✅ Corrupted data entries: No critical corruption found
- ✅ Overall Quality Score: 90.0/100 (Good)
- ✅ Critical Issues: 0 found
- ✅ Warnings: 16 minor issues identified
```

**Phase 1 Results:**
- **Dataset Size**: 2.8M+ records across 8 files
- **Feature Count**: 79 network flow features
- **Data Quality**: Good (90/100 score)
- **Critical Issues**: None
- **Status**: ✅ **READY FOR PHASE 2**

### **Phase 2: Data Cleaning** ✅ **COMPLETED**

#### 2.1 Handle Missing Values ✅
```python
# ✅ Completed missing value treatment
- ✅ Records with >20% missing features: Removed
- ✅ Numerical features: Median imputation applied
- ✅ Categorical features: Mode imputation applied
- ✅ Infinite values: Replaced with NaN before imputation
- ✅ Total missing values handled: 1,358
```

#### 2.2 Remove Invalid Records ✅
```python
# ✅ Completed invalid record removal
- ✅ Negative values in non-negative features: 1,001,274 records removed
- ✅ Invalid port numbers: Validated and cleaned
- ✅ Inconsistent flow data: Zero duration with packets removed
- ✅ Logical validation: Applied to all network flow features
```

#### 2.3 Remove Duplicates ✅
```python
# ✅ Completed duplicate removal
- ✅ Exact duplicates: Identified and removed
- ✅ Near-duplicates: Removed based on key features
- ✅ Total duplicates removed: 206,797 records
- ✅ Key features used: Destination Port, Flow Duration, Packet counts
```

#### 2.4 Data Type Corrections ✅
```python
# ✅ Completed data type fixes
- ✅ String to numeric conversions: Applied where needed
- ✅ Integer downcasting: Optimized memory usage
- ✅ Float precision: Maintained for decimal features
- ✅ Total corrections made: 512 data type fixes
```

**Phase 2 Results:**
- **Original Records**: 2,830,743
- **Cleaned Records**: 1,829,469
- **Data Retention**: 64.63%
- **Quality Improvement**: Significant
- **Status**: ✅ **READY FOR PHASE 3**

### **Phase 3: Feature Engineering** ✅ **COMPLETED**

#### 3.1 Feature Selection ✅
```python
# ✅ Completed feature selection
- ✅ Constant features: 9-11 removed per file (zero variance)
- ✅ Highly correlated features: 23-29 removed per file (>0.95 correlation)
- ✅ Top features selected: 30 most important features per file
- ✅ Selection method: Unsupervised variance-based + supervised (when labels available)
```

#### 3.2 Feature Transformation ✅
```python
# ✅ Completed feature transformations
- ✅ Log transformation: Applied to heavily skewed features
- ✅ Square root transformation: Applied to moderately skewed features
- ✅ Skewness handling: Features with |skew| > 2.0 transformed
- ✅ New transformed features: Created for optimal ML performance
```

#### 3.3 Feature Creation ✅
```python
# ✅ Completed engineered feature creation
- ✅ Packet_Size_Variance: Squared packet length standard deviation
- ✅ Flow_Efficiency: Bytes per second calculation
- ✅ Burstiness_Index: Packet burst intensity measurement
- ✅ Symmetry_Ratio: Forward/backward packet balance
- ✅ Packet_Size_Ratio: Max/min packet size relationship
- ✅ Flow_Intensity: Packets per microsecond
- ✅ Active_Time_Ratio: Active vs idle time proportion
- ✅ Flag_Activity_Score: Sum of TCP flag counts
- ✅ IAT_CV: Inter-arrival time coefficient of variation
- ✅ Throughput_Ratio: Forward vs backward throughput
- ✅ Total new features: 16 engineered features per file
```

#### 3.4 Feature Scaling ✅
```python
# ✅ Completed feature scaling
- ✅ Scaling method: MinMaxScaler (0-1 range)
- ✅ Autoencoder compatibility: Optimized for neural network training
- ✅ Numeric features: All scaled to [0,1] range
- ✅ Categorical features: Preserved in original form
- ✅ Scaler objects: Saved for consistent future transformations
```

**Phase 3 Results:**
- **Original Features**: 632 across all files
- **Final Features**: 648 (after adding engineered features)
- **New Features Created**: 16 per file
- **Average Features per File**: 81.0
- **Feature Optimization**: Moderate (good balance)
- **Status**: ✅ **READY FOR PHASE 4**

### **Phase 4: Label Processing** ✅ **COMPLETED (Enhanced)**

#### 4.1 Binary Classification ✅
```python
# ✅ Completed binary classification conversion
- ✅ Multi-class to binary: Normal (0) vs Anomaly (1)
- ✅ Attack categories mapped: 27 different attack types
- ✅ Original labels preserved: BENIGN and various attack types
- ✅ Binary mapping: {'Normal': 0, 'Anomaly': 1}
- ✅ Label extraction: From original raw CICIDS2017 data
```

#### 4.2 Attack Category Classification ✅
```python
# ✅ Completed attack category classification
- ✅ Attack categories: Normal, DoS, PortScan, WebAttack, Infiltration, Botnet, BruteForce, Heartbleed, Other
- ✅ Category encoding: LabelEncoder for numeric representation
- ✅ Multi-class support: 9 attack categories + Normal
- ✅ Category mapping: String to numeric conversion
- ✅ Attack type preservation: Original attack types maintained
```

#### 4.3 Two-Stage Classification System ✅
```python
# ✅ Implemented two-stage classification system
- ✅ Stage 1: Binary classification (Normal vs Anomaly)
- ✅ Stage 2: Attack category classification (if anomaly detected)
- ✅ Stage 3: Attack type classification (detailed attack identification)
- ✅ Hierarchical approach: Progressive detail based on detection confidence
- ✅ Lookup functions: Post-processing attack type identification
```

#### 4.4 Attack Type Lookup Functions ✅
```python
# ✅ Created attack type lookup functions
- ✅ Category lookup: get_attack_category(category_id)
- ✅ Attack type lookup: get_attack_type(attack_type_id)
- ✅ Two-stage classification: two_stage_classification()
- ✅ Result interpretation: interpret_anomaly_result()
- ✅ Mapping files: JSON mappings for easy reference
```

#### 4.5 Final Dataset Preparation ✅
```python
# ✅ Enhanced final dataset preparation
- ✅ Multiple target columns: Binary_Label, Attack_Category, Attack_Category_Numeric, Attack_Type_Numeric
- ✅ Flexible usage: Support for binary, multi-class, and hierarchical classification
- ✅ Lookup functions: Post-processing attack identification
- ✅ Data format: Ready for ML algorithms and federated learning
- ✅ File naming: processed_*.csv format with 4 target columns
```

**Phase 4 Results:**
- **Files Processed**: 8 datasets with enhanced labels
- **Total Samples**: 1,622,672
- **Normal Samples**: 1,174,675 (72.39%)
- **Anomaly Samples**: 447,997 (27.61%)
- **Attack Categories**: 9 categories + Normal
- **Attack Types**: 27 specific attack types
- **Target Columns**: 4 (Binary, Category, Category_Numeric, Type_Numeric)
- **ML Ready**: ✅ Yes (for all classification types)
- **Status**: ✅ **PREPROCESSING COMPLETE (ENHANCED)**

## 🎯 **Enhanced Data Preprocessing Pipeline Status**

### **All Phases Completed Successfully** ✅
1. **✅ Phase 1**: Data Quality Assessment - 90/100 quality score
2. **✅ Phase 2**: Data Cleaning - 64.63% data retention
3. **✅ Phase 3**: Feature Engineering - 81 features per file
4. **✅ Phase 4**: Label Processing - Enhanced with attack categories and two-stage classification

### **Final Dataset Statistics**
- **Original Records**: 2,830,743
- **Final Processed Records**: 1,622,672
- **Overall Data Retention**: 57.35%
- **Features per Dataset**: ~81 engineered features
- **Target Variables**: 4 columns (Binary, Category, Category_Numeric, Type_Numeric)
- **Attack Categories**: 9 + Normal
- **Specific Attack Types**: 27
- **Anomaly Rate**: 27.61%

### **🚀 Ready for Advanced Machine Learning**
Your CICIDS2017 dataset is now **fully enhanced** and ready for:
- **Binary Classification**: Normal vs Anomaly detection
- **Multi-Class Classification**: Attack category identification
- **Two-Stage Classification**: Hierarchical anomaly detection
- **Autoencoder Training**: Anomaly detection with attack type identification
- **Federated Learning**: Distributed training with detailed attack classification
- **Post-Processing**: Attack type lookup and interpretation

### **Generated Files**
- **Processed Data**: `data_preprocessing/processed_data/processed_*.csv` (4 target columns)
- **Lookup Functions**: `data_preprocessing/lookup_functions/`
  - `attack_category_lookup.py` - Category identification functions
  - `attack_type_lookup.py` - Attack type identification functions
  - `two_stage_classification.py` - Complete two-stage system
  - `mappings.json` - All label mappings
- **Results**: `phase4_label_processing_results_*.json`
- **Summary**: `phase4_summary_report_*.txt`
# Split data for federated clients
- Strategy 1: Time-based splitting (different days for different clients)
- Strategy 2: Feature-based splitting (different traffic types)
- Strategy 3: Random stratified splitting
- Ensure each client has both normal and anomaly samples
```

#### 6.2 Client Data Balancing
```python
# Balance anomaly distribution
- Each client: 5-10% anomalies (realistic)
- Maintain class balance across clients
- Ensure sufficient training samples per client
```

#### 6.3 Data Format Standardization
```python
# Save in optimized format
- NPZ files for efficient loading
- Include metadata (feature names, scaling parameters)
- Version control for data consistency
```

---

## 🛠️ Implementation Plan

### **Step 1: Create Preprocessing Pipeline**
```python
# File: data_preprocessing.py
class NetworkDataPreprocessor:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
    def load_raw_data(self):
        """Load all CSV files"""
        
    def clean_data(self, df):
        """Apply cleaning operations"""
        
    def engineer_features(self, df):
        """Create new features"""
        
    def normalize_data(self, df):
        """Apply scaling"""
        
    def split_for_federated(self, df, n_clients):
        """Split data for clients"""
        
    def save_processed_data(self, client_data):
        """Save in NPZ format"""
```

### **Step 2: Data Quality Validation**
```python
# File: data_validation.py
def validate_processed_data(data_dir):
    """Check processed data quality"""
    - Feature ranges (0-1)
    - Label distribution
    - Missing values check
    - Data integrity verification
```

### **Step 3: Integration with Existing System**
```python
# Update federated_anomaly_detection/utils/data_utils.py
def load_network_data(node_id, data_dir='data'):
    """Load preprocessed network data"""
    # Use new preprocessed data
```

---

## 📈 Expected Outcomes

### **Data Quality Metrics**
- **Missing Values**: <1% after imputation
- **Feature Count**: 20-30 optimized features
- **Anomaly Ratio**: 5-10% per client
- **Data Consistency**: 100% validated

### **Performance Benefits**
- **Faster Training**: Optimized features reduce training time
- **Better Detection**: Quality features improve anomaly detection
- **Scalability**: Efficient NPZ format for quick loading
- **Reproducibility**: Standardized preprocessing pipeline

---

## 🚀 Execution Checklist

### **Preprocessing Tasks**
- [ ] Analyze raw data quality and characteristics
- [ ] Implement data cleaning pipeline
- [ ] Perform feature engineering and selection
- [ ] Apply appropriate normalization
- [ ] Convert labels to binary classification
- [ ] Split data for federated learning clients
- [ ] Save processed data in NPZ format
- [ ] Validate processed data quality

### **Integration Tasks**
- [ ] Update data loading utilities
- [ ] Test with existing model pipeline
- [ ] Validate federated learning compatibility
- [ ] Update documentation

### **Quality Assurance**
- [ ] Cross-validate preprocessing results
- [ ] Test with sample model training
- [ ] Verify anomaly detection performance
- [ ] Document preprocessing parameters

---

## 📝 Next Steps

1. **Implement Preprocessing Pipeline**: Create the complete preprocessing code
2. **Run Data Analysis**: Execute preprocessing on all 8 CSV files
3. **Validate Results**: Ensure data quality and format consistency
4. **Update System**: Integrate with existing federated learning pipeline
5. **Test Performance**: Validate with autoencoder training

---

**Timeline Estimate**: 2-3 days for complete preprocessing implementation  
**Priority**: High - Critical for model performance and reliability  
**Dependencies**: Python libraries (pandas, numpy, scikit-learn)
