Nice idea 👍 — here’s a **quick-reference list** you can come back to anytime.

---

## Autoencoder Terms & Simple Meanings

### **Autoencoder**

A neural network that learns to **compress data and rebuild it**.

---

### **Input Features**

The original values given to the model (e.g., 81 numbers describing data).

---

### **Encoder**

The part of the network that **shrinks the data** and keeps important information.

---

### **Latent Space / Bottleneck**

A very small set of values (e.g., 4 numbers) that **summarize the entire input**.

---

### **Decoder**

The part of the network that **reconstructs the original data** from the compressed form.

---

### **Reconstruction**

The output produced by the autoencoder that tries to **match the input**.

---

### **Reconstruction Error**

A number that shows **how different the output is from the input**.

---

### **Low Reconstruction Error**

Input is well understood by the model → **normal data**.

---

### **High Reconstruction Error**

Input is poorly reconstructed → **abnormal or unusual data**.

---

### **Anomaly**

A data point that **does not follow normal patterns**.

---

### **Anomaly Detection**

Finding unusual data using **high reconstruction error**.

---

### **Feature Extraction**

Using bottleneck values as **new, meaningful features**.

---

### **Noise**

Random or useless information in data.

---

### **Denoising**

Removing noise by keeping only **important patterns**.

---

### **Dimensionality Reduction**

Reducing many features (81) into **fewer features** (4) without losing key information.

---

### **Latent Representation**

Another name for the bottleneck output; a **compact summary** of the data.

---

### **Training (Normal Data Only)**

Teaching the model what **normal patterns** look like.

---

### **Threshold**

A chosen limit:

* error below → normal
* error above → anomaly

---

### **Loss Function (e.g., MSE)**

How the model measures **reconstruction error** during training.

---

## Ultra-short memory trick 🧠

> Compress → Learn normal → Rebuild → Compare → Large error = anomaly

Accuracy: 64% of all predictions are correct
Precision: 81% of predicted anomalies are actually anomalies
Recall: 59% of actual anomalies are detected
F1-Score: 68% balanced performance (harmonic mean of precision and recall)
ROC-AUC: 69% ability to distinguish between normal and anomaly


Model Architecture: Autoencoder (79→128→64→32→64→128→79)
Parameters: 41,967 trainable parameters
Optimizer: AdamW with learning rate scheduling
Data Features: 79 network traffic features per sample
Anomaly Detection: Reconstruction error-based with 95th percentile threshold


Optimized checkpoint saved: checkpoints\round_5_optimized_model.npz
================================================================================

INFO :      configure_evaluate: strategy sampled 2 clients (out of 3)     
INFO :      aggregate_evaluate: received 2 results and 0 failures
INFO :      
INFO :      [SUMMARY]
INFO :      Run finished 5 round(s) in 149.34s
INFO :          History (loss, distributed):
INFO :                  round 1: 0.0947692058980465
INFO :                  round 2: 0.08319421857595444
INFO :                  round 3: 0.06086484715342522
INFO :                  round 4: 0.05834953859448433
INFO :                  round 5: 0.05542417988181114
INFO :          History (metrics, distributed, fit):
INFO :          {'train_loss': [(1, 0.0028893979997102483),
INFO :                          (2, 0.0005807760731161882),
INFO :                          (3, 0.00039940291470459975),
INFO :                          (4, 0.00032612964047390664),
INFO :                          (5, 0.0002803767166983259)]}
INFO :          History (metrics, distributed, evaluate):
INFO :          {'accuracy': [(1, 0.73525),
INFO :                        (2, 0.6619999999999999),
INFO :                        (3, 0.69425),
INFO :                        (4, 0.6977500000000001),
INFO :                        (5, 0.71025)],
INFO :           'f1_score': [(1, 0.44767356268665853),
INFO :                        (2, 0.2932124582413904),
INFO :                        (3, 0.3624308640547509),
INFO :                        (4, 0.36974277686692564),
INFO :                        (5, 0.39138673946523694)],
INFO :           'precision': [(1, 0.53625),
INFO :                         (2, 0.350625),
INFO :                         (3, 0.43374999999999997),
INFO :                         (4, 0.4425),
INFO :                         (5, 0.46625)],
INFO :           'recall': [(1, 0.3842188157499249),
INFO :                      (2, 0.2519685134726007),
INFO :                      (3, 0.3112593928464082),
INFO :                      (4, 0.3175393247169622),
INFO :                      (5, 0.3372399384443522)],
INFO :           'threshold': [(1, 0.0014979529427364472),
INFO :                         (2, 0.001331554271746427),
INFO :                         (3, 0.0010293317667674274),
INFO :                         (4, 0.0010192248097155242),
INFO :                         (5, 0.0010267522186040879)],
INFO :           'val_loss': [(1, 0.09476920461654663),
INFO :                        (2, 0.08319422030448914),
INFO :                        (3, 0.06086484643816948),
INFO :                        (4, 0.05834953707456589),
INFO :                        (5, 0.055424180358648295)]}
INFO :

🎉 Optimized server completed 5 rounds successfully!
📊 Checkpoints saved in 'checkpoints/' directory
📈 Enhanced training history available for analysis

🏆 FINAL PERFORMANCE SUMMARY:
   📊 Best Accuracy: 0.00%
   🎯 Best Precision: 0.00%
   🔍 Best Recall: 0.00%
   📈 Best F1-Score: 0.00%
PS C:\Users\prave\Desktop\Research Paper\FL, XAI\work\CICD  project\AI\federated_anomaly_detection>