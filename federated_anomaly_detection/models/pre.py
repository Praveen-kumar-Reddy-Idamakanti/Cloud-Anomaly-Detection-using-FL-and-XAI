import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report

# ---------- Step 1: Load dataset ----------
folder = r"C:\Users\prave\Desktop\Research Paper\FL, XAI\work\main_project\data\raw\BETH"
print("Files in BETH folder:")
print(os.listdir(folder))

df = pd.read_csv(
    r"C:\Users\prave\Desktop\Research Paper\FL, XAI\work\main_project\data\raw\BETH\labelled_2021may-ip-10-100-1-105-dns.csv"
)

print("Columns in dataset:", df.columns.tolist())
print(df.head(3))

# ---------- Step 2.1: Fix column names ----------
df = df.rename(columns={
    "processNa": "processName",
    "eventNam": "eventName",
    "returnValu": "returnValue"
})

# ---------- Step 2.2: Handle timestamp ----------
time_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
if time_candidates:
    time_col = time_candidates[0]  # pick first match
    print(f"Using time column: {time_col}")
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")

    # Extract features
    df["hour"] = df["timestamp"].dt.hour.fillna(0)
    df["day"] = df["timestamp"].dt.dayofweek.fillna(0)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)

    df = df.drop(columns=["timestamp", "hour", "day"])
else:
    print("⚠ No timestamp column found, skipping time features.")

# ---------- Step 2.3: Numeric columns ----------
scaler = StandardScaler()
for col in ["argsNum", "returnValue"]:
    if col in df.columns:
        df[[col]] = scaler.fit_transform(df[[col]])
    else:
        print(f"⚠ Column {col} not found, skipping scaling.")

# ---------- Step 2.4: Categorical columns ----------
categorical_cols = ["processId", "parentProcId", "userId",
                    "processName", "hostName", "eventId", "eventName"]

encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    else:
        print(f"⚠ Column {col} not found, skipping encoding.")

# ---------- Step 2.5: Args column ----------
if "args" in df.columns:
    def extract_args_features(arg):
        text = str(arg) if pd.notna(arg) else ""
        return pd.Series({
            "args_len": len(text),
            "args_num_tokens": len(text.split()),
            "args_has_ip": 1 if re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text) else 0,
            "args_has_path": 1 if ("/" in text or "\\" in text) else 0
        })

    args_features = df["args"].apply(extract_args_features)
    df = pd.concat([df.drop(columns=["args"]), args_features], axis=1)

    # Scale engineered features
    for col in ["args_len", "args_num_tokens"]:
        df[[col]] = scaler.fit_transform(df[[col]])
else:
    print("⚠ Column 'args' not found, skipping feature extraction.")

# ---------- Step 2.6: Separate Labels ----------
label_col = "sus"  # <-- change to "evil" if you want
if label_col in df.columns:
    y = df[label_col].copy().astype(int)   # ensure integer labels
    X = df.drop(columns=[label_col])
else:
    raise ValueError(f"❌ Label column {label_col} not found in dataset!")

# ---------- Step 2.7: Ensure all features are numeric ----------
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0).astype(np.float32)

print("✅ Final feature matrix shape:", X.shape)
print("✅ Labels shape:", y.shape)

# ---------- Step 3: Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)

# ---------- Step 4: Autoencoder ----------
input_dim = X_train.shape[1]

autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(input_dim, activation="sigmoid")  # reconstruct features
])

autoencoder.compile(optimizer="adam", loss="mse")

history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ---------- Step 5: Anomaly Detection ----------
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Threshold = 95th percentile
threshold = np.percentile(mse, 95)
print("Threshold:", threshold)

# Predictions: anomaly = 1 if mse > threshold
y_pred = (mse > threshold).astype(int)

# ---------- Step 6: Evaluation ----------
print(classification_report(y_test, y_pred, digits=4))
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Autoencoder Training Loss')
plt.legend()
plt.show()
import numpy as np

# Reconstruction error
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

plt.hist(mse[y_test==0], bins=50, alpha=0.6, label="Normal")
plt.hist(mse[y_test==1], bins=50, alpha=0.6, label="Anomaly")
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.show()
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, mse)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
from sklearn.metrics import confusion_matrix
import seaborn as sns

threshold = np.percentile(mse, 95)  # example threshold
y_pred = (mse > threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
