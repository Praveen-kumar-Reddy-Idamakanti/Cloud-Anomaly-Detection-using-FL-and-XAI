import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "model_artifacts"
FED_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "federated_anomaly_detection"
    / "checkpoints"
    / "round_5_optimized_model.npz"
)
FED_DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "federated_anomaly_detection"
    / "data"
    / "maximized"
)


def _load_federated_global_model() -> torch.nn.Module:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "federated_anomaly_detection"))
    from shared_model import create_shared_model  # noqa: E402

    if not FED_CHECKPOINT.exists():
        raise FileNotFoundError(f"Federated checkpoint not found: {FED_CHECKPOINT}")

    npz = np.load(FED_CHECKPOINT)
    arrays = [npz[k] for k in npz.files]

    if len(arrays) == 0:
        raise ValueError("Federated checkpoint npz contains no arrays")

    input_dim = int(arrays[0].shape[1])
    model = create_shared_model(input_dim=input_dim)

    state_keys = list(model.state_dict().keys())
    if len(state_keys) != len(arrays):
        raise ValueError(
            f"Checkpoint parameter count mismatch: model has {len(state_keys)} tensors, checkpoint has {len(arrays)}"
        )

    state_dict = {k: torch.tensor(v) for k, v in zip(state_keys, arrays)}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _collect_federated_val_errors(model: torch.nn.Module, device: torch.device):
    client_files = sorted(FED_DATA_DIR.glob("client_*.npz"))
    if not client_files:
        raise FileNotFoundError(f"No client_*.npz files found in {FED_DATA_DIR}")

    all_errors: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for client_file in client_files:
            data = np.load(client_file)
            X = data["val_features"].astype(np.float32)
            y = data.get("val_labels")
            if y is None:
                y = np.zeros((X.shape[0],), dtype=np.int64)
            y = np.asarray(y).astype(np.int64)

            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
            dl = DataLoader(ds, batch_size=1024, shuffle=False)

            for xb, yb in dl:
                xb = xb.to(device)
                recon = model(xb)
                err = torch.mean((recon - xb) ** 2, dim=1)
                all_errors.append(err.detach().cpu().numpy())
                all_labels.append(yb.detach().cpu().numpy())

    errors = np.concatenate(all_errors)
    labels = np.concatenate(all_labels)
    return errors, labels


def _threshold_candidates(errors: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(errors))
    std = float(np.std(errors))
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))

    return {
        "95th_percentile": float(np.percentile(errors, 95)),
        "mean_plus_2std": mean + 2.0 * std,
        "mean_plus_3std": mean + 3.0 * std,
        "median_plus_mad": median + 3.0 * mad,
    }


def _evaluate_threshold(labels: np.ndarray, errors: np.ndarray, threshold: float):
    preds = (errors > threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1_score": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def main() -> int:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_federated_global_model().to(device)
    errors, labels = _collect_federated_val_errors(model, device=device)

    try:
        roc_auc = float(roc_auc_score(labels, errors))
    except Exception:
        roc_auc = 0.5

    candidates = _threshold_candidates(errors)
    all_results = {name: _evaluate_threshold(labels, errors, thr) for name, thr in candidates.items()}
    best_name = max(all_results.keys(), key=lambda k: all_results[k]["f1_score"])

    federated_metrics = {
        "evaluation": {
            "dataset": "federated_client_validation (data/maximized/*.npz)",
            "num_samples": int(len(labels)),
            "anomaly_rate": float(np.mean(labels) * 100.0),
        },
        "model_info": {
            "checkpoint": str(FED_CHECKPOINT),
            "input_dim": int(errors.shape[0] and model.encoder[0].in_features),
        },
        "roc_auc": roc_auc,
        "best_threshold_method": best_name,
        "best_metrics": all_results[best_name],
        "all_thresholds": all_results,
        "reconstruction_error_stats": {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "min": float(np.min(errors)),
            "max": float(np.max(errors)),
        },
    }

    fed_out = ARTIFACTS_DIR / "federated_baseline_metrics.json"
    with open(fed_out, "w", encoding="utf-8") as f:
        json.dump(federated_metrics, f, indent=2)

    centralized_metrics_path = ARTIFACTS_DIR / "classification_metrics.json"
    centralized_metrics = None
    if centralized_metrics_path.exists():
        with open(centralized_metrics_path, "r", encoding="utf-8") as f:
            centralized_metrics = json.load(f)

    comparison = {
        "centralized": centralized_metrics,
        "federated": federated_metrics,
    }

    comp_out = ARTIFACTS_DIR / "centralized_vs_federated_comparison.json"
    with open(comp_out, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    cm = np.array(federated_metrics["best_metrics"]["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.title("Federated Global Model (Best Threshold)")
    plt.ylabel("True")
    plt.xlabel("Pred")

    plt.subplot(1, 2, 2)
    labels_bar = ["Precision", "Recall", "F1", "ROC-AUC"]

    fed_vals = [
        federated_metrics["best_metrics"]["precision"],
        federated_metrics["best_metrics"]["recall"],
        federated_metrics["best_metrics"]["f1_score"],
        federated_metrics["roc_auc"],
    ]

    cen_vals = None
    if centralized_metrics and "best_metrics" in centralized_metrics:
        cen_vals = [
            float(centralized_metrics["best_metrics"]["precision"]),
            float(centralized_metrics["best_metrics"]["recall"]),
            float(centralized_metrics["best_metrics"]["f1_score"]),
            float(centralized_metrics["best_metrics"]["roc_auc"]),
        ]

    x = np.arange(len(labels_bar))
    width = 0.35

    plt.bar(x - width / 2, fed_vals, width, label="Federated")
    if cen_vals is not None:
        plt.bar(x + width / 2, cen_vals, width, label="Centralized")

    plt.xticks(x, labels_bar)
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.title("Centralized vs Federated (Stage-1 Metrics)")
    plt.legend()

    plt.tight_layout()
    plot_out = ARTIFACTS_DIR / "federated_baseline_metrics.png"
    plt.savefig(plot_out, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Saved federated baseline metrics: {fed_out}")
    print(f"✅ Saved comparison JSON: {comp_out}")
    print(f"✅ Saved plot: {plot_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
