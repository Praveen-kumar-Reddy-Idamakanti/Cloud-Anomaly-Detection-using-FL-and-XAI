"""
federated_anomaly_viz.py

Visualization script for the "Federated Anomaly Detection System - Training Analysis Report"
Generates plots for:
 - training loss progression
 - validation loss progression
 - learning rate schedule
 - anomaly detection threshold
 - anomaly ratio
 - client data distribution
 - combined dashboard (multi-panel)

Usage:
    python federated_anomaly_viz.py

Outputs (PNG files saved to ./outputs):
    training_loss.png
    validation_loss.png
    lr_schedule.png
    threshold.png
    anomaly_ratio.png
    client_distribution.png
    dashboard.png

Requirements:
    - Python 3.8+
    - matplotlib
    - numpy
    - pandas

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- Data extracted from the report ---
rounds = np.arange(1, 21)

training_loss = np.array([
    0.4810, 0.4095, 0.1193, 0.0500, 0.0283, 0.0261, 0.0387, 0.0351, 0.0120, 0.0099,
    0.0486, 0.0151, 0.0061, 0.0136, 0.0673, 0.0139, 0.0129, 0.0659, 0.0071, 0.0147
])

validation_loss = np.array([
    0.4466, 0.3758, 0.1008, 0.0385, 0.0201, 0.0261, 0.0387, 0.0351, 0.0120, 0.0099,
    0.0486, 0.0151, 0.0061, 0.0136, 0.0673, 0.0139, 0.0129, 0.0659, 0.0071, 0.0147
])

learning_rate = np.array([
    0.001000, 0.000880, 0.000500, 0.000813, 0.000362, 0.000500, 0.000500, 0.000250,
    0.000500, 0.000250, 0.000031, 0.000063, 0.000063, 0.000016, 0.000001, 0.000002,
    0.000001, 0.000001, 0.000001, 0.000001
])

threshold = np.array([
    0.1706, 0.1702, 0.0498, 0.0332, 0.0105, 0.0081, 0.0078, 0.0072, 0.0076, 0.0070,
    0.0122, 0.0043, 0.0059, 0.0045, 0.0175, 0.0053, 0.0045, 0.0181, 0.0059, 0.0050
])

anomaly_ratio = np.array([
    4.94, 4.66, 4.94, 4.78, 4.95, 4.98, 4.94, 4.99, 4.99, 4.96,
    4.99, 4.98, 4.90, 4.99, 4.99, 4.92, 4.99, 4.99, 4.97, 4.95
])

clients_samples = {
    'Client 1': 108000,
    'Client 2': 112000,
    'Client 3': 113000,
    'Client 4': 81000,
    'Client 5': 107000,
    'Client 6': 76000,
}

# Overfitting rounds identified in the report
overfitting_rounds = [7, 11, 15, 18, 20]
best_training_round = 6
best_validation_round = 13

# Create output dir
OUTDIR = os.path.join(os.getcwd(), 'outputs')
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

# Utility: save figure helper
def save_fig(fig, name, dpi=150):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    print(f"Saved: {path}")

# 1) Training loss progression
def plot_training_loss():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, training_loss, marker='o')
    ax.set_title('Training Loss Progression (20 Rounds)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Training Loss')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Mark overfitting rounds
    for r in overfitting_rounds:
        val = training_loss[r-1]
        ax.axvline(r, linestyle='--', alpha=0.5)
        ax.annotate('Overfitting', xy=(r, val), xytext=(r+0.2, val*1.5), arrowprops=dict(arrowstyle='->', lw=0.6))

    # Highlight best round
    ax.scatter([best_training_round], [training_loss[best_training_round-1]], s=100, zorder=5)
    ax.annotate('Best (Round {})'.format(best_training_round), xy=(best_training_round, training_loss[best_training_round-1]),
                xytext=(best_training_round+0.5, training_loss[best_training_round-1]*1.2), arrowprops=dict(arrowstyle='->', lw=0.6))

    save_fig(fig, 'training_loss.png')
    plt.close(fig)

# 2) Validation loss progression
def plot_validation_loss():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, validation_loss, marker='o')
    ax.set_title('Validation Loss Progression (20 Rounds)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Validation Loss')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    for r in overfitting_rounds:
        val = validation_loss[r-1]
        ax.axvline(r, linestyle='--', alpha=0.5)

    ax.scatter([best_validation_round], [validation_loss[best_validation_round-1]], s=100, zorder=5)
    ax.annotate('Best (Round {})'.format(best_validation_round), xy=(best_validation_round, validation_loss[best_validation_round-1]),
                xytext=(best_validation_round+0.5, validation_loss[best_validation_round-1]*1.2), arrowprops=dict(arrowstyle='->', lw=0.6))

    save_fig(fig, 'validation_loss.png')
    plt.close(fig)

# 3) Learning rate schedule (log scale)
def plot_learning_rate():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rounds, learning_rate, marker='o')
    ax.set_yscale('log')
    ax.set_title('Learning Rate Schedule (log scale)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Learning Rate (log scale)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    save_fig(fig, 'lr_schedule.png')
    plt.close(fig)

# 4) Anomaly detection threshold
def plot_threshold():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rounds, threshold, marker='o')
    ax.set_title('Anomaly Detection Threshold by Round')
    ax.set_xlabel('Round')
    ax.set_ylabel('Threshold')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # annotate final threshold
    ax.annotate('Final threshold: {:.4f}'.format(threshold[-1]), xy=(rounds[-1], threshold[-1]),
                xytext=(rounds[-4], max(threshold)*0.9), arrowprops=dict(arrowstyle='->', lw=0.6))

    save_fig(fig, 'threshold.png')
    plt.close(fig)

# 5) Anomaly ratio
def plot_anomaly_ratio():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rounds, anomaly_ratio, marker='o')
    ax.set_title('Anomaly Ratio (% of data flagged as anomalous)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Anomaly Ratio (%)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # horizontal line at mean
    mean_ratio = anomaly_ratio.mean()
    ax.axhline(mean_ratio, linestyle='--', alpha=0.6)
    ax.annotate('Mean {:.2f}%'.format(mean_ratio), xy=(1, mean_ratio), xytext=(2, mean_ratio+0.05))

    save_fig(fig, 'anomaly_ratio.png')
    plt.close(fig)

# 6) Client distribution (bar chart)
def plot_client_distribution():
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(clients_samples.keys())
    values = list(clients_samples.values())
    ax.bar(names, values)
    ax.set_title('Per-client Sample Distribution')
    ax.set_ylabel('Number of samples')
    for i, v in enumerate(values):
        ax.annotate(str(v), xy=(i, v), xytext=(0, 4), textcoords='offset points', ha='center')

    save_fig(fig, 'client_distribution.png')
    plt.close(fig)

# 7) Combined dashboard: losses + lr + threshold
def plot_dashboard():
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # a) training & validation loss
    axs[0].plot(rounds, training_loss, marker='o', label='Training')
    axs[0].plot(rounds, validation_loss, marker='o', label='Validation')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss Progression')
    axs[0].legend()

    # mark overfitting rounds
    for r in overfitting_rounds:
        axs[0].axvline(r, linestyle='--', alpha=0.3)

    # b) learning rate (log)
    axs[1].plot(rounds, learning_rate, marker='o')
    axs[1].set_yscale('log')
    axs[1].set_ylabel('Learning Rate (log)')
    axs[1].set_title('Learning Rate Schedule')

    # c) threshold + anomaly ratio (twin axis)
    axs[2].plot(rounds, threshold, marker='o', label='Threshold')
    ax2 = axs[2].twinx()
    ax2.plot(rounds, anomaly_ratio, marker='s', linestyle='--', label='Anomaly Ratio (%)')
    axs[2].set_xlabel('Round')
    axs[2].set_ylabel('Threshold')
    ax2.set_ylabel('Anomaly Ratio (%)')
    axs[2].set_title('Threshold vs Anomaly Ratio')

    plt.tight_layout()
    save_fig(fig, 'dashboard.png')
    plt.close(fig)

# Run all plots
if __name__ == '__main__':
    plot_training_loss()
    plot_validation_loss()
    plot_learning_rate()
    plot_threshold()
    plot_anomaly_ratio()
    plot_client_distribution()
    plot_dashboard()
    print('\nAll visualizations generated in ./outputs')
