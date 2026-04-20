import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Client data
clients = [
    "Client 1",
    "Client 2", 
    "Client 3",
    "Client 4",
    "Client 5",
    "Client 6",
    "Client 7",
    "Client 8"
]

samples = [191000, 155000, 92000, 292000, 154000, 85000, 231000, 420000]
anomaly_rates = [65, 42, 0.3, 0, 0, 2.5, 4.7, 58]

# Create a single graph with dual y-axes
fig, ax1 = plt.subplots(figsize=(16, 8))

# Color map for anomaly rates
colors = plt.cm.RdYlBu_r(np.array(anomaly_rates) / 100)

# Plot sample counts as bars
bars1 = ax1.bar(clients, samples, color=colors, alpha=0.7, width=0.6)
ax1.set_xlabel('Clients', fontsize=12)
ax1.set_ylabel('Number of Samples', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars for sample counts
for bar, value in zip(bars1, samples):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'{value:,}', ha='center', va='bottom', fontsize=9, color='blue')

# Create second y-axis for anomaly rates
ax2 = ax1.twinx()
line = ax2.plot(clients, anomaly_rates, 'ro-', linewidth=2, markersize=8, label='Anomaly Rate')
ax2.set_ylabel('Anomaly Rate (%)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylim(0, 80)  # Reduce the scale of anomaly rate

# Add value labels for anomaly rates
for i, (client, rate) in enumerate(zip(clients, anomaly_rates)):
    ax2.text(i, rate + max(anomaly_rates)*0.02, f'{rate}%', 
             ha='center', va='bottom', fontsize=9, color='red')

# Add title and legend
plt.title('Federated Learning Client Analysis: Sample Counts and Anomaly Rates', 
          fontsize=14, fontweight='bold')
ax1.legend(['Sample Count'], loc='upper left')
ax2.legend(['Anomaly Rate'], loc='upper right')

# # Add colorbar for anomaly rate coloring
# sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=0, vmax=100))
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.15, aspect=30)
# cbar.set_label('Anomaly Rate (%)', fontsize=12)

plt.tight_layout()
plt.savefig('client_single_graph.png', dpi=300, bbox_inches='tight')
plt.show()

print("Single graph visualization complete! File saved:")
print("- client_single_graph.png (single graph with both metrics)")

# Create a combined scatter plot
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(samples, anomaly_rates, c=anomaly_rates, cmap='RdYlBu_r', 
                    s=100, alpha=0.7, vmin=0, vmax=100)

# Add labels for each point
for i, (client, sample, rate) in enumerate(zip(clients, samples, anomaly_rates)):
    ax.annotate(client.split('(')[1].replace(')', ''), 
                (sample, rate), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8)

ax.set_xlabel('Number of Samples', fontsize=12)
ax.set_ylabel('Anomaly Rate (%)', fontsize=12)
ax.set_title('Client Distribution: Sample Count vs Anomaly Rate', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# # Add colorbar
# cbar = plt.colorbar(scatter, ax=ax)
# cbar.set_label('Anomaly Rate (%)', fontsize=12)

plt.tight_layout()
plt.savefig('client_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization complete! Files saved:")
print("- client_analysis.png (bar charts)")
print("- client_scatter.png (scatter plot)")
