# Flower SuperLink Migration Guide

This guide explains how to use the new Flower SuperLink architecture for federated learning with the anomaly detection system.

## What's New

- **Modern Architecture**: Uses Flower's new SuperLink architecture for better scalability and reliability.
- **Improved Error Handling**: More robust handling of connection issues and client failures.
- **Better Resource Management**: More efficient use of system resources.
- **Deprecation Notice**: The old `server.py` is now deprecated in favor of the new SuperLink implementation.

## Quick Start

### 1. Start the SuperLink Server

```bash
# Navigate to the project root
cd federated_anomaly_detection

# Start the SuperLink server
python -m federated_anomaly_detection.server.superlink_config 
    --input-dim 9 
    --min-clients 2 
    --num-rounds 10 
    --address 0.0.0.0:8080
```

### 2. Start Multiple Clients

In separate terminal windows, start each client with a unique ID:

```bash
# Client 1
 python -m federated_anomaly_detection.client.superlink_client --client-id 1 --server-address 0.0.0.0:8080 --input-dim 9 --data-dir data/processed/client_1.npz

# Client 2
python -m federated_anomaly_detection.client.superlink_client 
    --client-id 2 
    --server-address 0.0.0.0:8080 
    --input-dim 9 
    --data-dir data/processed/client_2

# Add more clients as needed...
```

## Command Line Arguments

### Server Arguments

- `--input-dim`: Number of features in the input data (default: 9)
- `--min-clients`: Minimum number of available clients required for training (default: 2)
- `--num-rounds`: Number of federated learning rounds (default: 10)
- `--address`: Server address in the format `host:port` (default: 0.0.0.0:8080)

### Client Arguments

- `--client-id`: A unique identifier for this client (required)
- `--server-address`: Address of the SuperLink server (default: 0.0.0.0:8080)
- `--input-dim`: Number of features in the input data (default: 9)
- `--data-dir`: Directory containing the client's data (default: data/processed)
- `--batch-size`: Batch size for training (default: 32)
- `--test-batch-size`: Batch size for testing (default: 1000)
- `--learning-rate`: Learning rate for the optimizer (default: 0.001)

## Migration Notes

### From Old to New

1. **Server Initialization**
   - Old: `python -m federated_anomaly_detection.server.server`
   - New: `python -m federated_anomaly_detection.server.superlink_config`

2. **Client Initialization**
   - Old: `python -m federated_anomaly_detection.client.client`
   - New: `python -m federated_anomaly_detection.client.superlink_client`

3. **Configuration**
   - The new implementation uses command-line arguments instead of environment variables.
   - The data directory structure remains the same.

## Troubleshooting

1. **Connection Issues**
   - Ensure the server is running before starting clients.
   - Check that the server address and port are correct.
   - Verify that firewalls allow the specified port.

2. **Performance**
   - For better performance, adjust batch sizes based on available memory.
   - Monitor system resources to prevent out-of-memory errors.

3. **Logs**
   - Check the console output for detailed error messages.
   - Logs are saved in the `logs/` directory.

## Best Practices

1. **Client IDs**
   - Use unique, consistent IDs for each client.
   - Avoid reusing IDs across different training sessions.

2. **Data Distribution**
   - Ensure each client has access to its own data directory.
   - Balance the data distribution across clients for better model performance.

3. **Monitoring**
   - Monitor training progress through the console output.
   - Check the `logs/` directory for detailed training logs.
