#!/usr/bin/env python3
"""
Run maximized federated learning training with 8 clients and 1.6M+ samples
"""
import subprocess
import time
import sys
import os
from pathlib import Path

def run_maximized_training():
    """Run the maximized federated learning training"""
    print("🚀 MAXIMIZED FEDERATED LEARNING TRAINING")
    print("=" * 60)
    print("📊 Dataset: 1,622,672 samples across 8 clients")
    print("🎯 Goal: Train precision-optimized anomaly detection")
    print("=" * 60)
    
    # Change to the correct directory
    fl_dir = Path("AI/federated_anomaly_detection")
    if not fl_dir.exists():
        print(f"❌ Directory not found: {fl_dir}")
        return 1
    
    os.chdir(fl_dir)
    
    try:
        # Start the maximized server
        print("\n🖥️  Starting MAXIMIZED server...")
        server_cmd = [
            "python", "optimized_server.py",
            "--input_dim", "79",
            "--min_clients", "8",  # 8 clients now
            "--num_rounds", "5",   # Start with 5 rounds
            "--address", "localhost:8080"
        ]
        
        print(f"🔧 Command: {' '.join(server_cmd)}")
        server_process = subprocess.Popen(server_cmd)
        
        # Give server time to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Start 8 clients
        client_processes = []
        for client_id in range(1, 9):  # Clients 1-8
            print(f"\n👤 Starting Client {client_id}...")
            client_cmd = [
                "python", "optimized_client.py",
                "--client-id", str(client_id),
                "--data-dir", "data/maximized",  # Use maximized data
                "--server-address", "localhost:8080"
            ]
            
            print(f"🔧 Command: {' '.join(client_cmd)}")
            client_process = subprocess.Popen(client_cmd)
            client_processes.append(client_process)
            
            # Stagger client starts
            time.sleep(2)
        
        print(f"\n🎯 All 8 clients started!")
        print("📊 Training with 1,622,672 total samples...")
        print("🚂 Training samples per round: 1,298,135")
        print("⏱️  Expected training time: ~10-15 minutes")
        
        # Wait for server to complete
        server_process.wait()
        
        print("\n🎉 MAXIMIZED TRAINING COMPLETED!")
        print("📊 Check 'checkpoints/' for saved models")
        print("📈 Training history available for analysis")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_maximized_training()
    sys.exit(exit_code)
